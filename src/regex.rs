use std::{
    error::Error,
    fmt,
    ptr::{self, NonNull},
};

use super::ffi::{
    PCRE2_ERROR_NOMATCH, Pcre2Code8, Pcre2MatchData8, pcre2_code_free_8, pcre2_compile_8,
    pcre2_get_error_message_8, pcre2_get_ovector_pointer_8, pcre2_match_8,
    pcre2_match_data_create_from_pattern_8, pcre2_match_data_free_8,
};

#[derive(Debug)]
pub enum RegexError {
    Compile {
        code: i32,
        offset: usize,
        message: String,
    },
    Allocation,
    Match(i32),
    InvalidOffsets,
}

impl fmt::Display for RegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile {
                code,
                offset,
                message,
            } => {
                write!(
                    f,
                    "PCRE2 compilation failed at byte {offset}: {message} (error {code})"
                )
            }
            Self::Allocation => write!(f, "PCRE2 allocation failed"),
            Self::Match(code) => write!(f, "PCRE2 matching failed: error {code}"),
            Self::InvalidOffsets => write!(f, "PCRE2 returned invalid match offsets"),
        }
    }
}

impl Error for RegexError {}

pub struct Regex {
    code: NonNull<Pcre2Code8>,
}

impl Regex {
    pub fn new(pattern: &str) -> Result<Self, RegexError> {
        let mut error_code = 0;
        let mut error_offset = 0;

        let code = unsafe {
            pcre2_compile_8(
                pattern.as_ptr(),
                pattern.len(),
                0,
                &mut error_code,
                &mut error_offset,
                ptr::null_mut(),
            )
        };

        let code = NonNull::new(code).ok_or_else(|| RegexError::Compile {
            code: error_code,
            offset: error_offset,
            message: pcre2_error_message(error_code),
        })?;
        Ok(Self { code })
    }

    pub fn find(&self, subject: &[u8]) -> Result<Option<std::ops::Range<usize>>, RegexError> {
        let match_data =
            unsafe { pcre2_match_data_create_from_pattern_8(self.code.as_ptr(), ptr::null_mut()) };

        let mut match_data = MatchData::new(match_data).ok_or(RegexError::Allocation)?;

        let result = unsafe {
            pcre2_match_8(
                self.code.as_ptr(),
                subject.as_ptr(),
                subject.len(),
                0,
                0,
                match_data.as_mut_ptr(),
                ptr::null_mut(),
            )
        };

        if result == PCRE2_ERROR_NOMATCH {
            return Ok(None);
        }

        if result < 0 {
            return Err(RegexError::Match(result));
        }

        let offsets = unsafe {
            let pointer = pcre2_get_ovector_pointer_8(match_data.as_mut_ptr());

            if pointer.is_null() {
                return Err(RegexError::InvalidOffsets);
            }

            std::slice::from_raw_parts(pointer, 2)
        };

        let start = offsets[0];
        let end = offsets[1];

        if start > end || end > subject.len() {
            return Err(RegexError::InvalidOffsets);
        }

        Ok(Some(start..end))
    }
}

impl Drop for Regex {
    fn drop(&mut self) {
        unsafe {
            pcre2_code_free_8(self.code.as_ptr());
        }
    }
}

struct MatchData {
    pointer: NonNull<Pcre2MatchData8>,
}

impl MatchData {
    fn new(pointer: *mut Pcre2MatchData8) -> Option<Self> {
        NonNull::new(pointer).map(|pointer| Self { pointer })
    }

    fn as_mut_ptr(&mut self) -> *mut Pcre2MatchData8 {
        self.pointer.as_ptr()
    }
}

impl Drop for MatchData {
    fn drop(&mut self) {
        unsafe {
            pcre2_match_data_free_8(self.pointer.as_ptr());
        }
    }
}

fn pcre2_error_message(error_code: i32) -> String {
    let mut buffer = [0_u8; 256];
    let length =
        unsafe { pcre2_get_error_message_8(error_code, buffer.as_mut_ptr(), buffer.len()) };
    if length < 0 {
        return format!("unknown PCRE2 error {error_code}");
    }
    String::from_utf8_lossy(&buffer[..length as usize]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_test() {
        let regex = Regex::new(r"a++").expect("Regex should be compiled");

        let m = regex.find(b"caaab").expect("should be executed");
        assert_eq!(m, Some(1..4));

        let m = regex.find(b"xyz").expect("should be executed");
        assert_eq!(m, None);
    }
}
