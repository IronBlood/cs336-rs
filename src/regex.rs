use std::{
    error::Error,
    fmt,
    ops::Range,
    ptr::{self, NonNull},
};

use super::ffi::{
    PCRE2_ERROR_NOMATCH, PCRE2_UCP, PCRE2_UTF, Pcre2Code8, Pcre2MatchData8, pcre2_code_free_8,
    pcre2_compile_8, pcre2_get_error_message_8, pcre2_get_ovector_pointer_8, pcre2_match_8,
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

// A compiled PCRE2 pattern is immutable after construction. Matching state is
// kept in per-call match data, so sharing the compiled pattern across scoped
// worker threads is safe.
unsafe impl Send for Regex {}
unsafe impl Sync for Regex {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    start: usize,
    end: usize,
}

impl Match {
    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn range(&self) -> Range<usize> {
        self.start..self.end
    }
}

impl Regex {
    pub fn new(pattern: &str) -> Result<Self, RegexError> {
        let mut error_code = 0;
        let mut error_offset = 0;

        let code = unsafe {
            pcre2_compile_8(
                pattern.as_ptr(),
                pattern.len(),
                PCRE2_UTF | PCRE2_UCP,
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

    pub fn find(&self, subject: &[u8]) -> Result<Option<Range<usize>>, RegexError> {
        self.find_from(subject, 0)
            .map(|found| found.map(|m| m.range()))
    }

    pub fn find_iter<'regex, 'subject>(
        &'regex self,
        subject: &'subject str,
    ) -> FindIter<'regex, 'subject> {
        FindIter {
            regex: self,
            subject,
            next_start: 0,
            finished: false,
        }
    }

    fn find_from(&self, subject: &[u8], start_offset: usize) -> Result<Option<Match>, RegexError> {
        if start_offset > subject.len() {
            return Err(RegexError::InvalidOffsets);
        }

        let match_data =
            unsafe { pcre2_match_data_create_from_pattern_8(self.code.as_ptr(), ptr::null_mut()) };

        let mut match_data = MatchData::new(match_data).ok_or(RegexError::Allocation)?;

        let result = unsafe {
            pcre2_match_8(
                self.code.as_ptr(),
                subject.as_ptr(),
                subject.len(),
                start_offset,
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

        Ok(Some(Match { start, end }))
    }
}

pub fn escape_literal(literal: &str) -> String {
    format!(r"\Q{}\E", literal.replace(r"\E", r"\E\\E\Q"))
}

pub struct FindIter<'regex, 'subject> {
    regex: &'regex Regex,
    subject: &'subject str,
    next_start: usize,
    finished: bool,
}

impl Iterator for FindIter<'_, '_> {
    type Item = Result<Match, RegexError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let subject = self.subject.as_bytes();
        let found = match self.regex.find_from(subject, self.next_start) {
            Ok(Some(found)) => found,
            Ok(None) => {
                self.finished = true;
                return None;
            }
            Err(err) => {
                self.finished = true;
                return Some(Err(err));
            }
        };

        if found.end > self.next_start {
            self.next_start = found.end;
        } else if self.next_start < subject.len() {
            self.next_start += 1;
        } else {
            self.finished = true;
        }

        Some(Ok(found))
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

    #[test]
    fn escape_literal_should_quote_regex_metacharacters() {
        let pattern = escape_literal(r"a.|b\Ec");
        let regex = Regex::new(&pattern).expect("escaped literal should compile");

        let m = regex.find(br"xxa.|b\Ecyy").expect("should be executed");
        assert_eq!(m, Some(2..9));
    }

    #[test]
    fn unicode_properties_should_match_utf8_codepoints() {
        let regex = Regex::new(r"\p{L}++").expect("Regex should be compiled");

        let text = "é";
        let m = regex.find(text.as_bytes()).expect("should be executed");
        assert_eq!(m, Some(0..text.len()));
    }
}
