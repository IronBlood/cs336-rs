use crate::regex::RegexError;

#[derive(Debug)]
pub enum CustomError {
    InvalidUtf8(std::str::Utf8Error),
    InvalidRegex(RegexError),
    ThreadPanic,
}

impl From<std::str::Utf8Error> for CustomError {
    fn from(err: std::str::Utf8Error) -> Self {
        Self::InvalidUtf8(err)
    }
}

impl From<RegexError> for CustomError {
    fn from(err: RegexError) -> Self {
        Self::InvalidRegex(err)
    }
}
