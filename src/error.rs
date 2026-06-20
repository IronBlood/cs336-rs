#[derive(Debug)]
pub enum CustomError {
    InvalidUtf8(std::str::Utf8Error),
    InvalidRegex(regex::Error),
    ThreadPanic,
}

impl From<std::str::Utf8Error> for CustomError {
    fn from(err: std::str::Utf8Error) -> Self {
        Self::InvalidUtf8(err)
    }
}

impl From<regex::Error> for CustomError {
    fn from(err: regex::Error) -> Self {
        Self::InvalidRegex(err)
    }
}
