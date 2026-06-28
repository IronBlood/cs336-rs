use crate::regex::RegexError;
use std::{error::Error, fmt};

#[derive(Debug)]
pub enum CustomError {
    InvalidUtf8(std::str::Utf8Error),
    InvalidRegex(RegexError),
    ThreadPanic,
}

impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8(err) => write!(f, "invalid UTF-8: {err}"),
            Self::InvalidRegex(err) => write!(f, "invalid regex: {err}"),
            Self::ThreadPanic => write!(f, "worker thread panicked"),
        }
    }
}

impl Error for CustomError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidUtf8(err) => Some(err),
            Self::InvalidRegex(err) => Some(err),
            Self::ThreadPanic => None,
        }
    }
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
