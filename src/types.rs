use std::collections::HashMap;

pub type TokenBytes = Vec<u8>;
pub type TokenIds = Vec<u16>;
pub type Span = (usize, usize);
pub type TokenId = u16;
pub type PackedPair = u32;
pub type BorrowedWordFreqMap<'a> = HashMap<&'a [u8], usize>;
