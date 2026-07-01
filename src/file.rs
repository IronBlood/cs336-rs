use std::{fs, path::PathBuf};

use crate::{
    encode::{bytes_to_string, string_to_bytes},
    types::TokenBytes,
};

pub fn parse_vocab(content: &str) -> Vec<TokenBytes> {
    content
        .trim_end()
        .lines()
        .map(|line| string_to_bytes(line))
        .collect()
}

pub fn serialize_vocab(vocab: &[TokenBytes]) -> String {
    vocab
        .iter()
        .map(|token| bytes_to_string(token))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn parse_merges(content: &str) -> Vec<(TokenBytes, TokenBytes)> {
    content
        .trim_end()
        .lines()
        .map(|line| {
            let (token_1, token_2) = line
                .split_once(' ')
                .expect("each merge line should contain two tokens");
            let token_1 = string_to_bytes(token_1);
            let token_2 = string_to_bytes(token_2);
            (token_1, token_2)
        })
        .collect()
}

pub fn serialize_merges(merges: &[(TokenBytes, TokenBytes)]) -> String {
    merges
        .iter()
        .map(|(token_1, token_2)| {
            format!("{} {}", bytes_to_string(token_1), bytes_to_string(token_2))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn load_vocab(path: &PathBuf) -> Vec<TokenBytes> {
    let content = fs::read_to_string(&path).expect("file should be readable");

    parse_vocab(&content)
}

pub fn write_vocab(path: &PathBuf, vocab: &[TokenBytes]) {
    fs::write(path, serialize_vocab(vocab)).expect("file should be writeable");
}

pub fn load_merges(path: &PathBuf) -> Vec<(TokenBytes, TokenBytes)> {
    let content = fs::read_to_string(&path).expect("file should be readable");

    parse_merges(&content)
}

pub fn write_merges(path: &PathBuf, merges: &[(TokenBytes, TokenBytes)]) {
    fs::write(path, serialize_merges(merges)).expect("file should be writeable");
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_vocab() {
        let content = "ā\n";
        let vocab = parse_vocab(&content);
        assert_eq!(vocab.len(), 1);
        assert_eq!(vocab[0], vec![1]);
    }

    #[test]
    fn test_serialize_vocab() {
        let mut vocab: Vec<TokenBytes> = Vec::new();
        vocab.push(vec![1]);

        let content = serialize_vocab(&vocab);
        assert_eq!(content, "ā");
    }

    #[test]
    fn test_parse_merges() {
        let content = "Ġ t\nĠ a";
        let merges = parse_merges(&content);
        assert_eq!(merges.len(), 2);
        assert_eq!(merges[0], (vec![32], vec![b't']));
        assert_eq!(merges[1], (vec![32], vec![b'a']));
    }

    #[test]
    fn test_serialize_merges() {
        let mut merges = Vec::new();
        merges.push((vec![32], vec![b't']));
        merges.push((vec![32], vec![b'a']));
        let content = serialize_merges(&merges);
        assert_eq!(content, "Ġ t\nĠ a");
    }
}
