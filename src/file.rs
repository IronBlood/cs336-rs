use std::collections::HashMap;

use crate::{
    encode::{bytes_to_string, string_to_bytes},
    error::CustomError,
    utils::TokenBytes,
};

fn parse_vocab(content: &str) -> Result<HashMap<u16, Vec<u8>>, CustomError> {
    let json: HashMap<String, u16> = serde_json::from_str(&content)?;

    let vocab = json
        .into_iter()
        .map(|(s, idx)| (idx, string_to_bytes(&s)))
        .collect();

    Ok(vocab)
}

fn serialize_vocab(vocab: &HashMap<u16, Vec<u8>>) -> Result<String, CustomError> {
    let serialized_vocab: HashMap<String, u16> = vocab
        .iter()
        .map(|(&idx, bytes)| (bytes_to_string(bytes), idx))
        .collect();

    Ok(serde_json::to_string_pretty(&serialized_vocab)?)
}

fn parse_merges(content: &str) -> Vec<(TokenBytes, TokenBytes)> {
    let merges: Vec<_> = content
        .lines()
        .map(|line| {
            let (token_1, token_2) = line
                .split_once(' ')
                .expect("each merge line should contain two tokens");
            let token_1 = string_to_bytes(token_1);
            let token_2 = string_to_bytes(token_2);
            (token_1, token_2)
        })
        .collect();

    merges
}

fn serialize_merges(merges: &[(TokenBytes, TokenBytes)]) -> String {
    merges
        .iter()
        .map(|(token_1, token_2)| {
            format!("{} {}", bytes_to_string(token_1), bytes_to_string(token_2))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_vocab() {
        let content = "{\"ā\":1}";
        let vocab = parse_vocab(&content).expect("should be valid json");
        assert_eq!(vocab.len(), 1);
        let v = vocab.get(&1);
        assert!(v.is_some());
        let v = v.unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 1);
    }

    #[test]
    fn test_serialize_vocab() {
        let mut vocab: HashMap<u16, Vec<u8>> = HashMap::new();
        vocab.insert(42, vec![1]);

        let json = serialize_vocab(&vocab).expect("should be valid json");
        assert_eq!(json, "{\n  \"ā\": 42\n}");
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
