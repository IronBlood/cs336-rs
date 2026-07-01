mod common;
use std::{
    collections::{HashMap, HashSet},
    fs,
};

use crate::common::{gpt2_bytes_to_unicode, load_merges, load_vocab};
use cs336_rs::tokenizer::Tokenizer;

fn check_vocab(vocab_path: &str) {
    let gpt2_byte_decoder: HashMap<char, u8> = gpt2_bytes_to_unicode()
        .into_iter()
        .map(|(k, v)| (v, k))
        .collect();
    let vocab = load_vocab(vocab_path, &gpt2_byte_decoder).expect("should be a valid json file");

    if vocab.len() >= u16::MAX as usize {
        panic!("vocab is too long");
    }

    for i in 0..vocab.len() {
        assert!(vocab.get(&(i as u16)).is_some());
    }
}

fn get_tokenizer_from_vocab_merges_path(
    vocab_path: &str,
    merges_path: &str,
    special_tokens: Option<&[String]>,
) -> Tokenizer {
    let gpt2_byte_decoder: HashMap<char, u8> = gpt2_bytes_to_unicode()
        .into_iter()
        .map(|(k, v)| (v, k))
        .collect();
    let mut vocab =
        load_vocab(vocab_path, &gpt2_byte_decoder).expect("should be a valid json file");
    let merges = load_merges(merges_path, &gpt2_byte_decoder).expect("should read file");

    if vocab.len() >= u16::MAX as usize {
        panic!("vocab is too long");
    }

    for i in 0..vocab.len() {
        assert!(vocab.get(&(i as u16)).is_some());
    }

    if let Some(special_tokens) = special_tokens {
        let mut unique_bytes: HashSet<_> = vocab.values().cloned().collect();
        for st in special_tokens {
            let st_bytes = st.as_bytes().to_vec();
            if unique_bytes.insert(st_bytes.clone()) {
                if vocab.len() >= u16::MAX as usize {
                    panic!("vocab is too long");
                }
                vocab.insert(vocab.len() as u16, st_bytes.clone());
            }
        }
    }

    let gpt2_regex_str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
    Tokenizer::load_course(vocab, merges, special_tokens, gpt2_regex_str)
        .expect("should build a tokenizer")
}

const VOCAB_PATH: &str = "tests/fixtures/gpt2_vocab.json";
const MERGES_PATH: &str = "tests/fixtures/gpt2_merges.txt";

#[test]
fn test_continues() {
    check_vocab(VOCAB_PATH);
}

#[test]
fn test_roundtrip_empty() {
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, None);

    let test_string = "";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert!(encoded_ids.is_empty());
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);
}

#[test]
fn test_roundtrip_single_character() {
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, None);

    let test_string = "s";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert_eq!(encoded_ids, vec![82]);
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);
}

#[test]
fn test_roundtrip_single_unicode_character() {
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, None);

    let test_string = "🙃";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert_eq!(encoded_ids, vec![8582, 247, 225]);
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);
}

#[test]
fn test_roundtrip_ascii_string() {
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, None);

    let test_string = "Hello, how are you?";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert_eq!(encoded_ids, vec![15496, 11, 703, 389, 345, 30]);
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);
}

#[test]
fn test_roundtrip_unicode_string() {
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, None);

    let test_string = "Héllò hôw are ü? 🙃";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert_eq!(
        encoded_ids,
        vec![
            39, 2634, 297, 127, 110, 289, 27083, 86, 389, 6184, 120, 30, 12520, 247, 225
        ]
    );
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);
}

#[test]
fn test_roundtrip_unicode_string_with_special_tokens() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, Some(&special_tokens));

    let test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert_eq!(
        encoded_ids,
        vec![
            39, 2634, 297, 127, 110, 289, 27083, 86, 220, 50256, 50256, 389, 6184, 120, 30, 12520,
            247, 225, 50256
        ]
    );
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);
}

#[test]
fn test_roundtrip_special_tokens() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, Some(&special_tokens));

    let test_string = "<|endoftext|>";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert_eq!(encoded_ids, [50256]);
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);
}

#[test]
// NOTE:it was designed to be `50257` instead of two `50256`, but this is a rare case, even the reference tokenizer doesn't support it
fn test_overlapping_special_tokens() {
    let special_tokens = vec![
        "<|endoftext|>".to_string(),
        "<|endoftext|><|endoftext|>".to_string(),
    ];
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, Some(&special_tokens));

    let test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert_eq!(
        encoded_ids,
        vec![15496, 11, 703, 220, 50256, 50256, 389, 345, 30, 50256]
    );
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);
}

#[test]
fn test_address_roundtrip() {
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, None);

    let content = fs::read_to_string("tests/fixtures/address.txt").expect("file should exist");
    let encoded_ids = t.encode(&content).expect("should be valid input");

    let ref_content = fs::read_to_string("tests/fixtures/address.ids").expect("file should exist");
    let ref_ids: Vec<_> = ref_content
        .trim()
        .split(",")
        .map(|s| s.trim().parse::<u16>().expect("valid token id"))
        .collect();

    assert_eq!(encoded_ids, ref_ids);

    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(content, decoded_string);
}

#[test]
fn test_german_roundtrip() {
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, None);

    let content = fs::read_to_string("tests/fixtures/german.txt").expect("file should exist");
    let encoded_ids = t.encode(&content).expect("should be valid input");

    let ref_content = fs::read_to_string("tests/fixtures/german.ids").expect("file should exist");
    let ref_ids: Vec<_> = ref_content
        .trim()
        .split(",")
        .map(|s| s.trim().parse::<u16>().expect("valid token id"))
        .collect();

    assert_eq!(encoded_ids, ref_ids);

    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(content, decoded_string);
}

#[test]
fn test_tinystories_sample_roundtrip() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, Some(&special_tokens));

    let content =
        fs::read_to_string("tests/fixtures/tinystories_sample.txt").expect("file should exist");
    let encoded_ids = t.encode(&content).expect("should be valid input");

    let ref_content =
        fs::read_to_string("tests/fixtures/tinystories_sample.ids").expect("file should exist");
    let ref_ids: Vec<_> = ref_content
        .trim()
        .split(",")
        .map(|s| s.trim().parse::<u16>().expect("valid token id"))
        .collect();

    assert_eq!(encoded_ids.len(), ref_ids.len());
    let mismatches: Vec<_> = encoded_ids
        .iter()
        .zip(ref_ids.iter())
        .enumerate()
        .filter(|(_, (actual, expected))| actual != expected)
        .collect();
    eprintln!("mismatched tokens: {}", mismatches.len());
    for (idx, (actual, expected)) in mismatches {
        eprintln!("mismatch at {idx}: actual={actual}, expected={expected}");
    }
    assert_eq!(encoded_ids, ref_ids);

    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(content, decoded_string);
}

#[test]
fn test_tinystories_sample_debug() {
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, None);

    let test_string = "”";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert_eq!(encoded_ids, vec![447, 251]);
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);

    let test_string = "?”";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert_eq!(encoded_ids, vec![30, 447, 251]);
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);

    let test_string = "!”";
    let encoded_ids = t.encode(test_string).expect("should be valid input");
    assert_eq!(encoded_ids, vec![0, 447, 251]);
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(test_string, decoded_string);
}

#[test]
fn test_encode_special_token_trailing_newlines() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, Some(&special_tokens));

    let content = fs::read_to_string("tests/fixtures/special_token_trailing_newlines.txt")
        .expect("file should exist");
    let encoded_ids = t.encode(&content).expect("should be valid input");
    assert_eq!(encoded_ids, vec![50256, 628]);
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(content, decoded_string);
}

#[test]
fn test_encode_special_token_double_newline_non_whitespace() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, Some(&special_tokens));

    let content =
        fs::read_to_string("tests/fixtures/special_token_double_newlines_non_whitespace.txt")
            .expect("file should exist");
    let encoded_ids = t.encode(&content).expect("should be valid input");
    assert_eq!(encoded_ids, vec![50256, 198, 198, 33407, 0]);
    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(content, decoded_string);
}

#[test]
fn test_tinystories_5m_roundtrip() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let t = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, Some(&special_tokens));

    let content =
        fs::read_to_string("tests/fixtures/tinystories_sample_5M.txt").expect("file should exist");
    let encoded_ids = t.encode(&content).expect("should be valid input");

    let ref_content =
        fs::read_to_string("tests/fixtures/tinystories_sample_5M.ids").expect("file should exist");
    let ref_ids: Vec<_> = ref_content
        .trim()
        .split(",")
        .map(|s| s.trim().parse::<u16>().expect("valid token id"))
        .collect();

    assert_eq!(encoded_ids.len(), ref_ids.len());
    let mismatches: Vec<_> = encoded_ids
        .iter()
        .zip(ref_ids.iter())
        .enumerate()
        .filter(|(_, (actual, expected))| actual != expected)
        .collect();
    eprintln!("mismatched tokens: {}", mismatches.len());
    for (idx, (actual, expected)) in mismatches {
        eprintln!("mismatch at {idx}: actual={actual}, expected={expected}");
    }
    assert_eq!(encoded_ids, ref_ids);

    let decoded_string = t.decode(&encoded_ids);
    assert_eq!(content, decoded_string);
}
