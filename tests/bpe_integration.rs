mod common;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
    process, thread,
    time::Instant,
};

use cs336_rs::utils::*;

use crate::common::{load_reference_merges, load_reference_vocab};

type BpeTrainingResultVocab = HashMap<usize, Vec<u8>>;
type BpeTrainingResultMerges = Vec<(Vec<u8>, Vec<u8>)>;
type PyBpeTrainingResult = (BpeTrainingResultVocab, BpeTrainingResultMerges);

fn bpe(input_path: PathBuf, vocab_size: u16, special_tokens: Vec<String>) -> PyBpeTrainingResult {
    let content = fs::read(&input_path).expect("file should be readable");
    let cpus = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);

    assert_eq!(special_tokens.len(), 1);

    let boundaries = find_chunk_boundaries(&content, cpus, &special_tokens);
    let mut spans =
        find_pretoken_spans(&content, &boundaries, &special_tokens).expect("should succeed");
    spans.sort();

    let gpt2_regex_str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

    let all_pieces: Vec<_> = spans.into_iter().flatten().collect();
    let freq_map =
        build_token_freq_map(&content, &all_pieces, cpus, &gpt2_regex_str).expect("should succeed");

    let freq_map = convert_freq_map_to_u16(freq_map);
    let result =
        train_bpe(freq_map, vocab_size as usize, &special_tokens, cpus).expect("should succeed");
    let mut vocab: BpeTrainingResultVocab = HashMap::new();
    vocab.insert(0, special_tokens[0].clone().into());
    for (id, bytes) in result.vocab.into_iter().enumerate() {
        vocab.insert(id + 1, bytes);
    }
    (vocab, result.merges)
}

#[test]
fn test_train_bpe_speed() {
    let input_path = "tests/fixtures/corpus.en";
    let input_path = fs::canonicalize(input_path).unwrap_or_else(|err| {
        eprintln!("invalid file path {input_path}: {err}",);
        process::exit(1);
    });

    let start_time = Instant::now();
    let (_, _) = bpe(input_path, 500, vec!["<|endoftext|>".to_string()]);
    let elapsed = start_time.elapsed();

    assert!(
        elapsed.as_secs_f64() < 1.5,
        "BPE training took {elapsed:?}, expected less than 1.5 seconds"
    );
}

#[test]
fn test_train_bpe() {
    let input_path = "tests/fixtures/corpus.en";
    let input_path = fs::canonicalize(input_path).unwrap_or_else(|err| {
        eprintln!("invalid file path {input_path}: {err}",);
        process::exit(1);
    });
    let (vocab, merges) = bpe(input_path, 500, vec!["<|endoftext|>".to_string()]);

    let reference_merges_path = "tests/fixtures/train-bpe-reference-merges.txt";
    let reference_vocab_path = "tests/fixtures/train-bpe-reference-vocab.json";

    let gpt2_byte_decoder: HashMap<char, u8> = common::gpt2_bytes_to_unicode()
        .into_iter()
        .map(|(k, v)| (v, k))
        .collect();

    let reference_merges = load_reference_merges(&reference_merges_path, &gpt2_byte_decoder)
        .expect("should read file");

    let reference_vocab = load_reference_vocab(&reference_vocab_path, &gpt2_byte_decoder)
        .expect("should be a valid json file");

    assert_eq!(merges, reference_merges);

    let vocab_keys: HashSet<_> = vocab.keys().copied().collect();
    let reference_vocab_keys: HashSet<_> = reference_vocab.keys().copied().collect();
    assert_eq!(vocab_keys, reference_vocab_keys);

    let vocab_values: HashSet<_> = vocab.values().cloned().collect();
    let reference_vocab_values: HashSet<_> = reference_vocab.values().cloned().collect();
    assert_eq!(vocab_values, reference_vocab_values);
}
