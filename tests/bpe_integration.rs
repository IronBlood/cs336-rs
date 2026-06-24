mod common;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs, io,
    path::PathBuf,
    process, thread,
};

use cs336_rs::utils::*;

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

    let all_pieces: Vec<Span> = spans.into_iter().flatten().collect();
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

fn load_reference_merges(
    reference_merges_path: &str,
    gpt2_byte_decoder: &HashMap<char, u8>,
) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let contents = fs::read_to_string(reference_merges_path)?;

    let reference_merges = contents
        .lines()
        .map(|line| {
            let (merge_token_1, merge_token_2) = line
                .split_once(' ')
                .expect("each merge line should contain two tokens");

            let token_1 = merge_token_1
                .chars()
                .map(|ch| gpt2_byte_decoder[&ch])
                .collect::<Vec<u8>>();

            let token_2 = merge_token_2
                .chars()
                .map(|ch| gpt2_byte_decoder[&ch])
                .collect::<Vec<u8>>();

            (token_1, token_2)
        })
        .collect();

    Ok(reference_merges)
}

fn load_reference_vocab(
    reference_vocab_path: &str,
    gpt2_byte_decoder: &HashMap<char, u8>,
) -> Result<HashMap<usize, Vec<u8>>, Box<dyn Error>> {
    let contents = fs::read_to_string(reference_vocab_path)?;

    let gpt2_reference_vocab: HashMap<String, usize> = serde_json::from_str(&contents)?;

    let reference_vocab = gpt2_reference_vocab
        .into_iter()
        .map(|(gpt2_vocab_item, gpt2_vocab_index)| {
            let bytes = gpt2_vocab_item
                .chars()
                .map(|token| gpt2_byte_decoder[&token])
                .collect::<Vec<u8>>();
            (gpt2_vocab_index, bytes)
        })
        .collect();

    Ok(reference_vocab)
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
