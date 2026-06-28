use std::{
    collections::HashMap,
    env,
    fs::{self, read},
    path::PathBuf,
    process, thread,
    time::Instant,
};

use cs336_rs::utils::{
    Span, build_token_freq_map, convert_freq_map_to_u16, find_chunk_boundaries,
    find_pretoken_spans, train_bpe_profile,
};

struct StepTimer {
    last: Instant,
}

impl StepTimer {
    fn new() -> Self {
        Self {
            last: Instant::now(),
        }
    }

    fn mark(&mut self, label: &str) {
        let elapsed = self.last.elapsed();
        println!("{label}: {:.3}s", elapsed.as_secs_f64());
        self.last = Instant::now();
    }
}

type BpeTrainingResultVocab = HashMap<usize, Vec<u8>>;
type BpeTrainingResultMerges = Vec<(Vec<u8>, Vec<u8>)>;
type PyBpeTrainingResult = (BpeTrainingResultVocab, BpeTrainingResultMerges);

fn bpe(input_path: PathBuf, vocab_size: u16, special_tokens: Vec<String>) -> PyBpeTrainingResult {
    let cpus = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);

    assert_eq!(special_tokens.len(), 1);

    let gpt2_regex_str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

    let mut timer = StepTimer::new();

    let content = read(&input_path).expect("file should be readable");
    timer.mark("read file");

    let boundaries = find_chunk_boundaries(&content, cpus, &special_tokens);
    timer.mark("find boundaries");

    let mut spans =
        find_pretoken_spans(&content, &boundaries, &special_tokens).expect("should succeed");
    timer.mark("find spans");

    spans.sort();
    timer.mark("span sort");

    let all_pieces: Vec<Span> = spans.into_iter().flatten().collect();
    timer.mark("flatten spans");

    let freq_map =
        build_token_freq_map(&content, &all_pieces, cpus, &gpt2_regex_str).expect("should succeed");
    timer.mark("build freq map");

    let freq_map = convert_freq_map_to_u16(freq_map);
    timer.mark("convert freq map");

    let result = train_bpe_profile(freq_map, vocab_size as usize, &special_tokens, cpus)
        .expect("should succeed");
    timer.mark("train bpe");

    let mut vocab: BpeTrainingResultVocab = HashMap::new();
    vocab.insert(0, special_tokens[0].clone().into());
    for (id, bytes) in result.vocab.into_iter().enumerate() {
        vocab.insert(id + 1, bytes);
    }
    timer.mark("build vocab");
    (vocab, result.merges)
}

fn main() {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "profile_bpe".to_string());
    let Some(input_path) = args.next() else {
        eprintln!("usage: {program} <input_path> <vocab_size>");
        process::exit(1);
    };
    let Some(vocab_size) = args.next() else {
        eprintln!("usage: {program} <input_path> <vocab_size>");
        process::exit(1);
    };

    let vocab_size = vocab_size.parse::<u16>().unwrap_or_else(|err| {
        eprintln!("invalid vocab size {vocab_size}: {err}");
        process::exit(1);
    });

    let input_path = fs::canonicalize(&input_path).unwrap_or_else(|err| {
        eprintln!("invalid file path {input_path}: {err}",);
        process::exit(1);
    });

    bpe(input_path, vocab_size, vec!["<|endoftext|>".to_string()]);
}
