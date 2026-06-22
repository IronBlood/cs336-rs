use std::{env, fs, path::PathBuf, process, thread};

use cs336_rs::utils::{Span, build_token_freq_map, find_chunk_boundaries, find_pretoken_spans};

struct CliArgs {
    file_path: PathBuf,
    threads: usize,
}

fn parse_args() -> Result<CliArgs, String> {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "cs336_rs".to_string());

    let file_path = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("usage: {program} <file_path> <threads>"))?;

    let threads = args
        .next()
        .ok_or_else(|| format!("usage: {program} <file_path> <threads>"))?
        .parse::<usize>()
        .map_err(|err| format!("invalid thread count: {err}"))?;

    if args.next().is_some() {
        return Err(format!("usage: {program} <file_path> <threads>"));
    }

    Ok(CliArgs { file_path, threads })
}

fn main() {
    let mut args = parse_args().unwrap_or_else(|err| {
        eprintln!("{err}");
        process::exit(1);
    });

    let cpus = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    args.threads = args.threads.min(cpus);
    args.file_path = fs::canonicalize(&args.file_path).unwrap_or_else(|err| {
        eprintln!("invalid file path {}: {err}", args.file_path.display());
        process::exit(1);
    });

    let special_tokens = vec!["<|endoftext|>".to_string()];
    let content = fs::read(&args.file_path).expect("file should be readable");

    let boundaries = find_chunk_boundaries(&content, args.threads, &special_tokens);
    let mut spans =
        find_pretoken_spans(&content, &boundaries, &special_tokens).expect("should succeed");
    spans.sort();

    let gpt2_regex_str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

    let all_pieces: Vec<Span> = spans.into_iter().flatten().collect();
    let freq_map = build_token_freq_map(&content, &all_pieces, args.threads, &gpt2_regex_str)
        .expect("should succeed");
    println!("result: {} entries", freq_map.len());
    let total_count: usize = freq_map.values().sum();
    println!("result: {} total tokens", total_count);
}
