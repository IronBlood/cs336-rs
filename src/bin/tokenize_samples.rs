use std::{
    env, fs, process, thread,
    time::{Duration, Instant},
};

use cs336_rs::{
    tokenizer::Tokenizer,
    utils::{find_chunk_boundaries, find_pretoken_spans},
};

struct TokenizerCfg {
    vocab_path: String,
    merges_path: String,
}

struct CliArgs {
    file_path: String,
    tokenizer_cfg: Option<TokenizerCfg>,
}

fn parse_args() -> Result<CliArgs, String> {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "tokenize_samples".to_string());

    let err_msg = format!(
        "usage: \"{program} <file_path>\" or \"{program}\" --vocab <vocab_path> --merges <merges_path>"
    );

    let file_path = args.next().ok_or_else(|| err_msg.clone())?;

    let tokenizer_cfg = match args.next() {
        Some(flag) if flag == "--vocab" => {
            let vocab_path = args.next().ok_or_else(|| err_msg.clone())?;
            let merges_path = match args.next() {
                Some(flag) if flag == "--merges" => args.next().ok_or_else(|| err_msg.clone())?,
                _ => {
                    return Err(err_msg);
                }
            };
            Some(TokenizerCfg {
                vocab_path,
                merges_path,
            })
        }
        Some(_) => {
            return Err(err_msg);
        }
        None => None,
    };

    Ok(CliArgs {
        file_path,
        tokenizer_cfg,
    })
}

struct State {
    bytes: usize,
    tokens: usize,
    duration: Duration,
}

fn main() {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("{err}");
        process::exit(1);
    });

    let cpus = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);

    let special_tokens = vec!["<|endoftext|>".to_string()];
    let content = fs::read(&args.file_path).expect("file should be readable");

    let boundaries = find_chunk_boundaries(&content, cpus, &special_tokens);
    let mut spans =
        find_pretoken_spans(&content, &boundaries, &special_tokens).expect("should succeed");
    spans.sort();

    let all_pieces: Vec<_> = spans.into_iter().flatten().collect();
    println!(
        "There are {} documents in {}",
        all_pieces.len(),
        args.file_path
    );

    let mt_mode = false;
    if let Some(cfg) = args.tokenizer_cfg {
        let gpt2_regex_str =
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
        let tokenizer = Tokenizer::load(
            &cfg.vocab_path,
            &cfg.merges_path,
            &special_tokens,
            &gpt2_regex_str,
        )
        .unwrap_or_else(|err| {
            eprintln!("{}", err);
            process::exit(1);
        });

        let mut state = State {
            bytes: 0,
            tokens: 0,
            duration: Duration::ZERO,
        };

        for (span_start, span_end) in all_pieces.iter().take(10000) {
            let piece = &content[*span_start..*span_end];
            let piece = str::from_utf8(piece).expect("should be valid utf-8 content");
            let t = Instant::now();

            let ids = if mt_mode {
                tokenizer.encode_mt(piece, cpus).expect("should be encoded")
            } else {
                tokenizer.encode(piece).expect("should be encoded")
            };

            state.duration += t.elapsed();
            state.bytes += piece.as_bytes().len();
            state.tokens += ids.len();
        }

        let len = state.bytes;
        let ratio = len as f64 / state.tokens as f64;
        let throughput = len as f64 / state.duration.as_secs_f64();

        println!(
            "Encoded {} bytes in {:.3}s, ratio: {:.3}, throughput: {:.3} bytes/s",
            len,
            state.duration.as_secs_f64(),
            ratio,
            throughput
        );
    }
}
