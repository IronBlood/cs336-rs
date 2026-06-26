use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process, thread,
};

use cs336_rs::utils::{Span, build_token_freq_map, find_chunk_boundaries, find_pretoken_spans};

struct CliArgs {
    file_path: PathBuf,
    threads: usize,
    output_path: Option<PathBuf>,
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

    let output_path = match args.next() {
        Some(flag) if flag == "-o" => Some(
            args.next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("usage: {program} <file_path> <threads> [-o output.tsv]"))?,
        ),
        Some(_) => {
            return Err(format!(
                "usage: {program} <file_path> <threads> [-o output.tsv]"
            ));
        }
        None => None,
    };

    if args.next().is_some() {
        return Err(format!(
            "usage: {program} <file_path> <threads> [-o output.tsv]"
        ));
    }

    Ok(CliArgs {
        file_path,
        threads,
        output_path,
    })
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn write_freq_map<'a, I>(path: &PathBuf, entries: I) -> io::Result<()>
where
    I: IntoIterator<Item = (&'a [u8], usize)>,
{
    let mut entries = entries.into_iter().collect::<Vec<_>>();
    entries.sort_by(|(left, _), (right, _)| left.cmp(right));

    let mut file = fs::File::create(path)?;
    for (token, count) in entries {
        writeln!(file, "{}\t{}", hex_encode(token), count)?;
    }

    Ok(())
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

    if let Some(output_path) = args.output_path {
        write_freq_map(
            &output_path,
            freq_map.iter().map(|(token, count)| (*token, *count)),
        )
        .unwrap_or_else(|err| {
            eprintln!("failed to write {}: {err}", output_path.display());
            process::exit(1);
        });
    }
}
