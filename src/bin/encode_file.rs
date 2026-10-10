use std::{
    env,
    error::Error,
    fs,
    io::{self, BufWriter, Write},
    process, thread,
};

use cs336_rs::tokenizer::Tokenizer;

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let usage =
        "usage: encode_file <input> <output> --vocab <path> --merges <path> [--threads <count>]";
    if args.len() < 2 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, usage).into());
    }

    let mut vocab = None;
    let mut merges = None;
    let mut threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut options = args[2..].iter();
    while let Some(option) = options.next() {
        let value = options
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, usage))?;
        match option.as_str() {
            "--vocab" => vocab = Some(value.as_str()),
            "--merges" => merges = Some(value.as_str()),
            "--threads" => threads = value.parse()?,
            _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, usage).into()),
        }
    }
    if threads == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "threads must be greater than zero",
        )
        .into());
    }
    let vocab = vocab.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, usage))?;
    let merges = merges.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, usage))?;
    let pattern =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
    let tokenizer = Tokenizer::load(vocab, merges, &["<|endoftext|>".to_string()], pattern)?;
    let content = fs::read_to_string(&args[0])?;
    let ids = tokenizer.encode_mt(&content, threads)?;
    let mut output = BufWriter::new(fs::File::create(&args[1])?);
    for id in &ids {
        output.write_all(&id.to_le_bytes())?;
    }
    output.flush()?;
    println!("Wrote {} tokens to {}", ids.len(), args[1]);
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        process::exit(1);
    }
}
