use std::{env, error::Error, fs, io, process};

use cs336_rs::tokenizer::Tokenizer;

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let usage = "usage: decode_file <input> <output> --vocab <path> --merges <path>";
    if args.len() < 2 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, usage).into());
    }

    let mut vocab = None;
    let mut merges = None;
    let mut options = args[2..].iter();
    while let Some(option) = options.next() {
        let value = options
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, usage))?;
        match option.as_str() {
            "--vocab" => vocab = Some(value.as_str()),
            "--merges" => merges = Some(value.as_str()),
            _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, usage).into()),
        }
    }
    let vocab = vocab.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, usage))?;
    let merges = merges.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, usage))?;
    let pattern =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
    let tokenizer = Tokenizer::load(vocab, merges, &["<|endoftext|>".to_string()], pattern)?;
    let bytes = fs::read(&args[0])?;
    if bytes.len() % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input must contain complete uint16 tokens",
        )
        .into());
    }
    let ids: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    let content = tokenizer.decode(&ids);
    fs::write(&args[1], content.as_bytes())?;
    println!("Decoded {} tokens to {}", ids.len(), args[1]);
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        process::exit(1);
    }
}
