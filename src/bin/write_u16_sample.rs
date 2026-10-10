use std::{
    env,
    error::Error,
    fs::File,
    io::{self, BufWriter, Write},
    process,
};

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: write_u16_sample <output_path>",
        )
        .into());
    }

    let mut output = BufWriter::new(File::create(&args[0])?);
    for value in [0_u16, 1, 2, 3, 65535] {
        output.write_all(&value.to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        process::exit(1);
    }
}
