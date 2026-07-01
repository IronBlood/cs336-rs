use std::{collections::HashMap, error::Error, fs, io};

pub fn gpt2_bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();

    let mut cs: Vec<u32> = bs.iter().map(|&b| u32::from(b)).collect();

    let mut n = 0u32;

    for b in 0u8..=u8::MAX {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let characters = cs
        .into_iter()
        .map(|code_point| char::from_u32(code_point).expect("valid Unicode code point"));

    bs.into_iter().zip(characters).collect()
}

pub fn load_vocab(
    vocab_path: &str,
    gpt2_byte_decoder: &HashMap<char, u8>,
) -> Result<HashMap<u16, Vec<u8>>, Box<dyn Error>> {
    let contents = fs::read_to_string(vocab_path)?;

    let gpt2_reference_vocab: HashMap<String, u16> = serde_json::from_str(&contents)?;

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

pub fn load_merges(
    merges_path: &str,
    gpt2_byte_decoder: &HashMap<char, u8>,
) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let contents = fs::read_to_string(merges_path)?;

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
