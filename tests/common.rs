use std::collections::HashMap;

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
