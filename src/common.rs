/// This is for GPT2-compatible byte conversion.
///
/// In the tables of `Basic Latin` (0x00-0x7F) and `Latin-1
/// Supplement` (0x80-0xFF) listed in the web page
/// https://www.ssec.wisc.edu/~tomw/java/unicode.html
/// printable characters are from this range:
/// - 0x21 (35) `!` to 0x7E (126) `~`
/// - 0xA1 (161) `¡` to 0xAC (172) `¬`
/// - 0xAE (174) `®` to 0xFF (255) `ÿ`
fn is_gpt2_visible_byte(b: u8) -> bool {
    (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b != 0xAD)
}

pub fn bytes_to_string(bytes: &[u8]) -> String {
    // at most double size
    let mut buf: Vec<u8> = Vec::with_capacity(bytes.len() * 2);

    for &b in bytes {
        if is_gpt2_visible_byte(b) {
            buf.push(b);
        } else {
            buf.push(0x01);
            buf.push(b);
        }
    }

    // at this stage the sequence should be a valid UTF-8
    String::from_utf8(buf).expect("valid UTF-8")
}

pub fn string_to_bytes(s: &str) -> Vec<u8> {
    let raw_bytes = s.as_bytes();
    let mut bytes: Vec<u8> = Vec::with_capacity(raw_bytes.len());
    let mut read = 0;
    let len = raw_bytes.len();

    while read < len {
        if raw_bytes[read] == 1 && read + 1 < len && !is_gpt2_visible_byte(raw_bytes[read + 1]) {
            bytes.push(raw_bytes[read + 1]);
            read += 2;
        } else {
            bytes.push(raw_bytes[read]);
            read += 1;
        }
    }

    // TODO: this block should be benchmarked
    if bytes.len() * 2 <= len {
        bytes.shrink_to_fit();
    }

    bytes
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bs() {
        let bytes = vec![1, b'a'];
        let s = bytes_to_string(&bytes);
        assert_eq!("ā".as_bytes(), vec![0xC4, 0x81]);
        assert_eq!(s, "āa");
    }
}
