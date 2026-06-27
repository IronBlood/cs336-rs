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

fn byte_to_utf8(bytes: &mut Vec<u8>, x: u16) {
    debug_assert!(x < 0x0200, "invalid range");
    if x < 0x80 {
        bytes.push(x as u8);
    } else {
        bytes.push(0xC0 | ((x >> 6) as u8));
        bytes.push(0x80 | ((x & 0x3F) as u8));
    }
}

pub fn bytes_to_string(bytes: &[u8]) -> String {
    // at most double size
    let mut buf: Vec<u8> = Vec::with_capacity(bytes.len() * 2);

    for &b in bytes {
        if is_gpt2_visible_byte(b) {
            byte_to_utf8(&mut buf, b as u16);
        } else {
            byte_to_utf8(&mut buf, 0x100 | b as u16);
        }
    }

    // at this stage the sequence should be a valid UTF-8
    String::from_utf8(buf).expect("valid UTF-8")
}

fn is_utf8_byte_3(x: u8) -> bool {
    ((x >> 4) & 0xf != 0) && ((x >> 3) & 0x1 == 0)
}

fn is_utf8_byte_2(x: u8) -> bool {
    ((x >> 5) & 0x7 != 0) && ((x >> 4) & 0x1 == 0)
}

fn is_utf8_byte_1(x: u8) -> bool {
    ((x >> 6) & 0x3 != 0) && ((x >> 5) & 0x1 == 0)
}

fn is_utf8_byte_0(x: u8) -> bool {
    ((x >> 7) & 0x1 != 0) && ((x >> 6) & 0x1 == 0)
}

pub fn string_to_bytes(s: &str) -> Vec<u8> {
    let raw_bytes = s.as_bytes();
    let mut bytes: Vec<u8> = Vec::with_capacity(raw_bytes.len());
    let mut read = 0;
    let len = raw_bytes.len();

    while read < len {
        let curr = raw_bytes[read];

        // TODO: validate the bytes
        if is_utf8_byte_1(curr) {
            if read + 1 < len && is_utf8_byte_0(raw_bytes[read + 1]) {
                let next = raw_bytes[read + 1];
                let byte: u8 = ((curr & 0x1F) << 6) | (next & 0x3F);
                bytes.push(byte);
                read += 2;
            } else {
                panic!("not a valid sequence");
            }
        } else {
            bytes.push(curr);
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
        assert_eq!(s, "āa");
    }

    #[test]
    fn test_cs() {
        let s = "āa";
        let bytes = string_to_bytes(&s);
        assert_eq!(bytes, vec![1, b'a']);
    }
}
