use crate::error::CustomError;
use crate::regex::{Regex, escape_literal};
use std::{
    collections::{HashMap, HashSet},
    thread,
};

pub type Span = (usize, usize);
pub type WordFreqMap = HashMap<Vec<u8>, usize>;
type BorrowedWordFreqMap<'a> = HashMap<&'a [u8], usize>;

fn split_indices(size: usize, threads: usize) -> Vec<Span> {
    if size == 0 {
        return Vec::new();
    }

    let threads = threads.clamp(1, size);
    if threads == 1 {
        return vec![(0, size)];
    }

    let chunk_size = size.div_ceil(threads);
    (0..threads)
        .map(|idx| {
            let start = idx * chunk_size;
            let end = ((idx + 1) * chunk_size).min(size);
            (start, end)
        })
        .filter(|(start, end)| start < end)
        .collect()
}

fn find_special_tokens(chunk: &[u8], special_tokens_bytes: &[Vec<u8>]) -> Option<usize> {
    let first_offset: Option<usize> = special_tokens_bytes
        .iter()
        .filter(|needle| !needle.is_empty())
        .filter_map(|needle| {
            chunk
                .windows(needle.len())
                .position(|window| window == needle.as_slice())
        })
        .min();

    first_offset
}

pub fn find_chunk_boundaries(
    content: &[u8],
    desired_num_chunks: usize,
    special_tokens: &[String],
) -> Vec<usize> {
    let desired_num_chunks = if desired_num_chunks == 0 {
        1
    } else {
        desired_num_chunks
    };

    let file_size = content.len();
    let chunk_size = file_size / desired_num_chunks;

    if desired_num_chunks == 1 {
        return vec![0, file_size];
    }

    let mut chunk_boundaries: Vec<usize> =
        (0..=desired_num_chunks).map(|x| x * chunk_size).collect();
    chunk_boundaries[desired_num_chunks] = file_size;

    let special_tokens_bytes: Vec<Vec<u8>> = special_tokens
        .iter()
        .filter(|token| !token.is_empty())
        .map(|token| token.as_bytes().to_vec())
        .collect();

    let mini_chunk_size = 4096; // 4 KiB
    for bi in 1..desired_num_chunks {
        let prev_bi = bi - 1;
        if chunk_boundaries[bi] < chunk_boundaries[prev_bi] {
            chunk_boundaries[bi] = chunk_boundaries[prev_bi];
        }

        let mut initial_position = chunk_boundaries[bi];
        let mut found_boundary = false;
        while initial_position < file_size {
            let mini_chunk: &[u8] =
                &content[initial_position..file_size.min(initial_position + mini_chunk_size)];

            if let Some(found_at) = find_special_tokens(mini_chunk, &special_tokens_bytes) {
                chunk_boundaries[bi] = initial_position + found_at;
                found_boundary = true;
                break;
            }
            initial_position += mini_chunk_size;
        }

        if !found_boundary {
            chunk_boundaries[bi] = file_size;
        }
    }

    let mut chunk_boundaries: Vec<_> = chunk_boundaries
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    chunk_boundaries.sort();

    chunk_boundaries
}

pub fn find_pretoken_spans(
    content: &[u8],
    boundaries: &[usize],
    special_tokens: &[String],
) -> Result<Vec<Vec<Span>>, CustomError> {
    let special_tokens = special_tokens
        .iter()
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();

    if special_tokens.len() == 0 {
        return Ok(vec![vec![(0, content.len())]]);
    }

    let mut chunks: Vec<Span> = vec![];
    for i in 0..boundaries.len() - 1 {
        chunks.push((boundaries[i], boundaries[i + 1]));
    }

    let split_pattern = special_tokens
        .iter()
        .map(|token| escape_literal(token))
        .collect::<Vec<_>>()
        .join("|");
    let re_split = Regex::new(&split_pattern)?;

    let all_pieces = thread::scope(|scope| -> Result<Vec<Vec<Span>>, CustomError> {
        let mut handles = vec![];
        for (start, end) in chunks {
            let chunk: &[u8] = &content[start..end];
            let re = &re_split;

            handles.push(scope.spawn(move || -> Result<Vec<Span>, CustomError> {
                let text = str::from_utf8(chunk)?;
                let mut pieces = vec![];

                let mut last = 0;
                let text_offset = start;

                for mat in re.find_iter(text)? {
                    let mat = mat?;
                    let piece_start = text_offset + last;
                    let piece_end = text_offset + mat.start();

                    if piece_start < piece_end {
                        pieces.push((piece_start, piece_end));
                    }
                    last = mat.end();
                }

                let piece_start = text_offset + last;
                let piece_end = end;
                if piece_start < piece_end {
                    pieces.push((piece_start, piece_end));
                }

                Ok(pieces)
            }));
        }

        let mut all_pieces = vec![];

        for handle in handles {
            let chunk_pieces = handle.join().map_err(|_| CustomError::ThreadPanic)??;
            all_pieces.push(chunk_pieces);
        }

        Ok(all_pieces)
    })?;

    Ok(all_pieces)
}

pub fn build_token_freq_map(
    content: &[u8],
    all_pieces: &[Span],
    threads: usize,
    regex_str: &str,
) -> Result<HashMap<Vec<u8>, usize>, CustomError> {
    if all_pieces.is_empty() {
        return Ok(HashMap::new());
    }

    let re_match = Regex::new(regex_str)?;

    let chunks = split_indices(all_pieces.len(), threads);

    thread::scope(|scope| -> Result<WordFreqMap, CustomError> {
        let mut handles = Vec::new();
        let re = &re_match;

        for (span_start, span_end) in chunks {
            handles.push(
                scope.spawn(move || -> Result<BorrowedWordFreqMap<'_>, CustomError> {
                    let mut local_freq_map: BorrowedWordFreqMap<'_> = HashMap::new();
                    for span_idx in span_start..span_end {
                        let piece: Span = all_pieces[span_idx];
                        let (s, e) = piece;
                        let chunk = &content[s..e];
                        let text = str::from_utf8(chunk)?;

                        let text_offset = s;
                        for mat in re.find_iter(text)? {
                            let mat = mat?;
                            let matched_start = text_offset + mat.start();
                            let matched_end = text_offset + mat.end();
                            let matched_bytes = &content[matched_start..matched_end];
                            *local_freq_map.entry(matched_bytes).or_insert(0) += 1;
                        }
                    }

                    Ok(local_freq_map)
                }),
            );
        }

        let mut freq_map: WordFreqMap = HashMap::new();
        for handle in handles {
            let thread_freq_map = handle.join().map_err(|_| CustomError::ThreadPanic)??;
            for (token, count) in thread_freq_map {
                if let Some(total) = freq_map.get_mut(token) {
                    *total += count;
                } else {
                    freq_map.insert(token.to_vec(), count);
                }
            }
        }

        Ok(freq_map)
    })
}

pub fn convert_freq_map_to_u16(map: HashMap<Vec<u8>, usize>) -> HashMap<Vec<u16>, usize> {
    map.into_iter()
        .map(|(token, count)| (token.into_iter().map(|b| b as u16).collect(), count))
        .collect()
}

/**
 * NOTE: this function assumes length of all keys are >= 2
 */
fn count_pairs_internal(all_pairs: &[(&[u16], usize)]) -> HashMap<[u16; 2], usize> {
    let mut count_map = HashMap::<[u16; 2], usize>::new();

    for (key, count) in all_pairs {
        for pair in key.windows(2) {
            let buf = [pair[0], pair[1]];
            *count_map.entry(buf).or_insert(0) += *count;
        }
    }

    count_map
}

fn count_pairs(
    map: &HashMap<Vec<u16>, usize>,
    threads: usize,
) -> Result<Option<HashMap<[u16; 2], usize>>, CustomError> {
    let mut all_pairs: Vec<(&[u16], usize)> = Vec::new();
    for (k, v) in map {
        if k.len() >= 2 {
            all_pairs.push((k.as_slice(), *v));
        }
    }

    if all_pairs.is_empty() {
        return Ok(None);
    }

    if threads == 1 {
        Ok(Some(count_pairs_internal(&all_pairs)))
    } else {
        let slices = split_indices(all_pairs.len(), threads);

        thread::scope(
            |scope| -> Result<Option<HashMap<[u16; 2], usize>>, CustomError> {
                let mut handles = Vec::new();
                for (s, e) in slices {
                    let pair_slice = &all_pairs[s..e];
                    handles.push(scope.spawn(move || -> HashMap<[u16; 2], usize> {
                        count_pairs_internal(pair_slice)
                    }));
                }

                let mut total_count = HashMap::<[u16; 2], usize>::new();
                for handle in handles {
                    let thread_count = handle.join().map_err(|_| CustomError::ThreadPanic)?;
                    for (k, v) in thread_count {
                        *total_count.entry(k).or_insert(0) += v;
                    }
                }

                Ok(Some(total_count))
            },
        )
    }
}

type WordTuple<'a> = (&'a Vec<u8>, &'a Vec<u8>);
fn cmp_tuple<'a>(a: WordTuple<'a>, b: WordTuple<'a>) -> std::cmp::Ordering {
    a.cmp(&b)
}

fn get_largest_pair(count_map: &HashMap<[u16; 2], usize>, vocab: &[Vec<u8>]) -> Option<[u16; 2]> {
    count_map
        .iter()
        .max_by(|(pair_a, count_a), (pair_b, count_b)| {
            count_a.cmp(count_b).then_with(|| {
                let a0 = &vocab[pair_a[0] as usize];
                let a1 = &vocab[pair_a[1] as usize];
                let b0 = &vocab[pair_b[0] as usize];
                let b1 = &vocab[pair_b[1] as usize];

                cmp_tuple((a0, a1), (b0, b1))
            })
        })
        .map(|(pair, _count)| *pair)
}

fn replace_pair_in_token(token: &mut Vec<u16>, pair: &[u16; 2], new_id: u16) {
    let mut read = 0;
    let mut write = 0;

    while read < token.len() {
        if read + 1 < token.len() && token[read] == pair[0] && token[read + 1] == pair[1] {
            token[write] = new_id;
            read += 2;
        } else {
            token[write] = token[read];
            read += 1;
        }
        write += 1;
    }

    token.truncate(write);
}

fn replace_pair_in_freq_map(
    freq_map: HashMap<Vec<u16>, usize>,
    pair: &[u16; 2],
    new_id: u16,
    threads: usize,
) -> HashMap<Vec<u16>, usize> {
    let mut entries: Vec<(Vec<u16>, usize)> = freq_map.into_iter().collect();
    if entries.is_empty() {
        return HashMap::new();
    }

    let threads = threads.clamp(1, entries.len());
    let chunk_size = entries.len().div_ceil(threads);

    thread::scope(|scope| {
        for entries in entries.chunks_mut(chunk_size) {
            scope.spawn(move || {
                for (token, _count) in entries {
                    replace_pair_in_token(token, pair, new_id);
                }
            });
        }
    });

    entries.into_iter().collect()
}

fn init_vocab() -> Vec<Vec<u8>> {
    (0..=255).map(|i| vec![i]).collect()
}

pub struct BpeTrainingResult {
    pub vocab: Vec<Vec<u8>>,
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
}

pub fn train_bpe(
    mut freq_map: HashMap<Vec<u16>, usize>,
    max_vocab_size: usize,
    threads: usize,
) -> Result<BpeTrainingResult, CustomError> {
    let mut vocab = init_vocab();
    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

    let largest_idx = (max_vocab_size.min(0x10000) - 1) as u16;

    for idx in 256..=largest_idx {
        let all_pairs = count_pairs(&freq_map, threads)?;
        if all_pairs.is_none() {
            // nothing to be merged
            break;
        }

        let pair = get_largest_pair(&all_pairs.unwrap(), &vocab);
        if pair.is_none() {
            // nothing found, shouldn't be here
            break;
        }

        let pair = pair.unwrap();
        freq_map = replace_pair_in_freq_map(freq_map, &pair, idx, threads);
        let a = vocab[pair[0] as usize].clone();
        let b = vocab[pair[1] as usize].clone();
        let c: Vec<u8> = a.iter().chain(b.iter()).copied().collect();
        vocab.push(c);
        merges.push((a, b));
    }

    Ok(BpeTrainingResult { vocab, merges })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_special_tokens_should_return_the_smallest_occurance() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5];
        let special_tokens_bytes: Vec<Vec<u8>> = vec![vec![8], vec![5], vec![3]];

        let offset = find_special_tokens(&data, &special_tokens_bytes);
        assert_eq!(offset, Some(2));
    }

    #[test]
    fn test_find_special_tokens_should_return_none() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5];
        let special_tokens_bytes: Vec<Vec<u8>> = vec![vec![8]];
        let offset = find_special_tokens(&data, &special_tokens_bytes);
        assert!(offset.is_none());
    }

    #[test]
    fn test_find_special_tokens_with_empty_special_tokens() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5];
        let special_tokens_bytes: Vec<Vec<u8>> = vec![];
        let offset = find_special_tokens(&data, &special_tokens_bytes);
        assert!(offset.is_none());
    }

    #[test]
    fn test_find_chunk_boundaries_1_chunk_should_return_start_and_end() {
        let content = vec![1, 2, 3];
        let special_tokens = vec!["a".to_string()];
        let boundaries = find_chunk_boundaries(&content, 1, &special_tokens);
        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0], 0);
        assert_eq!(boundaries[1], content.len());
    }

    #[test]
    fn test_find_chunk_boundaries_0_chunk_should_fall_to_1() {
        let content = vec![1, 2, 3];
        let special_tokens = vec!["a".to_string()];
        let boundaries = find_chunk_boundaries(&content, 0, &special_tokens);
        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0], 0);
        assert_eq!(boundaries[1], content.len());
    }

    #[test]
    fn test_find_chunk_boundaries_should_return_the_correct_location() {
        let text = "abc<|>abc<|>abc";
        let content = text.as_bytes();
        let special_token = vec!["<|>".to_string()];
        let boundaries = find_chunk_boundaries(&content, 2, &special_token);
        assert_eq!(boundaries.len(), 3);
        assert_eq!(boundaries[0], 0);
        assert_eq!(boundaries[1], 9);
        assert_eq!(boundaries[2], content.len());
    }

    #[test]
    fn test_find_pretoken_spans() {
        let text = "abc<|>abc<|>abc";
        let content = text.as_bytes();
        let special_token = vec!["<|>".to_string()];
        let boundaries = find_chunk_boundaries(&content, 2, &special_token);
        let spans = find_pretoken_spans(&content, &boundaries, &special_token);
        let mut spans = spans.expect("operation should succeed");
        spans.sort();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], vec![(0, 3), (6, 9)]);
        assert_eq!(spans[1], vec![(12, 15)]);
    }

    #[test]
    fn test_find_token_freq_map() {
        let text = "abc<|>abc<|>abc";
        let content = text.as_bytes();
        let special_token = vec!["<|>".to_string()];
        let boundaries = find_chunk_boundaries(&content, 2, &special_token);
        let spans = find_pretoken_spans(&content, &boundaries, &special_token);
        let mut spans = spans.expect("operation should succeed");
        spans.sort();

        // continue from `test_find_pretoken_spans`
        let all_pieces: Vec<Span> = spans.into_iter().flatten().collect();
        let freq_map = build_token_freq_map(&content, &all_pieces, 2, "\\w+");
        let freq_map = freq_map.expect("operation should succeed");
        assert_eq!(freq_map.len(), 1);
        let count = freq_map.get("abc".as_bytes());
        assert_eq!(count, Some(&3));
    }

    #[test]
    fn test_largest_pair() {
        let vocab = init_vocab();
        let mut count_map: HashMap<[u16; 2], usize> = HashMap::new();

        count_map.insert(['l' as u16, 'o' as u16], 7);
        count_map.insert(['o' as u16, 'w' as u16], 7);
        count_map.insert(['w' as u16, 'e' as u16], 8);
        count_map.insert(['e' as u16, 'r' as u16], 2);
        count_map.insert(['w' as u16, 'i' as u16], 3);
        count_map.insert(['i' as u16, 'd' as u16], 3);
        count_map.insert(['d' as u16, 'e' as u16], 3);
        count_map.insert(['e' as u16, 's' as u16], 9);
        count_map.insert(['s' as u16, 't' as u16], 9);
        count_map.insert(['n' as u16, 'e' as u16], 6);
        count_map.insert(['e' as u16, 'w' as u16], 6);

        let largest = get_largest_pair(&count_map, &vocab);
        assert_eq!(largest, Some(['s' as u16, 't' as u16]));
    }

    #[test]
    fn test_train_bpe() {
        let text = b"aaabdaaabac";
        let text_vec: Vec<u16> = text.iter().map(|&b| b as u16).collect();
        let mut freq_map: HashMap<Vec<u16>, usize> = HashMap::new();
        freq_map.insert(text_vec, 1);

        // 1
        let x = freq_map.clone();
        let result = train_bpe(x, 256 + 1, 1).expect("shouln't throw error");
        assert_eq!(result.vocab.len(), 256 + 1);
        assert_eq!(result.vocab[256 + 0], b"aa".to_vec());
        assert_eq!(result.merges.len(), 1);
        assert_eq!(result.merges[0], (b"a".to_vec(), b"a".to_vec()));

        // 2
        let x = freq_map.clone();
        let result = train_bpe(x, 256 + 2, 1).expect("shouln't throw error");
        assert_eq!(result.vocab.len(), 256 + 2);
        assert_eq!(result.vocab[256 + 0], b"aa".to_vec());
        assert_eq!(result.vocab[256 + 1], b"aaa".to_vec());
        assert_eq!(result.merges.len(), 2);
        assert_eq!(result.merges[0], (b"a".to_vec(), b"a".to_vec()));
        assert_eq!(result.merges[1], (b"aa".to_vec(), b"a".to_vec()));

        // 3
        let x = freq_map.clone();
        let result = train_bpe(x, 256 + 3, 1).expect("shouln't throw error");
        assert_eq!(result.vocab.len(), 256 + 3);
        assert_eq!(result.vocab[256 + 0], b"aa".to_vec());
        assert_eq!(result.vocab[256 + 1], b"aaa".to_vec());
        assert_eq!(result.vocab[256 + 2], b"aaab".to_vec());
        assert_eq!(result.merges.len(), 3);
        assert_eq!(result.merges[0], (b"a".to_vec(), b"a".to_vec()));
        assert_eq!(result.merges[1], (b"aa".to_vec(), b"a".to_vec()));
        assert_eq!(result.merges[2], (b"aaa".to_vec(), b"b".to_vec()));
    }

    #[test]
    fn test_cmp_tuple() {
        let data: Vec<(Vec<u8>, Vec<u8>)> = vec![
            (vec![65], vec![66]),       // A B
            (vec![65], vec![67]),       // A C
            (vec![66], b"ZZ".to_vec()), // B ZZ
            (vec![66, 65], vec![65]),   // BA A
        ];

        let max = data
            .iter()
            .max_by(|a, b| cmp_tuple((&a.0, &a.1), (&b.0, &b.1)))
            .cloned();
        assert_eq!(max, Some((vec![66, 65], vec![65])));
    }
}
