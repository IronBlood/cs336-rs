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

fn convert_freq_map_to_u16(map: HashMap<Vec<u8>, usize>) -> HashMap<Vec<u16>, usize> {
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
) -> Result<HashMap<[u16; 2], usize>, CustomError> {
    let mut all_pairs: Vec<(&[u16], usize)> = Vec::new();
    for (k, v) in map {
        if k.len() >= 2 {
            all_pairs.push((k.as_slice(), *v));
        }
    }

    if all_pairs.is_empty() {
        // TODO: should stop counting
        return Ok(HashMap::new());
    }

    if threads == 1 {
        Ok(count_pairs_internal(&all_pairs))
    } else {
        let slices = split_indices(all_pairs.len(), threads);

        thread::scope(|scope| -> Result<HashMap<[u16; 2], usize>, CustomError> {
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

            Ok(total_count)
        })
    }
}

fn get_largest_pair(count_map: &HashMap<[u16; 2], usize>) -> Option<[u16; 2]> {
    count_map
        .iter()
        .max_by_key(|(pair, count)| (*count, *pair))
        .map(|(pair, _count)| *pair)
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

        let largest = get_largest_pair(&count_map);
        assert_eq!(largest, Some(['s' as u16, 't' as u16]));
    }
}
