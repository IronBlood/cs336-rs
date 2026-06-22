use crate::error::CustomError;
use crate::regex::{Regex, escape_literal};
use std::{
    collections::{HashMap, HashSet},
    thread,
};

pub type Span = (usize, usize);
pub type WordFreqMap = HashMap<Vec<u8>, usize>;

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

                for mat in re.find_iter(text) {
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

    let threads = threads.clamp(1, all_pieces.len().max(1));
    let pieces_per_thread = 1usize.max(all_pieces.len().div_ceil(threads));
    let mut chunks: Vec<Span> = (0..threads)
        .map(|idx| (idx * pieces_per_thread, (idx + 1) * pieces_per_thread))
        .collect();
    chunks[threads - 1].1 = all_pieces.len();

    thread::scope(|scope| -> Result<WordFreqMap, CustomError> {
        let mut handles = Vec::new();
        let re = &re_match;

        for span in chunks {
            let span_start = span.0;
            let span_end = span.1;

            handles.push(scope.spawn(move || -> Result<WordFreqMap, CustomError> {
                let mut local_freq_map: WordFreqMap = HashMap::new();
                for span_idx in span_start..span_end {
                    let piece: Span = all_pieces[span_idx];
                    let (s, e) = piece;
                    let chunk = &content[s..e];
                    let text = str::from_utf8(chunk)?;

                    let text_offset = s;
                    for mat in re.find_iter(text) {
                        let mat = mat?;
                        let matched_start = text_offset + mat.start();
                        let matched_end = text_offset + mat.end();
                        let matched_bytes = &content[matched_start..matched_end];
                        *local_freq_map.entry(matched_bytes.into()).or_insert(0) += 1;
                    }
                }

                Ok(local_freq_map)
            }));
        }

        let mut freq_map: WordFreqMap = HashMap::new();
        for handle in handles {
            let thread_freq_map = handle.join().map_err(|_| CustomError::ThreadPanic)??;
            for (token, count) in thread_freq_map {
                *freq_map.entry(token).or_insert(0) += count;
            }
        }

        Ok(freq_map)
    })
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
}
