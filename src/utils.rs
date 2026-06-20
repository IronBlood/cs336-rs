use crate::error::CustomError;
use regex::Regex;
use std::{collections::HashSet, thread};

type Span = (usize, usize);

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
    special_tokens: &[&String],
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
        .map(|token| regex::escape(token))
        .collect::<Vec<_>>()
        .join("|");
    let re_split = Regex::new(&split_pattern).unwrap();

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
            let chunk_pieces = handle.join().unwrap()?;
            all_pieces.push(chunk_pieces);
        }

        Ok(all_pieces)
    })?;

    Ok(all_pieces)
}
