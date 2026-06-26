use crate::error::CustomError;
use crate::regex::{Regex, escape_literal};
use std::{
    collections::{HashMap, HashSet},
    thread,
};

pub type TokenBytes = Vec<u8>;
pub type TokenIds = Vec<u16>;
pub type Span = (usize, usize);
type TokenId = u16;
type PackedPair = u32;
type BorrowedWordFreqMap<'a> = HashMap<&'a [u8], usize>;

fn find_special_tokens(chunk: &[u8], special_tokens_bytes: &[TokenBytes]) -> Option<usize> {
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
    let file_size = content.len();

    let desired_num_chunks = if desired_num_chunks == 0 {
        1
    } else {
        desired_num_chunks
    };

    let chunk_size = file_size / desired_num_chunks;

    if desired_num_chunks == 1 {
        return vec![0, file_size];
    }

    let mut chunk_boundaries: Vec<usize> =
        (0..=desired_num_chunks).map(|x| x * chunk_size).collect();
    chunk_boundaries[desired_num_chunks] = file_size;

    let special_tokens_bytes: Vec<TokenBytes> = special_tokens
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

/// This function builds a hashmap of word-count to reduce the time for BPE training.
///
/// There are a lot of repeating words in the original content. When doing BPE, we need
/// to count every byte pairs. There are lots of repeating words, so counting once then
/// multiplied by how many time this word appears, would save a lot of time. Even during
/// the process of merging, for example in the word `ABCDE` when `BC` is replaced by `X`
/// and making a new word `AXDE`, it is still unique, won't be turned to another existing
/// key, thus the pair counting will still be correct.
pub fn build_token_freq_map<'content>(
    content: &'content [u8],
    all_pieces: &[Span],
    threads: usize,
    regex_str: &str,
) -> Result<BorrowedWordFreqMap<'content>, CustomError> {
    if all_pieces.is_empty() {
        return Ok(HashMap::new());
    }

    let re_match = Regex::new(regex_str)?;

    thread::scope(|scope| -> Result<_, CustomError> {
        let mut handles = Vec::new();
        let re = &re_match;
        let threads = threads.clamp(1, all_pieces.len());
        let chunk_size = all_pieces.len().div_ceil(threads);

        for piece_chunk in all_pieces.chunks(chunk_size) {
            handles.push(scope.spawn(move || -> Result<_, CustomError> {
                let mut local_freq_map: BorrowedWordFreqMap<'_> = HashMap::new();
                for &(s, e) in piece_chunk {
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
            }));
        }

        let mut freq_map = HashMap::new();
        for handle in handles {
            let thread_freq_map = handle.join().map_err(|_| CustomError::ThreadPanic)??;
            merge_count_map(&mut freq_map, thread_freq_map);
        }

        Ok(freq_map)
    })
}

/// This function converts the raw borrowed freq_map to the format (owned u16) used in BPE training.
///
/// The raw freq_map might be useful for debugging.
pub fn convert_freq_map_to_u16(map: BorrowedWordFreqMap<'_>) -> HashMap<TokenIds, usize> {
    map.into_iter()
        .map(|(token, count)| (token.into_iter().map(|&b| b as TokenId).collect(), count))
        .collect()
}

/// Turning a [u16; 2] to a u32 to be used as the hash key to save time
fn pack_pair(hi: TokenId, lo: TokenId) -> PackedPair {
    (hi as u32) << 16 | lo as u32
}

/// extract the internal used u32 form to the original byte pair
fn unpack_pair(x: PackedPair) -> [TokenId; 2] {
    [(x >> 16) as u16, x as u16]
}

/**
 * NOTE: this function assumes length of all keys are >= 2
 */
fn count_pairs_internal(all_pairs: &[(&[TokenId], usize)]) -> HashMap<PackedPair, usize> {
    let mut count_map = HashMap::new();

    for (key, count) in all_pairs {
        for pair in key.windows(2) {
            let x = pack_pair(pair[0], pair[1]);
            *count_map.entry(x).or_insert(0) += *count;
        }
    }

    count_map
}

fn count_pairs(
    entries: &[(TokenIds, usize)],
    threads: usize,
) -> Result<Option<HashMap<PackedPair, usize>>, CustomError> {
    let mut all_pairs: Vec<(&[TokenId], usize)> = Vec::new();
    for (k, v) in entries {
        if k.len() >= 2 {
            all_pairs.push((k.as_slice(), *v));
        }
    }

    if all_pairs.is_empty() {
        return Ok(None);
    }

    let threads = threads.clamp(1, all_pairs.len());
    let chunk_size = all_pairs.len().div_ceil(threads);

    if threads == 1 {
        Ok(Some(count_pairs_internal(&all_pairs)))
    } else {
        thread::scope(
            |scope| -> Result<Option<HashMap<PackedPair, usize>>, CustomError> {
                let mut handles = Vec::new();
                for pair_slice in all_pairs.chunks(chunk_size) {
                    handles.push(scope.spawn(move || -> HashMap<PackedPair, usize> {
                        count_pairs_internal(pair_slice)
                    }));
                }

                let mut total_count = HashMap::new();
                for handle in handles {
                    let thread_count = handle.join().map_err(|_| CustomError::ThreadPanic)?;
                    merge_count_map(&mut total_count, thread_count);
                }

                Ok(Some(total_count))
            },
        )
    }
}

type WordTuple<'a> = (&'a TokenBytes, &'a TokenBytes);
fn cmp_tuple<'a>(a: WordTuple<'a>, b: WordTuple<'a>) -> std::cmp::Ordering {
    a.cmp(&b)
}

fn get_largest_pair(
    count_map: &HashMap<PackedPair, usize>,
    vocab: &[TokenBytes],
) -> Option<[TokenId; 2]> {
    count_map
        .iter()
        .max_by(|(pair_a, count_a), (pair_b, count_b)| {
            count_a.cmp(count_b).then_with(|| {
                let a0 = &vocab[(*pair_a >> 16) as usize];
                let a1 = &vocab[(*pair_a & 0xffff) as usize];
                let b0 = &vocab[(*pair_b >> 16) as usize];
                let b1 = &vocab[(*pair_b & 0xffff) as usize];

                cmp_tuple((a0, a1), (b0, b1))
            })
        })
        .map(|(pair, _count)| unpack_pair(*pair))
}

/// Pair-count changes produced by applying one BPE merge.
///
/// Whne a pair `(a, b)` is replaced by a new token `x`, only neighboring pairs
/// around each replacement site change. `removed` records pair counts that
/// should be subtracted from the global pair-count map, and `added` records
/// pair counts that should be added.
struct PairCountDelta {
    removed: HashMap<PackedPair, usize>,
    added: HashMap<PackedPair, usize>,
}

fn replace_pair_in_token(
    token: &mut TokenIds,
    pair: &[TokenId; 2],
    new_id: TokenId,
    multiplier: usize,
    count_delta: &mut PairCountDelta,
) {
    // a two-pointer in-place approach
    let mut read = 0;
    let mut write = 0;

    while read < token.len() {
        if read + 1 < token.len() && token[read] == pair[0] && token[read + 1] == pair[1] {
            // Imagine we are going to replace `A B` from `[L] A B [R}` with `X`, where `L` and `R` are optional.
            // This block means `L` exists, so we need to remove `LA`, then add `LX`
            if write > 0 {
                let prev = pack_pair(token[write - 1], pair[0]);
                *count_delta.removed.entry(prev).or_insert(0) += multiplier;
                let prev = pack_pair(token[write - 1], new_id);
                *count_delta.added.entry(prev).or_insert(0) += multiplier;
            }
            // This block means `R` exists, so we need to remove `BR`, then add `XR`
            if read + 2 < token.len() {
                let next = pack_pair(pair[1], token[read + 2]);
                *count_delta.removed.entry(next).or_insert(0) += multiplier;
                let next = pack_pair(new_id, token[read + 2]);
                *count_delta.added.entry(next).or_insert(0) += multiplier;
            }
            // Now to remove`AB` itself
            let curr = pack_pair(pair[0], pair[1]);
            *count_delta.removed.entry(curr).or_insert(0) += multiplier;

            token[write] = new_id;
            read += 2;
        } else {
            token[write] = token[read];
            read += 1;
        }
        write += 1;
    }

    // finally discard the rest
    token.truncate(write);
}

fn replace_pair_in_freq_map(
    entries: &mut [(TokenIds, usize)],
    pair: &[TokenId; 2],
    new_id: TokenId,
    threads: usize,
) -> Result<PairCountDelta, CustomError> {
    let mut total_removed: HashMap<PackedPair, usize> = HashMap::new();
    let mut total_added: HashMap<PackedPair, usize> = HashMap::new();

    if entries.is_empty() {
        return Ok(PairCountDelta {
            removed: total_removed,
            added: total_added,
        });
    }

    let threads = threads.clamp(1, entries.len());
    let chunk_size = entries.len().div_ceil(threads);

    thread::scope(|scope| -> Result<PairCountDelta, CustomError> {
        let mut handles = vec![];
        for entries in entries.chunks_mut(chunk_size) {
            handles.push(scope.spawn(move || -> PairCountDelta {
                let mut thread_count_delta = PairCountDelta {
                    removed: HashMap::new(),
                    added: HashMap::new(),
                };
                for (token, count) in entries {
                    replace_pair_in_token(token, pair, new_id, *count, &mut thread_count_delta);
                }
                thread_count_delta
            }));
        }

        for handle in handles {
            let thread_delta = handle.join().map_err(|_| CustomError::ThreadPanic)?;
            merge_count_map(&mut total_removed, thread_delta.removed);
            merge_count_map(&mut total_added, thread_delta.added);
        }

        Ok(PairCountDelta {
            removed: total_removed,
            added: total_added,
        })
    })
}

fn apply_count_delta(all_pairs: &mut HashMap<PackedPair, usize>, delta: PairCountDelta) {
    // During the merges, there might be a chance to deal with this sequence:
    // `[67, 65, 66, 65, 66, 67]` and `[65, 66]` should be replaced by 300u16,
    // which will finally become: `[67, 300, 300, 67]`.
    //
    // The function `replace_pair_in_token` may produce for the first round:
    // removed: -> [67, 65], [65, 66], [66, 65]
    // added: -> [67, 300], [300, 65]
    //                      ~~~~~~~~~
    //
    // Then for the second round:
    // removed: -> [300, 65], [65, 66], [66, 67]
    //             ~~~~~~~~~
    //
    // added: -> [300, 300], [300, 67]
    //
    // It will be safer to deal with the diffs
    let all_keys: HashSet<PackedPair> = delta
        .removed
        .keys()
        .chain(delta.added.keys())
        .cloned()
        .collect();
    for k in all_keys {
        let del = delta.removed.get(&k).copied().unwrap_or(0);
        let add = delta.added.get(&k).copied().unwrap_or(0);

        if del == add {
            // do nothing
        } else if del > add {
            let count = all_pairs
                .get_mut(&k)
                .expect("pair count must exist before deletion");
            *count = count
                .checked_sub(del - add)
                .expect("pair count deletion underflow");

            if *count == 0 {
                all_pairs.remove(&k);
            }
        } else {
            *all_pairs.entry(k).or_insert(0) += add - del;
        }
    }
}

fn init_vocab() -> Vec<TokenBytes> {
    (0..=u8::MAX).map(|i| vec![i]).collect()
}

pub struct BpeTrainingResult {
    pub vocab: Vec<TokenBytes>,
    pub merges: Vec<(TokenBytes, TokenBytes)>,
}

pub fn train_bpe(
    freq_map: HashMap<TokenIds, usize>,
    max_vocab_size: usize,
    special_tokens: &[String],
    threads: usize,
) -> Result<BpeTrainingResult, CustomError> {
    let mut vocab = init_vocab();
    let mut merges: Vec<(TokenBytes, TokenBytes)> = Vec::new();
    let mut entries: Vec<(TokenIds, usize)> = freq_map.into_iter().collect();

    // TODO skip training
    let all_pairs = count_pairs(&entries, threads)?;
    if all_pairs.is_none() {
        return Ok(BpeTrainingResult { vocab, merges });
    }

    let mut all_pairs = all_pairs.unwrap();

    let largest_idx = (max_vocab_size.min(0x10000) - 1 - special_tokens.len()) as u16;

    for idx in 256..=largest_idx {
        if all_pairs.len() == 0 {
            // nothing to be merged
            break;
        }

        let pair = get_largest_pair(&all_pairs, &vocab);
        if pair.is_none() {
            // nothing found, shouldn't be here
            break;
        }

        let pair = pair.unwrap();
        let delta = replace_pair_in_freq_map(&mut entries, &pair, idx, threads)?;

        let a = vocab[pair[0] as usize].clone();
        let b = vocab[pair[1] as usize].clone();
        let c: TokenBytes = a.iter().chain(b.iter()).copied().collect();
        vocab.push(c);
        merges.push((a, b));

        apply_count_delta(&mut all_pairs, delta);
    }

    Ok(BpeTrainingResult { vocab, merges })
}

/// Adds counts from `source` into `target`.
fn merge_count_map<T>(target: &mut HashMap<T, usize>, source: HashMap<T, usize>)
where
    T: Eq + std::hash::Hash,
{
    for (k, v) in source {
        *target.entry(k).or_insert(0) += v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_special_tokens_should_return_the_smallest_occurance() {
        let data: TokenBytes = vec![1, 2, 3, 4, 5];
        let special_tokens_bytes: Vec<TokenBytes> = vec![vec![8], vec![5], vec![3]];

        let offset = find_special_tokens(&data, &special_tokens_bytes);
        assert_eq!(offset, Some(2));
    }

    #[test]
    fn test_find_special_tokens_should_return_none() {
        let data: TokenBytes = vec![1, 2, 3, 4, 5];
        let special_tokens_bytes: Vec<TokenBytes> = vec![vec![8]];
        let offset = find_special_tokens(&data, &special_tokens_bytes);
        assert!(offset.is_none());
    }

    #[test]
    fn test_find_special_tokens_with_empty_special_tokens() {
        let data: TokenBytes = vec![1, 2, 3, 4, 5];
        let special_tokens_bytes: Vec<TokenBytes> = vec![];
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
        let mut count_map: HashMap<PackedPair, usize> = HashMap::new();

        count_map.insert(pack_pair('l' as u16, 'o' as u16), 7);
        count_map.insert(pack_pair('o' as u16, 'w' as u16), 7);
        count_map.insert(pack_pair('w' as u16, 'e' as u16), 8);
        count_map.insert(pack_pair('e' as u16, 'r' as u16), 2);
        count_map.insert(pack_pair('w' as u16, 'i' as u16), 3);
        count_map.insert(pack_pair('i' as u16, 'd' as u16), 3);
        count_map.insert(pack_pair('d' as u16, 'e' as u16), 3);
        count_map.insert(pack_pair('e' as u16, 's' as u16), 9);
        count_map.insert(pack_pair('s' as u16, 't' as u16), 9);
        count_map.insert(pack_pair('n' as u16, 'e' as u16), 6);
        count_map.insert(pack_pair('e' as u16, 'w' as u16), 6);

        let largest = get_largest_pair(&count_map, &vocab);
        assert_eq!(largest, Some(['s' as u16, 't' as u16]));
    }

    #[test]
    fn test_train_bpe() {
        let text = b"aaabdaaabac";
        let text_vec: Vec<u16> = text.iter().map(|&b| b as u16).collect();
        let mut freq_map: HashMap<TokenIds, usize> = HashMap::new();
        freq_map.insert(text_vec, 1);
        let special_tokens = Vec::new();

        // 1
        let x = freq_map.clone();
        let result = train_bpe(x, 256 + 1, &special_tokens, 1).expect("shouln't throw error");
        assert_eq!(result.vocab.len(), 256 + 1);
        assert_eq!(result.vocab[256 + 0], b"aa".to_vec());
        assert_eq!(result.merges.len(), 1);
        assert_eq!(result.merges[0], (b"a".to_vec(), b"a".to_vec()));

        // 2
        let x = freq_map.clone();
        let result = train_bpe(x, 256 + 2, &special_tokens, 1).expect("shouln't throw error");
        assert_eq!(result.vocab.len(), 256 + 2);
        assert_eq!(result.vocab[256 + 0], b"aa".to_vec());
        assert_eq!(result.vocab[256 + 1], b"aaa".to_vec());
        assert_eq!(result.merges.len(), 2);
        assert_eq!(result.merges[0], (b"a".to_vec(), b"a".to_vec()));
        assert_eq!(result.merges[1], (b"aa".to_vec(), b"a".to_vec()));

        // 3
        let x = freq_map.clone();
        let result = train_bpe(x, 256 + 3, &special_tokens, 1).expect("shouln't throw error");
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
        let data: Vec<(TokenBytes, TokenBytes)> = vec![
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

    #[test]
    fn test_replace_pair_in_token() {
        let mut delta = PairCountDelta {
            removed: HashMap::new(),
            added: HashMap::new(),
        };
        let mut token: TokenIds = vec![67, 65, 66, 65, 66, 67];
        let pair: [TokenId; 2] = [65, 66];
        replace_pair_in_token(&mut token, &pair, 300, 1, &mut delta);

        assert_eq!(delta.removed.len(), 5);
        let removed_entries: [(PackedPair, usize); 5] = [
            (pack_pair(67, 65), 1),
            (pack_pair(65, 66), 2),
            (pack_pair(66, 65), 1),
            (pack_pair(66, 67), 1),
            (pack_pair(300, 65), 1),
        ];
        for (k, v) in removed_entries {
            assert_eq!(delta.removed.get(&k), Some(&v));
        }

        let added_entries: [(PackedPair, usize); 4] = [
            (pack_pair(67, 300), 1),
            (pack_pair(300, 65), 1),
            (pack_pair(300, 300), 1),
            (pack_pair(300, 67), 1),
        ];
        for (k, v) in added_entries {
            assert_eq!(delta.added.get(&k), Some(&v));
        }
    }
}
