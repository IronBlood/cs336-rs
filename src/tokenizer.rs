use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    thread,
};

use crate::{
    error::CustomError,
    file::{load_merges, load_vocab},
    regex::{Regex, escape_literal},
    types::{PackedPair, TokenBytes, TokenId, TokenIds},
    utils::{init_vocab, pack_pair},
};

struct MergeRule {
    /// lOwer rank means the merge has higher priority
    rank: usize,
    new_id: TokenId,
}

pub struct Tokenizer {
    // both raw and for decoding
    decoder: HashMap<TokenId, TokenBytes>,
    // for encoding
    special_tokens_encoder: HashMap<String, TokenId>,
    // this should be built from merges for encoding
    encoder: HashMap<PackedPair, MergeRule>,
    // this converts raw bytes to TokenId
    byte_encoder: HashMap<u8, TokenId>,
    pretokenize_regex: Regex,
    special_regex: Option<Regex>,
}

/// This function assumes merges start from index 256 in a vocab
fn build_encoder(
    vocab_map: &HashMap<&TokenBytes, TokenId>,
    merges: &[(TokenBytes, TokenBytes)],
) -> HashMap<PackedPair, MergeRule> {
    let mut map = HashMap::new();
    for (rank, (a, b)) in merges.iter().enumerate() {
        let a_id = *vocab_map.get(a).expect("token should exist");
        let b_id = *vocab_map.get(b).expect("token should exist");
        let pair = pack_pair(a_id, b_id);
        let merged: TokenBytes = a.iter().chain(b.iter()).copied().collect();
        let new_id = *vocab_map
            .get(&merged)
            .expect("merged token should exist in vocab");
        map.insert(pair, MergeRule { rank, new_id });
    }
    map
}

/// This function returns a HashMap, which is used to convert the raw bytes to token ids
///
/// Not all initial vocabularies (the first 256) are placed in the same order, this encoder
/// should be used before encoding.
fn build_byte_encoder(vocab: &HashMap<TokenId, TokenBytes>) -> HashMap<u8, TokenId> {
    (0..256)
        .map(|id| {
            let token_bytes = vocab.get(&id).expect("id should exist");
            if token_bytes.len() != 1 {
                panic!("vocab < 256 should be single character");
            }
            (token_bytes[0], id)
        })
        .collect()
}

impl Tokenizer {
    pub fn load(
        vocab_path: &str,
        merges_path: &str,
        special_tokens: &[String],
        pretokenize_regex_str: &str,
    ) -> Result<Self, CustomError> {
        let vocab_path = PathBuf::from(vocab_path);
        let merges_path = PathBuf::from(merges_path);

        let vocab = load_vocab(&vocab_path);
        let merges = load_merges(&merges_path);

        let mut special_token_map: HashMap<String, TokenId> = HashMap::new();
        {
            let id = vocab.len();
            if id + special_tokens.len() > (u16::MAX as usize) {
                panic!("invalid vocab and special token size");
            }

            let mut id = id as u16;
            for special_token in special_tokens {
                special_token_map.insert(special_token.clone(), id);
                id += 1;
            }
        }

        let vocab_encoder_map: HashMap<_, _> = vocab
            .iter()
            .enumerate()
            .map(|(id, bytes)| (bytes, id as TokenId))
            .collect();
        let encoder = build_encoder(&vocab_encoder_map, &merges);
        let byte_encoder: HashMap<_, _> = init_vocab()
            .into_iter()
            .enumerate()
            .map(|(id, bytes)| {
                // protections
                if bytes.len() != 1 {
                    panic!("should be single character");
                }
                if id > (u8::MAX as usize) {
                    eprintln!("invalid id {}", id);
                    panic!("invalid id");
                }

                (bytes[0], id as TokenId)
            })
            .collect();

        let special_regex = {
            let pat = special_tokens
                .iter()
                .filter(|st| !st.is_empty())
                .map(|s| escape_literal(s))
                .collect::<Vec<_>>()
                .join("|");
            Some(Regex::new(&pat)?)
        };

        let pretokenize_regex = Regex::new(pretokenize_regex_str)?;

        let decoder: HashMap<_, _> = vocab
            .into_iter()
            .enumerate()
            .map(|(id, bytes)| (id as u16, bytes))
            .collect();

        Ok(Tokenizer {
            decoder,
            encoder,
            special_tokens_encoder: special_token_map,
            byte_encoder,
            pretokenize_regex,
            special_regex,
        })
    }

    // The test cases used in CS336 isn't in the best shape, so this function
    // transform to the internal data structures
    pub fn load_course(
        vocab: HashMap<u16, TokenBytes>,
        merges: Vec<(TokenBytes, TokenBytes)>,
        special_tokens: Option<&[String]>,
        pretokenize_regex_str: &str,
    ) -> Result<Self, CustomError> {
        let mut special_token_map: HashMap<String, TokenId> = HashMap::new();
        {
            // this block extracts special tokens from `Vec<u8>` to `String`
            // and stores to `special_token_map` for easier searching and encoding.
            // special token bytes remain in the vocab for decoding
            if let Some(special_tokens) = special_tokens {
                // NOTE: this copy should be fine since usually there are not a lot of special tokens
                let special_tokens_set: HashSet<Vec<u8>> = special_tokens
                    .iter()
                    .map(|t| t.as_bytes().to_vec())
                    .collect();
                let special_token_counts = special_tokens.len();
                let total_vocab_len = vocab.len();
                // TODO: should use better error handling
                if special_token_counts >= (u16::MAX as usize)
                    || total_vocab_len >= (u16::MAX as usize)
                    || special_token_counts > total_vocab_len
                {
                    panic!("invalid inputs");
                }

                // now this should be valid
                let special_token_counts = special_token_counts as u16;
                let total_vocab_len = total_vocab_len as u16;
                for idx in (total_vocab_len - special_token_counts)..total_vocab_len {
                    // the course's test code mixed special tokens inside vocab
                    if let Some(bytes) = vocab.get(&idx) {
                        if !special_tokens_set.contains(bytes) {
                            panic!("invalid inputs");
                        }

                        let special_token = str::from_utf8(bytes)
                            .expect("should be valid UTF-8")
                            .to_string();

                        special_token_map.insert(special_token, idx);
                    } else {
                        panic!("{idx} should be a valid special token");
                    }
                }
            }
        }

        let vocab_encoder_map: HashMap<_, _> = vocab.iter().map(|(i, b)| (b, *i)).collect();
        let encoder = build_encoder(&vocab_encoder_map, &merges);
        let byte_encoder = build_byte_encoder(&vocab);

        let special_regex = if let Some(special_tokens) = special_tokens {
            let pat = special_tokens
                .iter()
                .filter(|st| !st.is_empty())
                .map(|s| escape_literal(s))
                .collect::<Vec<_>>()
                .join("|");
            Some(Regex::new(&pat)?)
        } else {
            None
        };

        let pretokenize_regex = Regex::new(pretokenize_regex_str)?;

        Ok(Tokenizer {
            decoder: vocab,
            encoder,
            special_tokens_encoder: special_token_map,
            byte_encoder,
            pretokenize_regex,
            special_regex,
        })
    }

    // TODO: encode in parallel
    pub fn encode(&self, content: &str) -> Result<TokenIds, CustomError> {
        let mut result: TokenIds = Vec::new();

        if let Some(special_re) = &self.special_regex {
            let mut cursor = 0;
            for mat in special_re.find_iter(content)? {
                let mat = mat?;
                self.encode_normal(
                    &self.pretokenize_regex,
                    &content[cursor..mat.start()],
                    &mut result,
                )?;

                let special = &content[mat.start()..mat.end()];
                let id = self
                    .special_tokens_encoder
                    .get(special)
                    .expect("special token should exist");
                result.push(*id);
                cursor = mat.end();
            }

            self.encode_normal(&self.pretokenize_regex, &content[cursor..], &mut result)?;
        } else {
            self.encode_normal(&self.pretokenize_regex, &content, &mut result)?;
        }

        Ok(result)
    }

    pub fn encode_mt(&self, content: &str, threads: usize) -> Result<TokenIds, CustomError> {
        let mut result: TokenIds = Vec::new();
        let mut encoded_token_cache: HashMap<&str, TokenIds> = HashMap::new();

        if let Some(special_re) = &self.special_regex {
            let mut cursor = 0;
            for mat in special_re.find_iter(content)? {
                let mat = mat?;
                self.encode_normal_mt(
                    &self.pretokenize_regex,
                    &content[cursor..mat.start()],
                    &mut result,
                    threads,
                    &mut encoded_token_cache,
                )?;

                let special = &content[mat.start()..mat.end()];
                let id = self
                    .special_tokens_encoder
                    .get(special)
                    .expect("special token should exist");
                result.push(*id);
                cursor = mat.end();
            }

            self.encode_normal_mt(
                &self.pretokenize_regex,
                &content[cursor..],
                &mut result,
                threads,
                &mut encoded_token_cache,
            )?;
        } else {
            self.encode_normal_mt(
                &self.pretokenize_regex,
                &content,
                &mut result,
                threads,
                &mut encoded_token_cache,
            )?;
        }

        Ok(result)
    }

    fn encode_normal(
        &self,
        regex: &Regex,
        content: &str,
        result: &mut TokenIds,
    ) -> Result<(), CustomError> {
        for mat in regex.find_iter(content)? {
            let mat = mat?;
            let s = &content[mat.start()..mat.end()];
            let mut buf: TokenIds = s
                .as_bytes()
                .iter()
                .map(|b| {
                    let b = self.byte_encoder.get(b).expect("character should exist");
                    *b
                })
                .collect();
            self.encode_internal(&mut buf);
            result.extend(buf);
        }
        Ok(())
    }

    fn encode_normal_mt<'content>(
        &self,
        regex: &Regex,
        content: &'content str,
        result: &mut TokenIds,
        threads: usize,
        encoded_token_cache: &mut HashMap<&'content str, TokenIds>,
    ) -> Result<(), CustomError> {
        let mut str_to_be_encoded = Vec::new();
        let mut unique_string_slices = HashSet::new();

        for mat in regex.find_iter(content)? {
            let mat = mat?;
            let s = &content[mat.start()..mat.end()];
            str_to_be_encoded.push(s);
            if !encoded_token_cache.contains_key(s) {
                unique_string_slices.insert(s);
            }
        }

        self.encode_unique_tokens(unique_string_slices, threads, encoded_token_cache)?;

        for s in str_to_be_encoded {
            let ids = encoded_token_cache.get(s).expect("token should exist");
            result.extend(ids.iter().copied());
        }

        Ok(())
    }

    fn encode_unique_tokens<'content>(
        &self,
        tokens: HashSet<&'content str>,
        threads: usize,
        encoded_token_cache: &mut HashMap<&'content str, TokenIds>,
    ) -> Result<(), CustomError> {
        let chunk_size = tokens.len().div_ceil(threads).max(1);
        let tokens: Vec<&str> = tokens.into_iter().collect();
        thread::scope(|scope| -> Result<(), CustomError> {
            let mut handles = Vec::new();
            for chunk in tokens.chunks(chunk_size) {
                handles.push(scope.spawn(move || -> HashMap<&'content str, TokenIds> {
                    chunk
                        .iter()
                        .map(|&s| {
                            let mut buf: TokenIds = s
                                .as_bytes()
                                .iter()
                                .map(|b| {
                                    let b =
                                        self.byte_encoder.get(b).expect("character should exist");
                                    *b
                                })
                                .collect();
                            self.encode_internal(&mut buf);
                            (s, buf)
                        })
                        .collect()
                }));
            }

            for handle in handles {
                let thread_cache = handle.join().map_err(|_| CustomError::ThreadPanic)?;
                for (k, v) in thread_cache.into_iter() {
                    encoded_token_cache.insert(k, v);
                }
            }

            Ok(())
        })
    }

    // NOTE: this is a naive implementation
    // Inside of the `loop` every pair is computed unless no more pair can
    // be merged. This is fine for this course, since the raw text could
    // be split into short chunks, so this naive approach is easy to
    // implement. In OpenAI's `tiktoken`, BinaryHeap is used. A heap-based
    // implementation avoids rescanning the whole token list on each merge
    // by updating only candidate pairs adjacent to the replacement. (A
    // `Vec<>` can be used as a double linked list.) This approach reduces
    // the time complexity from O(N ^ 2) to O(N log N).
    fn encode_internal(&self, buf: &mut Vec<TokenId>) {
        // len holds the actual length
        let mut len = buf.len();

        loop {
            let mut min_pair: Option<Replacement> = None;
            if len < 2 {
                break;
            }

            for i in 0..(len - 1) {
                let a = buf[i];
                let b = buf[i + 1];
                let pair = pack_pair(a, b);
                if let Some(rule) = self.encoder.get(&pair) {
                    match min_pair.as_mut() {
                        Some(x) => {
                            if rule.rank < x.rank {
                                x.pair = pair;
                                x.rank = rule.rank;
                                x.new_id = rule.new_id;
                            }
                        }
                        None => {
                            min_pair = Some(Replacement {
                                pair,
                                rank: rule.rank,
                                new_id: rule.new_id,
                            })
                        }
                    }
                }
            }

            if let Some(min_pair) = &min_pair {
                let count = merge_token_with_id(buf, len, min_pair);
                len -= count;
            } else {
                // no more to be merged
                break;
            }
        }

        buf.truncate(len);
    }

    // TODO: decode in parallel
    pub fn decode(&self, ids: &[TokenId]) -> String {
        let mut buf = Vec::new();
        for id in ids {
            buf.extend(self.decoder.get(id).expect("id should exist").clone());
        }
        // TODO: Better error handling
        String::from_utf8(buf).expect("should be a valid UTF-8 string")
    }
}

struct Replacement {
    pair: PackedPair,
    rank: usize,
    new_id: TokenId,
}

// This helper replaces a pair in place
//
// TODO: it should be the similar or exact the same as the merge step in training
fn merge_token_with_id(buf: &mut Vec<TokenId>, len: usize, replacement: &Replacement) -> usize {
    let mut read = 0;
    let mut write = 0;
    let mut count = 0;

    while read + 1 < len {
        let a = buf[read];
        let b = buf[read + 1];
        let pair = pack_pair(a, b);
        if pair == replacement.pair {
            buf[write] = replacement.new_id;
            read += 2;
            count += 1;
        } else {
            buf[write] = a;
            read += 1;
        }
        write += 1;
    }

    if read == len - 1 {
        buf[write] = buf[len - 1];
    }

    return count;
}
