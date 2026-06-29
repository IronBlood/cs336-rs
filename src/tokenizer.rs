use crate::{
    error::CustomError,
    file::{parse_merges, parse_vocab},
    types::TokenBytes,
    utils::{
        build_token_freq_map, convert_freq_map_to_u16, find_chunk_boundaries, find_pretoken_spans,
        train_bpe,
    },
};

pub struct Tokenizer {
    vocab: Vec<TokenBytes>,
    merges: Vec<(TokenBytes, TokenBytes)>,
    special_tokens: Vec<String>,
}

impl Tokenizer {
    pub fn train(
        content: &[u8],
        vocab_size: u16,
        special_tokens: &[String],
        threads: usize,
        pretokenize_regex_str: &str,
    ) -> Result<Self, CustomError> {
        let boundaries = find_chunk_boundaries(content, threads, &special_tokens);
        let mut spans =
            find_pretoken_spans(content, &boundaries, &special_tokens).expect("should succeed");
        spans.sort();
        let all_pieces: Vec<_> = spans.into_iter().flatten().collect();
        let freq_map =
            build_token_freq_map(&content, &all_pieces, threads, &pretokenize_regex_str)?;
        let freq_map = convert_freq_map_to_u16(freq_map);
        let result = train_bpe(freq_map, vocab_size as usize, &special_tokens, threads)?;
        Ok(Tokenizer {
            vocab: result.vocab,
            merges: result.merges,
            special_tokens: special_tokens.iter().map(|s| s.clone()).collect(),
        })
    }

    pub fn load(vocab_content: &str, merges_content: &str, special_tokens: &[String]) -> Self {
        Tokenizer {
            vocab: parse_vocab(vocab_content),
            merges: parse_merges(merges_content),
            special_tokens: special_tokens.iter().map(|s| s.clone()).collect(),
        }
    }
}
