//! Tokenizer wrapper over the `tokenizers` crate (HF `tokenizer.json`, byte-level BPE).
//!
//! Encoding never adds BOS/EOS (`add_special_tokens = false`), matching `tools/tok_cases.py` and
//! `tools/prompts.py`; decoding keeps special tokens. Special ids come from GGUF metadata through
//! `SpecialIds::from_config`, not from the tokenizer file.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::config::ModelConfig;

#[derive(Debug, thiserror::Error)]
pub enum TokError {
    #[error("tokenizer: {0}")]
    Lib(String),
    #[error("tokenizer file {0}: {1}")]
    Load(String, String),
    #[error("tokenizer has no token {0:?}")]
    NoToken(String),
    #[error("tokenizer id {id} for {what} is {found:?} in tokenizer.json but GGUF metadata expects {expected:?}")]
    IdMismatch { what: &'static str, id: u32, found: Option<String>, expected: String },
}

pub type Result<T> = std::result::Result<T, TokError>;

/// Special token ids, taken from GGUF metadata (see `ModelConfig`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialIds {
    pub bos: Option<u32>,
    /// Every id that ends generation. GGUF lists `tokenizer.ggml.eos_token_id` (and `eot_token_id` if present).
    pub eos: Vec<u32>,
    pub pad: Option<u32>,
    /// Whether a BOS token is prepended when encoding a prompt (`tokenizer.ggml.add_bos_token`).
    pub add_bos: bool,
}

impl SpecialIds {
    pub fn from_config(c: &ModelConfig) -> Self {
        SpecialIds { bos: Some(c.bos_id), eos: c.eos_ids.clone(), pad: Some(c.pad_id), add_bos: c.add_bos }
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }
}

pub struct Tok {
    inner: Tokenizer,
}

impl Tok {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let inner = Tokenizer::from_file(p).map_err(|e| TokError::Load(p.display().to_string(), e.to_string()))?;
        Ok(Tok { inner })
    }

    pub fn from_bytes(json: &[u8]) -> Result<Self> {
        let inner = Tokenizer::from_bytes(json).map_err(|e| TokError::Lib(e.to_string()))?;
        Ok(Tok { inner })
    }

    /// Raw ids for `text`; no BOS/EOS added. Literal special-token text (e.g. `<|im_start|>`) maps to its id,
    /// as in the HF tokenizer.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, false).map_err(|e| TokError::Lib(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Text for `ids`; special tokens are kept in the output.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, false).map_err(|e| TokError::Lib(e.to_string()))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    pub fn vocab_size_with_added(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Check that the ids named in GGUF metadata refer to the expected token strings in tokenizer.json.
    pub fn check_special(&self, ids: &SpecialIds, expect: &[(u32, &'static str, &str)]) -> Result<()> {
        let _ = ids;
        for &(id, what, text) in expect {
            let found = self.id_to_token(id);
            if found.as_deref() != Some(text) {
                return Err(TokError::IdMismatch { what, id, found, expected: text.to_string() });
            }
        }
        Ok(())
    }
}
