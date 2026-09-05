//! The chat template (Phase 5.3): `chat_template.jinja` rendered with minijinja the way HF's
//! `apply_chat_template` renders it (`trim_blocks`, `lstrip_blocks`, no trailing newline kept), with the two
//! callables the template uses that Jinja does not provide: `raise_exception` (HF registers it) and a
//! `tojson` filter that formats like Python's `json.dumps(x, ensure_ascii=False)` (HF overrides Jinja's
//! HTML-escaping one). Template kwargs the engine drives: `enable_thinking` (undefined = on), `reasoning_effort`
//! (`xhigh` default, `medium`, `low`), `preserve_thinking`, `add_generation_prompt`. Facts and the two
//! reference renderings are in `docs/chat-template.md`; `tests/chat_template.rs` checks byte identity with HF
//! on those two and five multi-turn cases (`tests/fixtures/chat_cases.json`).

use std::collections::BTreeMap;
use std::path::Path;

use minijinja::value::{Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind};

#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("template: {0}")]
    Template(#[from] minijinja::Error),
    #[error("read {path}: {err}")]
    Read { path: String, err: std::io::Error },
}

/// One turn. `reasoning_content` is the assistant's thinking of a prior turn (kept or dropped by the template
/// per `preserve_thinking`).
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Message {
        Message { role: role.into(), content: content.into(), reasoning_content: None }
    }
}

/// What the render is asked for. `None` leaves the template kwarg undefined (its default branch).
#[derive(Debug, Clone, PartialEq)]
pub struct RenderOpts {
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub preserve_thinking: Option<bool>,
    pub add_generation_prompt: bool,
}

impl Default for RenderOpts {
    fn default() -> Self {
        RenderOpts { enable_thinking: None, reasoning_effort: None, preserve_thinking: None, add_generation_prompt: true }
    }
}

pub struct ChatTemplate {
    env: Environment<'static>,
    pub source: String,
}

impl ChatTemplate {
    pub fn from_file(path: impl AsRef<Path>) -> Result<ChatTemplate, ChatError> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path).map_err(|err| ChatError::Read { path: path.display().to_string(), err })?;
        Self::from_source(source)
    }

    pub fn from_source(source: String) -> Result<ChatTemplate, ChatError> {
        let mut env = Environment::new();
        // HF: ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True); keep_trailing_newline default False
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> { Err(Error::new(ErrorKind::InvalidOperation, msg)) });
        // Python string methods the template calls (minijinja has no built-in ones): only what it uses
        env.set_unknown_method_callback(|_state, value, method, args| -> Result<Value, Error> {
            let s = value.as_str().ok_or_else(|| Error::new(ErrorKind::UnknownMethod, format!("{method} on a non-string")))?;
            let arg = || -> Result<String, Error> {
                args.first().and_then(|a| a.as_str()).map(String::from).ok_or_else(|| Error::new(ErrorKind::InvalidOperation, format!("{method}: one string argument expected")))
            };
            match method {
                "startswith" => Ok(Value::from(s.starts_with(arg()?.as_str()))),
                "endswith" => Ok(Value::from(s.ends_with(arg()?.as_str()))),
                "strip" => Ok(Value::from(s.trim())),
                _ => Err(Error::new(ErrorKind::UnknownMethod, format!("string has no method named {method}"))),
            }
        });
        env.add_filter("tojson", |v: Value| -> Result<String, Error> {
            let mut s = String::new();
            py_json(&v, &mut s)?;
            Ok(s)
        });
        env.add_template_owned("chat", source.clone())?;
        Ok(ChatTemplate { env, source })
    }

    pub fn render(&self, messages: &[Message], opts: &RenderOpts) -> Result<String, ChatError> {
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| {
                let mut map: BTreeMap<String, Value> = BTreeMap::new();
                map.insert("role".into(), Value::from(m.role.as_str()));
                map.insert("content".into(), Value::from(m.content.as_str()));
                if let Some(r) = &m.reasoning_content {
                    map.insert("reasoning_content".into(), Value::from(r.as_str()));
                }
                Value::from(map)
            })
            .collect();
        let mut ctx: BTreeMap<String, Value> = BTreeMap::new();
        ctx.insert("messages".into(), Value::from(msgs));
        ctx.insert("add_generation_prompt".into(), Value::from(opts.add_generation_prompt));
        if let Some(b) = opts.enable_thinking {
            ctx.insert("enable_thinking".into(), Value::from(b));
        }
        if let Some(e) = &opts.reasoning_effort {
            ctx.insert("reasoning_effort".into(), Value::from(e.as_str()));
        }
        if let Some(b) = opts.preserve_thinking {
            ctx.insert("preserve_thinking".into(), Value::from(b));
        }
        let t = self.env.get_template("chat")?;
        Ok(t.render(Value::from(ctx))?)
    }
}

/// Python `json.dumps(v, ensure_ascii=False)`: `, ` and `: ` separators, no key sorting, non-ASCII verbatim.
fn py_json(v: &Value, out: &mut String) -> Result<(), Error> {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => {
            if let Some(i) = v.as_i64() {
                out.push_str(&i.to_string());
            } else {
                let f = f64::try_from(v.clone()).map_err(|_| Error::new(ErrorKind::InvalidOperation, "tojson: bad number"))?;
                if f.fract() == 0.0 && f.abs() < 1e16 {
                    out.push_str(&format!("{f:.1}"));
                } else {
                    out.push_str(&f.to_string());
                }
            }
        }
        ValueKind::String => py_json_str(v.as_str().unwrap_or(""), out),
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            for (i, item) in v.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json(&item, out)?;
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (i, key) in v.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json_str(key.as_str().unwrap_or(&key.to_string()), out);
                out.push_str(": ");
                py_json(&v.get_item(&key)?, out)?;
            }
            out.push('}');
        }
        other => return Err(Error::new(ErrorKind::InvalidOperation, format!("tojson: cannot serialise a {other:?}"))),
    }
    Ok(())
}

fn py_json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tojson_formats_like_python() {
        let t = ChatTemplate::from_source("{{ x | tojson }}".into()).unwrap();
        let mut m: BTreeMap<String, Value> = BTreeMap::new();
        m.insert("b".into(), Value::from(vec![Value::from(1), Value::from("é\n")]));
        m.insert("a".into(), Value::from(true));
        let mut ctx: BTreeMap<String, Value> = BTreeMap::new();
        ctx.insert("x".into(), Value::from(m));
        let s = t.env.get_template("chat").unwrap().render(Value::from(ctx)).unwrap();
        assert_eq!(s, "{\"a\": true, \"b\": [1, \"é\\n\"]}");
    }
}
