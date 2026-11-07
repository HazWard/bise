//! What a provider's `models_url` names (the provider's own model
//! list), pure: parse each API's answer to model entries, and keep the
//! raw answers in a store (the text of `<cache>/provider-models.json`).
//! The HTTP and the file are the caller's ([`live_fetch::refresh`], the
//! hub's thread), like [`live_efforts`] and [`voice_live`]. A fetched
//! list only adds models the built-in and config tables do not list:
//! those two stay in charge of every model they name. A failure (no
//! network, an error status, an answer that is not a list of ids) keeps
//! the old store until its next fetch.
//!
//! The answers read today (an OpenAI-compatible `{"data": [...]}`, the
//! shape opencode's zen Go gives):
//! - `id` (the only field a plain gateway lists, the local ones included)
//! - `limit.context` / `context_length`, `limit.output` / `max_output_tokens`
//! - image input: `modalities.input` or `architecture.input_modalities`
//! - reasoning: `capabilities.reasoning.supported`; its effort words:
//!   `capabilities.reasoning.effort`, then OpenRouter's
//!   `reasoning.supported_efforts`
//! - tools: `capabilities.tools.supported`
//! - prices, USD per million tokens: `cost.text.input.price` (+ its
//!   `cache.read` and `cache.write`), `cost.text.output.price`

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::{Catalog, Model, PartialCaps, Source};

/// One model of a fetched list: its id and what the answer said about
/// it (the keys it leaves out come from its provider's table).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    pub id: String,
    pub caps: PartialCaps,
}

/// A provider's list is fetched again after a day.
pub const TTL_MS: u64 = 24 * 3600 * 1000;

/// `<cache>/provider-models.json`: the providers' fetched answers, each
/// with the time it was fetched.
pub fn store_path(cache_dir: &Path) -> std::path::PathBuf {
    cache_dir.join("provider-models.json")
}

/// The answers fetched so far, per provider. The raw answers, not
/// parsed lists ([`live_efforts`]' shape): the parser may grow without
/// a refetch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Store {
    /// provider id -> the time of its last fetch, in ms
    fetched: BTreeMap<String, u64>,
    /// provider id -> its model list as the API answered it
    raw: BTreeMap<String, Value>,
}

impl Store {
    /// The stored store, empty when there is none (or it is not one).
    pub fn read(cache_dir: &Path) -> Store {
        std::fs::read_to_string(store_path(cache_dir))
            .ok()
            .and_then(|t| Store::parse(&t))
            .unwrap_or_default()
    }

    pub fn parse(text: &str) -> Option<Store> {
        let v: Value = serde_json::from_str(text).ok()?;
        let mut fetched = BTreeMap::new();
        for (id, n) in v.get("fetched")?.as_object()? {
            fetched.insert(id.clone(), n.as_u64()?);
        }
        let mut raw = BTreeMap::new();
        for (id, n) in v.get("raw")?.as_object()? {
            raw.insert(id.clone(), n.clone());
        }
        Some(Store { fetched, raw })
    }

    pub fn to_json(&self) -> String {
        json!({"fetched": self.fetched, "raw": self.raw}).to_string()
    }

    /// Whether `provider`'s list is due: it has none, or its last fetch
    /// is past [`TTL_MS`].
    pub fn due(&self, provider: &str, now_ms: u64) -> bool {
        self.fetched.get(provider).is_none_or(|t| now_ms.saturating_sub(*t) >= TTL_MS)
    }

    /// A new fetch of `provider`: its answer replaces the old one.
    pub fn with(mut self, provider: &str, now_ms: u64, answer: Value) -> Store {
        self.fetched.insert(provider.to_string(), now_ms);
        self.raw.insert(provider.to_string(), answer);
        self
    }
}

/// The stored answers, one `(provider, answer)` pair per provider. An
/// answer of a provider that names no `models_url` is refused below,
/// not here: an answer is only for the provider that asked for it.
pub fn read(cache_dir: &Path) -> Vec<(String, Value)> {
    Store::read(cache_dir).raw.into_iter().collect()
}

/// An answer's model entries, in `data`'s order. An entry without an
/// id is skipped; one whose `type` is not `"chat"` too (a transcription
/// or image model: not for the chat lists). Every other field stays
/// what the answer says, a left-out key included.
pub fn parse(answer: &Value) -> Vec<Entry> {
    let Some(data) = answer.get("data").and_then(Value::as_array) else { return Vec::new() };
    data.iter()
        .filter(|m| m.get("type").and_then(Value::as_str).is_none_or(|t| t == "chat"))
        .filter_map(|m| {
            let id = m.get("id").and_then(Value::as_str)?.trim().to_string();
            (!id.is_empty()).then_some(Entry { id, caps: caps_of(m) })
        })
        .collect()
}

/// One model's caps as the answer describes it: a key the answer leaves
/// out stays unpart (its provider's table, then the defaults).
fn caps_of(m: &Value) -> PartialCaps {
    let mut c = PartialCaps::default();
    let limit = m.get("limit");
    c.context = int(limit.and_then(|l| l.get("context"))).or_else(|| int(m.get("context_length")));
    c.max_output = int(limit.and_then(|l| l.get("output"))).or_else(|| int(m.get("max_output_tokens")));
    // an input list with an image format: it reads images; a list
    // without one: it does not; no list: its provider's
    let go_inputs = texts(m.get("modalities").and_then(|v| v.get("input")));
    let router_inputs = texts(m.get("architecture").and_then(|v| v.get("input_modalities")));
    c.vision = [go_inputs, router_inputs]
        .into_iter()
        .find(|l| !l.is_empty())
        .map(|l| l.iter().any(|x| x.starts_with("image/")));
    let reasoning = m.get("capabilities").and_then(|v| v.get("reasoning"));
    let words = match reasoning.map(|v| texts(v.get("effort"))) {
        Some(l) if !l.is_empty() => Some(l),
        _ => {
            let or = texts(m.get("reasoning").and_then(|v| v.get("supported_efforts")));
            (!or.is_empty()).then_some(or)
        }
    };
    // OpenRouter's shape: a reasoning object with effort words, and no
    // supported flag
    c.reasoning = reasoning
        .and_then(|v| bool_of(v.get("supported")))
        .or_else(|| words.as_ref().map(|_| true));
    c.efforts = words.map(|l| crate::live_efforts::ordered(l).join(",").clone()).filter(|s| !s.is_empty());
    c.tools = m.get("capabilities").and_then(|v| v.get("tools")).and_then(|v| bool_of(v.get("supported")));
    // prices, USD per million tokens -> millionths of a dollar
    // ([`cap_field`]'s unit and range: 0 .. 1e6)
    let cost = m.get("cost").and_then(|v| v.get("text"));
    let price_at = |node: Option<&Value>, (a, b): (&str, &str)| -> Option<u64> {
        let v = node.and_then(|v| v.get(a))?;
        let v = if b.is_empty() { v } else { v.get(b)? };
        v.as_f64().and_then(usd)
    };
    let (input, output) = (cost.and_then(|v| v.get("input")), cost.and_then(|v| v.get("output")));
    c.input_price = price_at(input, ("price", ""));
    c.output_price = price_at(output, ("price", ""));
    c.cache_read_price = price_at(input, ("cache", "read"));
    c.cache_write_price = price_at(input, ("cache", "write"));
    c
}

/// USD per million -> millionths of a dollar, out of range: None
/// ([`cap_field`]'s range).
fn usd(x: f64) -> Option<u64> {
    (x.is_finite() && x >= 0.0 && x < 1e6).then(|| (x * 1e6).round() as u64)
}

/// An integer of `node`, > 0 ([`cap_field`]'s rule).
fn int(node: Option<&Value>) -> Option<u64> {
    node.and_then(Value::as_u64).filter(|n| *n > 0)
}

/// An array of strings at `node` (a field that is not one: none).
fn texts(node: Option<&Value>) -> Vec<String> {
    node.and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|x| x.as_str().map(str::trim).filter(|s| !s.is_empty()).map(String::from))
        .collect()
}

fn bool_of(v: Option<&Value>) -> Option<bool> {
    v.and_then(Value::as_bool)
}

impl Catalog {
    /// The fetched lists over the tables, under them: a listed model
    /// (built-in or config) keeps its entry, an unlisted one gets the
    /// list's own ([`Source::Live`]). A list of a provider that names no
    /// `models_url` (an old store entry, a config that dropped the key)
    /// is left out.
    pub fn with_discovered(mut self, lists: &[(String, Value)]) -> Catalog {
        for (pid, raw) in lists {
            let Some(p) = self.provider(pid) else { continue };
            if p.models_url.is_empty() || !p.chats() {
                continue;
            }
            for Entry { id, caps } in parse(raw) {
                if self.model(&format!("{pid}/{id}")).is_none() {
                    self.models
                        .push(Model { provider: pid.clone(), id, api: None, stt: false, caps, source: Source::Live });
                }
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A config applied over the built-in list, read through
    /// [`Catalog::apply`] (private to the crate: the tests' own input).
    fn catalog(cfg: &str) -> Catalog {
        let mut c = Catalog::builtin();
        let t = cfg.parse::<toml::Table>().unwrap();
        c.apply(&t, Source::Config, "config.toml");
        c
    }

    /// opencode's zen Go shape, a subset of one real answer (beam-free).
    fn go_style(id: &str, vision: bool, price: f64) -> Value {
        json!({
            "id": id,
            "type": "chat",
            "modalities": {"input": if vision { vec!["text", "image/png"] } else { vec!["text"] }, "output": ["text"]},
            "capabilities": {
                "tools": {"supported": true},
                "reasoning": {"supported": true, "effort": ["low", "high", "max"]},
            },
            "limit": {"context": 1048576, "output": 131072},
            "cost": {"text": {"input": {"price": price, "cache": {"read": price}}, "output": {"price": price}}},
        })
    }

    fn caps(c: &Catalog, name: &str) -> (Source, PartialCaps) {
        let m = c.model(name).unwrap_or_else(|| panic!("{name}"));
        (m.source, m.caps.clone())
    }

    #[test]
    fn the_go_shape_becomes_model_entries() {
        let c = catalog("[providers.zen]\nbase_url = \"https://zen.test/v1\"\nmodels_url = \"https://zen.test/v1/models\"\n")
            .with_discovered(&[("zen".into(), json!({"data": [go_style("beam", true, 0.15)]}))]);
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        let (source, p) = caps(&c, "zen/beam");
        assert_eq!(source, Source::Live);
        assert_eq!(p.context, Some(1_048_576));
        assert_eq!(p.max_output, Some(131_072));
        assert_eq!(p.vision, Some(true));
        assert_eq!(p.reasoning, Some(true));
        assert_eq!(p.tools, Some(true));
        assert_eq!(p.efforts, Some("low,high,max".into()));
        assert_eq!(p.input_price, Some(150_000));
        assert_eq!(p.output_price, Some(150_000));
        assert_eq!(p.cache_read_price, Some(150_000));
    }

    #[test]
    fn a_plain_list_lists_its_ids() {
        let c = catalog("[providers.zen]\nbase_url = \"https://zen.test/v1\"\nmodels_url = \"https://zen.test/v1/models\"\n")
            .with_discovered(&[("zen".into(), json!({"data": [{"id": "q"}]}))]);
        let (source, p) = caps(&c, "zen/q");
        assert_eq!(source, Source::Live);
        assert_eq!(p, PartialCaps::default(), "its provider's caps apply");
    }

    #[test]
    fn a_garbage_or_non_chat_answer_lists_nothing() {
        for raw in [Value::Null, json!({"data": "one"}), json!({"data": [{"id": 12}]}), json!({})] {
            assert!(parse(&raw).is_empty());
        }
        // a transcription model, not for the chat lists
        let stt = json!({"data": [{"id": "whisper", "type": "stt"}]});
        assert!(parse(&stt).is_empty());
    }

    #[test]
    fn config_and_built_in_stay_in_charge_of_their_models() {
        let c = catalog("[providers.zen]\nmodels_url = \"https://zen.test/v1/models\"\n[models.\"zen/loud\"]\ncontext = 42\n")
            .with_discovered(&[("zen".into(), json!({"data": [go_style("loud", true, 1.0)]}))]);
        let (source, p) = caps(&c, "zen/loud");
        assert_eq!(source, Source::Config, "the config's entry stays");
        assert_eq!(p.context, Some(42));
    }

    #[test]
    fn an_answer_of_a_provider_without_models_url_is_refused() {
        let c = catalog("[providers.zen]\nbase_url = \"https://zen.test/v1\"\n")
            .with_discovered(&[("zen".into(), json!({"data": [go_style("beam", false, 0.0)]}))]);
        assert!(c.model("zen/beam").is_none());
    }

    #[test]
    fn the_store_roundtrips_and_says_what_is_due() {
        let s = Store::default().with("zen", 1000, json!({"data": []}));
        let t = s.to_json();
        let s = Store::parse(&t).unwrap();
        assert!(!s.due("zen", 1000 + TTL_MS - 1));
        assert!(s.due("zen", 1000 + TTL_MS));
        assert!(s.due("other", 0), "a provider with no fetch yet");
        assert_eq!(s.to_json(), t);
    }
}
