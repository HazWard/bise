//! The one writer of `<cache>/efforts.json` (effort-table step 2) and of
//! `<cache>/voice-models.json` (what Mistral transcribes now:
//! [`voice_live`], fetched the same way, `refresh_voice`): the
//! hub calls [`refresh`] in a thread at its start, then at most once a
//! day per provider. It fetches only the providers he uses (a key found,
//! or signed in): Anthropic and OpenRouter here, ChatGPT through
//! chatgpt.rs's own fetch of its model list (no second request). A
//! failure (no network, an error status, an answer that is not a list)
//! keeps the old file; a short timeout, so no network costs little.
//! [`read`] gives the lists to [`crate::Setup`] (the hub, the TUI, the
//! hand-off). The parsing is `live_efforts`, pure. [`export_handoff`]
//! writes the hand-off from config.toml and these cached lists.
use std::path::Path;
use std::time::Duration;

use bend_plugins::http::{self, Url};
use serde_json::Value;

use crate::auth::Keys;
use crate::discover;
use crate::live_efforts::{self, Store, FETCHED};
use crate::{auth_cli::Paths, chatgpt, voice_live, Provider, Setup};

/// The longest a fetch may wait, connect and each read.
pub const TIMEOUT: Duration = Duration::from_secs(5);

fn store_path(cache_dir: &Path) -> std::path::PathBuf {
    cache_dir.join("efforts.json")
}

/// The stored store, empty when there is none (or it is not one).
pub fn read_store(cache_dir: &Path) -> Store {
    std::fs::read_to_string(store_path(cache_dir)).ok().and_then(|t| Store::parse(&t)).unwrap_or_default()
}

/// Every API list bise holds: efforts.json, and ChatGPT's model list
/// (`chatgpt-models.json`, written by chatgpt.rs) as "chatgpt/<slug>".
pub fn read(cache_dir: &Path) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = read_store(cache_dir).models.into_iter().collect();
    let plan = std::fs::read_to_string(cache_dir.join("chatgpt-models.json")).ok();
    let plan: Option<Value> = plan.and_then(|t| serde_json::from_str(&t).ok());
    for m in plan.iter().flat_map(|v| v.get("models").and_then(Value::as_array).into_iter().flatten()) {
        let (Some(slug), l) = (m.get("slug").and_then(Value::as_str), live_efforts::chatgpt_levels(m)) else {
            continue;
        };
        if !l.is_empty() {
            out.push((format!("chatgpt/{slug}"), l));
        }
    }
    out
}

/// One provider's lists from its API, or why not.
fn fetch(p: &Provider, key: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    let v = get_json(&format!("{}/models", p.base_url.trim_end_matches('/')), p, key)?;
    let lists = if p.api == "anthropic" { live_efforts::parse_anthropic(&v) } else { live_efforts::parse_openrouter(&v) };
    if lists.is_empty() {
        return Err(format!("{}'s model list has no effort list", p.name));
    }
    Ok(lists)
}

/// A GET's answer, or why not. The URL is the caller's: the APIs' own
/// `{base_url}/models`, or a provider's `models_url`.
fn get_json(url: &str, p: &Provider, key: &str) -> Result<Value, String> {
    let url = Url::parse(url)?;
    let h: Vec<(String, String)> = match p.api.as_str() {
        "anthropic" => vec![("x-api-key".into(), key.into()), ("anthropic-version".into(), "2023-06-01".into())],
        _ => vec![("Authorization".into(), format!("Bearer {key}"))],
    };
    let r = http::send(&http::Request { method: "GET", url: &url, headers: &h, body: b"", timeout: TIMEOUT }).map_err(|e| e.to_string())?;
    let status = r.status;
    let body = r.read_all(32 << 20).map_err(|e| e.to_string())?;
    if !(200..300).contains(&status) {
        return Err(format!("{} answered {status} to the model list", p.name));
    }
    serde_json::from_slice(&body).map_err(|_| format!("{}'s model list is not JSON", p.name))
}

/// The voice lists that are due ([`voice_live`]): one line per provider
/// tried. A failure, or a list with no voice model, keeps the old one.
fn refresh_voice(cache_dir: &Path, setup: &Setup, keys: &Keys, now_ms: u64) -> Vec<String> {
    let mut out = Vec::new();
    let mut store = voice_live::read_store(cache_dir);
    let mut changed = false;
    for id in voice_live::LISTED {
        let Some(p) = setup.catalog.provider(id) else { continue };
        let Some(key) = keys.for_provider(p).map(|f| f.key) else { continue };
        if p.base_url.is_empty() || !store.due(id, now_ms) {
            continue;
        }
        match get_json(&format!("{}/models", p.base_url.trim_end_matches('/')), p, &key).map(|v| voice_live::parse(&v)) {
            Ok(ids) if !ids.is_empty() => {
                out.push(format!("voice: {} transcription models from {}", ids.len(), id));
                store = store.with(id, now_ms, ids);
                changed = true;
            }
            Ok(_) => out.push(format!("voice: {id} kept its old list: its model list has no transcription model")),
            Err(e) => out.push(format!("voice: {id} kept its old list: {e}")),
        }
    }
    if changed {
        if let Err(e) = write_file(&voice_live::store_path(cache_dir), &store.to_json()) {
            out.push(format!("voice: cannot write {}: {e}", voice_live::store_path(cache_dir).display()));
        }
    }
    out
}

/// Fetch the lists that are due (`now_ms`) and write them; one line per
/// provider tried, for the hub's log. Never a key in a line.
pub fn refresh(paths: &Paths, cache_dir: &Path, setup: &Setup, keys: &Keys, now_ms: u64) -> Vec<String> {
    let mut out = Vec::new();
    let mut store = read_store(cache_dir);
    let mut changed = false;
    for id in FETCHED {
        let Some(p) = setup.catalog.provider(id) else { continue };
        let Some(key) = keys.for_provider(p).map(|f| f.key) else { continue };
        if p.base_url.is_empty() || !store.due(id, now_ms) {
            continue;
        }
        match fetch(p, &key) {
            Ok(lists) => {
                out.push(format!("efforts: {} lists from {}", lists.len(), id));
                store = store.with(id, now_ms, lists);
                changed = true;
            }
            Err(e) => out.push(format!("efforts: {id} kept its old lists: {e}")),
        }
    }
    if changed {
        if let Err(e) = write(cache_dir, &store) {
            out.push(format!("efforts: cannot write {}: {e}", store_path(cache_dir).display()));
        }
    }
    out.extend(refresh_discovered(cache_dir, setup, keys, now_ms));
    // ChatGPT: its own model list carries the levels (chatgpt.rs writes it)
    if setup.catalog.provider("chatgpt").is_some_and(|p| keys.signed_in(p)) && !chatgpt::models_fresh(paths) {
        match chatgpt::fetch_models(paths) {
            Ok(m) => out.push(format!("efforts: {} models from chatgpt", m.len())),
            Err(e) => out.push(format!("efforts: chatgpt kept its old list: {e}")),
        }
    }
    out.extend(refresh_voice(cache_dir, setup, keys, now_ms));
    out
}

/// The key a GET may send: none (an auth-less gateway), or the
/// provider's (a key not found: None, not the empty string).
fn key_of(p: &Provider, keys: &Keys) -> Option<String> {
    if p.key_env.is_empty() {
        Some(String::new())
    } else {
        keys.for_provider(p).map(|f| f.key)
    }
}

/// The providers whose config names a model list (`models_url`): the
/// ones that are due ([`discover::TTL_MS`], a day), a chat provider with
/// a key that can go out (an auth-less gateway: no key needed, bise
/// sends the family's headers either way). One line each, for the hub's
/// log; a failure keeps the old store.
fn refresh_discovered(cache_dir: &Path, setup: &Setup, keys: &Keys, now_ms: u64) -> Vec<String> {
    let mut out = Vec::new();
    let mut raws = discover::Store::read(cache_dir);
    let mut changed = false;
    for p in setup
        .catalog
        .providers
        .iter()
        .filter(|p| !p.models_url.is_empty() && p.chats() && p.needs.is_empty())
    {
        if !raws.due(&p.id, now_ms) {
            continue;
        }
        let Some(key) = key_of(p, keys) else { continue };
        match get_json(&p.models_url, p, &key) {
            Ok(v) => {
                let n = discover::parse(&v).len();
                out.push(format!("models: {n} listed by {}", p.id));
                raws = raws.with(&p.id, now_ms, v);
                changed = true;
            }
            Err(e) => out.push(format!("models: {} kept its old list: {e}", p.id)),
        }
    }
    if changed {
        let path = discover::store_path(cache_dir);
        if let Err(e) = write_file(&path, &raws.to_json()) {
            out.push(format!("models: cannot write {}: {e}", path.display()));
        }
    }
    out
}

/// How often the hub's thread looks: [`refresh`] fetches only what is due.
pub const EVERY: Duration = Duration::from_secs(3600);

/// The hub's thread: [`refresh`] now, then every [`EVERY`], with the
/// keys and config as they are at each look; its lines go to `log`.
pub fn spawn_refresher(paths: Paths, cache_dir: std::path::PathBuf, log: fn(&str)) {
    let _ = std::thread::Builder::new().name("live-efforts".into()).spawn(move || loop {
        let files = crate::auth::EnvFile::read_all(&paths.env_files);
        let store = crate::auth::Store::read(&paths.auth_file).unwrap_or_default();
        let env = |k: &str| std::env::var(k).ok();
        let keys = Keys { env: &env, store: &store, files: &files };
        let setup = Setup::load(&paths.config);
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
        for line in refresh(&paths, &cache_dir, &setup, &keys, now) {
            log(&line);
        }
        std::thread::sleep(EVERY);
    });
}

/// What a [`Setup::load`] reads changed since: the latest change of
/// `config` and of the API lists (efforts.json, provider-models.json,
/// chatgpt-models.json, voice-models.json).
pub fn stamp(config: &Path, cache_dir: &Path) -> Option<std::time::SystemTime> {
    let at = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let lists = [
        store_path(cache_dir),
        discover::store_path(cache_dir),
        cache_dir.join("chatgpt-models.json"),
        voice_live::store_path(cache_dir),
    ];
    std::iter::once(at(config)).chain(lists.iter().map(|p| at(p))).flatten().max()
}

/// efforts.json, atomically.
fn write(cache_dir: &Path, store: &Store) -> std::io::Result<()> {
    write_file(&store_path(cache_dir), &store.to_json())
}

/// Atomically: a temp file of this write, then a rename.
fn write_file(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_extension(format!("json.{}.{}.tmp", std::process::id(), n));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Write the hand-off where the runtime will find it and return its
/// path: `<cache_dir>/models.toml`, else the temp dir. None when neither
/// is writable (the runtime then keeps its old table). `env` gives the
/// providers' `base_url_env` variables (the hub: its environment, then
/// the .env files read again, so an edit reaches the next call).
pub fn export_handoff(config: &Path, cache_dir: &Path, env: &dyn Fn(&str) -> Option<String>) -> Option<std::path::PathBuf> {
    let text = std::fs::read_to_string(config).ok();
    let setup = Setup::from_parts(text.as_deref(), &read(cache_dir), &discover::read(cache_dir), env, env)
        .with_plan_lists(crate::plan_list::read(cache_dir));
    let fallback = std::env::temp_dir().join(format!("bise-models-{}.toml", std::process::id()));
    let mut why = Vec::new();
    let found = [cache_dir.join("models.toml"), fallback]
        .into_iter()
        .find(|p| setup.write_handoff(p).map_err(|e| why.push(format!("{}: {e}", p.display()))).is_ok());
    if found.is_none() {
        // no file: the runtime has no effort list of its own (efforts.rs)
        eprintln!("warn: no models file for the runtime, its calls go without an effort ({})", why.join("; "));
    }
    found
}
