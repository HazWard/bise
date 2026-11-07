//! The models a picker offers for a provider (`which model?`, the first
//! run): its recommended model for the role, then the plan's own list
//! (cached at sign-in), then the catalog's, but never a model the plan's
//! list lacks (bise_catalog::plan_list, chatgpt-default).

use super::*;
use bise_catalog::roles;

/// The catalog and the model choice (`Setup`: `BISE_MODEL` >
/// `BEND_MODEL` > `model` in config.toml > the default).
/// The base URLs' variables (foundry's ANTHROPIC_FOUNDRY_BASE_URL) are
/// also read from the .env files, like its key (~/.vibe/.env). The voice
/// models the providers list now (voice_live) over the catalog's.
pub(crate) fn setup_of(env: Env, home: &bise_home::Home) -> bise_catalog::Setup {
    let text = std::fs::read_to_string(home.config_file()).ok();
    let files = bise_catalog::auth::EnvFile::read_all(&home.env_files());
    let store = bise_catalog::auth::Store::read(&home.auth_file()).unwrap_or_default();
    // one key, every role: the checker and voice follow the keys found
    bise_catalog::Setup::from_parts(text.as_deref(), &[], &[], env, &|k| bise_catalog::with_files(env, &files, k))
        .with_live_stt(&bise_catalog::voice_live::read(&home.cache_dir()))
        .with_keys(&bise_catalog::auth::Keys { env, store: &store, files: &files })
        // a plan's own list first (plan_list): its pick, its swaps
        .with_plan_lists(bise_catalog::plan_list::read(&home.cache_dir()))
}

impl Onb {
    /// The models offered for `p`: its pick first, then the catalog's chat
    /// models of that provider.
    pub(crate) fn models_of(&self, p: &Provider) -> Vec<String> {
        // the voice picker (BISE-298): its voice models, its pick first
        if self.voice_pick() {
            return self.voice_models(p);
        }
        // a role's steps (BISE-301): the one recommended for the role first
        let first = self.picking().and_then(|id| self.recommended(id, p)).or_else(|| bise_catalog::roles::pick_for(&self.setup.catalog, &p.id, &self.model));
        let mut v: Vec<String> = first.into_iter().collect();
        // the plan: the account's own list (cached at sign-in and when
        // /models opens), then the catalog's
        if p.plan {
            for m in (self.logins.models)(&bise_catalog::auth_cli::Paths::of(&self.home)) {
                let full = format!("{}/{}", p.id, m.slug);
                if !v.contains(&full) {
                    v.push(full);
                }
            }
        }
        // the catalog's, but a model the plan's list lacks (plan_list)
        let listed = self.setup.catalog.listed_of(&p.id);
        for m in self.setup.catalog.models.iter().filter(|m| m.provider == p.id && !m.stt) {
            if !listed.is_empty() && !listed.contains(&m.id) {
                continue;
            }
            let full = format!("{}/{}", m.provider, m.id);
            if !v.contains(&full) {
                v.push(full);
            }
        }
        v
    }

    /// The model `which model?` recommends for a role on `p`: its voice
    /// pick, its small model (small jobs), else its pick.
    pub(crate) fn recommended(&self, id: &str, p: &Provider) -> Option<String> {
        let c = self.setup.catalog.provider(&p.id)?;
        let m = match id {
            roles::VOICE => &c.voice_model,
            // the checker: a small fast model is enough (design §4.2)
            roles::SMALL | roles::CLASSIFY => &c.small_model,
            _ => &c.model,
        };
        // a chat model the plan's list lacks: the plan's pick (plan_list)
        let m = (!m.is_empty()).then(|| format!("{}/{}", p.id, m))?;
        Some(if id == roles::VOICE { m } else { bise_catalog::plan_list::swap_of(&self.setup.catalog, &m).unwrap_or(m) })
    }
}
