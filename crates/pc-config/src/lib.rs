//! pc-config: the typed run configuration (one field per environment variable of the perseus C++/Python stack).
//!
//! `Config { chain, bts, mem, host, model, quirks, exec }`; the planner's recipe
//! knobs live in the separate `PlanConfig`. Every field is `Option<T>`: `None` = unset, which is what the legacy
//! readers' per-site defaults acted on (several sites disagree; the default each documents is the inventory's).
//! Field list, names and types come from `docs/ENV_INVENTORY.md` (generated into `fields.rs`); each field carries
//! its legacy env name, so `from_legacy_env` / `to_legacy_env` bridge the plan contract (`capture_env`, the
//! `runtime=` line of PLAN_CMD.txt). Presets are in `presets`.
pub mod error;
pub mod fields;
pub mod kind;
pub mod presets;

pub use error::{Error, Result};
pub use fields::*;
pub use kind::Kind;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One field's metadata: where it lives, its legacy env name ("" = none), kind, the legacy default, documentation.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct FieldInfo {
    pub path: &'static str,
    pub name: &'static str,
    pub env: &'static str,
    pub kind: Kind,
    pub default: &'static str,
    pub doc: &'static str,
}

impl FieldInfo {
    pub fn dotted(&self) -> String {
        format!("{}.{}", self.path, self.name)
    }
    /// The contract compared this name verbatim (`plan/contract.py` LOAD_BEARING_ENV).
    pub fn in_contract(&self) -> bool {
        self.doc.contains("[contract]")
    }
    /// A perseus-core kernel switch: it changes neither the op stream, its levels nor the keys (`load_bearing` leaves
    /// it out).
    pub fn kernel_switch(&self) -> bool {
        self.doc.contains("[kernel]")
    }
}

/// A section struct per group: fields `Option<kind>`, the static field table, env and JSON accessors.
#[macro_export]
macro_rules! section {
    ( $(#[$m:meta])* $S:ident $path:literal {
        $( $(#[doc = $d:literal])* $f:ident : $K:ident = $env:literal, $def:literal; )*
    } $( subs { $( $sf:ident : $ST:ident; )* } )? ) => {
        $(#[$m])*
        #[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct $S {
            $( $(#[doc = $d])* #[serde(default, skip_serializing_if = "Option::is_none")] pub $f: Option<$crate::kind::$K>, )*
            $($( #[serde(default, skip_serializing_if = "crate::is_default")] pub $sf: $ST, )*)?
        }
        impl $S {
            pub const PATH: &'static str = $path;
            pub const FIELDS: &'static [$crate::FieldInfo] = &[ $( $crate::FieldInfo {
                path: $path, name: stringify!($f), env: $env, kind: $crate::Kind::$K, default: $def,
                doc: concat!($($d, " "),*),
            }, )* ];
            pub fn all_fields(out: &mut Vec<&'static $crate::FieldInfo>) {
                out.extend(Self::FIELDS.iter());
                $($( $ST::all_fields(out); )*)?
            }
            /// `Some(value-or-unset)` if this section (or a sub-section) owns the env name.
            pub fn get_env(&self, env: &str) -> Option<Option<String>> {
                $( if !$env.is_empty() && env == $env {
                    return Some(self.$f.as_ref().map($crate::kind::Value::to_env));
                } )*
                $($( if let Some(v) = self.$sf.get_env(env) { return Some(v); } )*)?
                None
            }
            /// Set from a legacy env value (None = unset); `Ok(false)` if no field owns the name.
            pub fn set_env(&mut self, env: &str, v: Option<&str>) -> $crate::Result<bool> {
                $( if !$env.is_empty() && env == $env {
                    self.$f = match v { Some(s) => Some($crate::kind::Value::from_env($env, s)?), None => None };
                    return Ok(true);
                } )*
                $($( if self.$sf.set_env(env, v)? { return Ok(true); } )*)?
                Ok(false)
            }
            pub fn to_legacy_env_into(&self, out: &mut std::collections::BTreeMap<String, String>) {
                $( if !$env.is_empty() { if let Some(v) = &self.$f {
                    out.insert($env.to_string(), $crate::kind::Value::to_env(v));
                } } )*
                $($( self.$sf.to_legacy_env_into(out); )*)?
            }
            /// A field by its name relative to this section (`ln_cheb`, `prefill.attn`), as JSON (null = unset).
            pub fn get_json(&self, name: &str) -> Option<serde_json::Value> {
                $( if name == stringify!($f) {
                    return Some(serde_json::to_value(&self.$f).unwrap_or(serde_json::Value::Null));
                } )*
                $($( if let Some(rest) = name.strip_prefix(concat!(stringify!($sf), ".")) {
                    return self.$sf.get_json(rest);
                } )*)?
                None
            }
            pub fn set_json(&mut self, name: &str, v: serde_json::Value) -> $crate::Result<bool> {
                $( if name == stringify!($f) {
                    self.$f = serde_json::from_value(v)?;
                    return Ok(true);
                } )*
                $($( if let Some(rest) = name.strip_prefix(concat!(stringify!($sf), ".")) {
                    return self.$sf.set_json(rest, v);
                } )*)?
                Ok(false)
            }
        }
    };
}

pub(crate) fn is_default<T: Default + PartialEq>(t: &T) -> bool {
    *t == T::default()
}

/// The run configuration. JSON: `{"chain": {...}, "bts": {...}, ...}` with unset fields omitted.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default, skip_serializing_if = "is_default")]
    pub chain: ChainSection,
    #[serde(default, skip_serializing_if = "is_default")]
    pub bts: BtsSection,
    #[serde(default, skip_serializing_if = "is_default")]
    pub mem: MemSection,
    #[serde(default, skip_serializing_if = "is_default")]
    pub host: HostSection,
    #[serde(default, skip_serializing_if = "is_default")]
    pub model: ModelSection,
    #[serde(default, skip_serializing_if = "is_default")]
    pub quirks: QuirksSection,
    #[serde(default, skip_serializing_if = "is_default")]
    pub exec: ExecSection,
}

/// Names a `from_legacy_env` call did not place in the Config (planner recipe knobs, process env, typos).
pub type Unplaced = Vec<String>;

macro_rules! each_section {
    ($self:ident, $s:ident => $body:expr) => {{
        { let $s = &$self.chain; $body }
        { let $s = &$self.bts; $body }
        { let $s = &$self.mem; $body }
        { let $s = &$self.host; $body }
        { let $s = &$self.model; $body }
        { let $s = &$self.quirks; $body }
        { let $s = &$self.exec; $body }
    }};
}

impl Config {
    /// Every field (sections in order, sub-sections after their parent's own fields).
    pub fn fields() -> Vec<&'static FieldInfo> {
        let mut v = vec![];
        ChainSection::all_fields(&mut v);
        BtsSection::all_fields(&mut v);
        MemSection::all_fields(&mut v);
        HostSection::all_fields(&mut v);
        ModelSection::all_fields(&mut v);
        QuirksSection::all_fields(&mut v);
        ExecSection::all_fields(&mut v);
        v
    }

    pub fn from_json(s: &str) -> Result<Config> {
        Ok(serde_json::from_str(s)?)
    }
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Build from an env map. Values that a Config field owns are parsed strictly (a malformed value is an error
    /// naming the knob); names no field owns are returned (the planner's recipe knobs go to `PlanConfig`).
    pub fn from_legacy_env<'a, I>(env: I) -> Result<(Config, Unplaced)>
    where
        I: IntoIterator<Item = (&'a str, Option<&'a str>)>,
    {
        let mut c = Config::default();
        let mut rest = vec![];
        for (k, v) in env {
            if !c.set_env(k, v)? {
                rest.push(k.to_string());
            }
        }
        Ok((c, rest))
    }

    /// Set one legacy env name; `Ok(false)` if the Config has no field for it.
    pub fn set_env(&mut self, env: &str, v: Option<&str>) -> Result<bool> {
        Ok(self.chain.set_env(env, v)?
            || self.bts.set_env(env, v)?
            || self.mem.set_env(env, v)?
            || self.host.set_env(env, v)?
            || self.model.set_env(env, v)?
            || self.quirks.set_env(env, v)?
            || self.exec.set_env(env, v)?)
    }

    /// `Some(value-or-unset)` if a field owns the env name.
    pub fn get_env(&self, env: &str) -> Option<Option<String>> {
        each_section!(self, s => if let Some(v) = s.get_env(env) { return Some(v); });
        None
    }

    /// Every set field as its legacy env name and spelling.
    pub fn to_legacy_env(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        each_section!(self, s => s.to_legacy_env_into(&mut out));
        out
    }

    /// A field by dotted name (`model.ln_cheb`, `model.prefill.attn`) as JSON, null when unset.
    pub fn get(&self, dotted: &str) -> Result<serde_json::Value> {
        let (sec, rest) = dotted.split_once('.').ok_or_else(|| Error::NoField(dotted.to_string()))?;
        let v = match sec {
            "chain" => self.chain.get_json(rest),
            "bts" => self.bts.get_json(rest),
            "mem" => self.mem.get_json(rest),
            "host" => self.host.get_json(rest),
            "model" => self.model.get_json(rest),
            "quirks" => self.quirks.get_json(rest),
            "exec" => self.exec.get_json(rest),
            _ => None,
        };
        v.ok_or_else(|| Error::NoField(dotted.to_string()))
    }

    pub fn set(&mut self, dotted: &str, v: serde_json::Value) -> Result<()> {
        let (sec, rest) = dotted.split_once('.').ok_or_else(|| Error::NoField(dotted.to_string()))?;
        let ok = match sec {
            "chain" => self.chain.set_json(rest, v)?,
            "bts" => self.bts.set_json(rest, v)?,
            "mem" => self.mem.set_json(rest, v)?,
            "host" => self.host.set_json(rest, v)?,
            "model" => self.model.set_json(rest, v)?,
            "quirks" => self.quirks.set_json(rest, v)?,
            "exec" => self.exec.set_json(rest, v)?,
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err(Error::NoField(dotted.to_string()))
        }
    }

    /// The subset a plan is bound to (chain / bts / model / quirks) plus every field the legacy contract
    /// compared (`[contract]` readers, e.g. `exec.gpt2_cache`, `mem.fhe_pt_coeff_encode`), without the kernel switches
    /// (`[kernel]`): they bind no plan and no key bundle.
    pub fn load_bearing(&self) -> Config {
        let mut c = Config {
            chain: self.chain.clone(),
            bts: self.bts.clone(),
            model: self.model.clone(),
            quirks: self.quirks.clone(),
            ..Config::default()
        };
        for f in Config::fields() {
            if f.kernel_switch() {
                // a field of the copied sections; null unsets it
                let _ = c.set(&f.dotted(), serde_json::Value::Null);
            }
            if f.in_contract() && matches!(f.path.split('.').next(), Some("mem" | "host" | "exec")) {
                if let Ok(v) = self.get(&f.dotted()) {
                    // the field exists by construction; set cannot fail on its own value
                    let _ = c.set(&f.dotted(), v);
                }
            }
        }
        c
    }

    /// `self` with every field `over` sets replaced (presets are layered this way).
    pub fn overlay(&self, over: &Config) -> Config {
        let mut c = self.clone();
        for (k, v) in over.to_legacy_env() {
            // values come from a valid Config, so they parse back
            let _ = c.set_env(&k, Some(&v));
        }
        for f in Config::fields().into_iter().filter(|f| f.env.is_empty()) {
            if let Ok(v) = over.get(&f.dotted()) {
                if !v.is_null() {
                    let _ = c.set(&f.dotted(), v);
                }
            }
        }
        c
    }
}

impl PlanConfig {
    pub fn fields() -> Vec<&'static FieldInfo> {
        let mut v = vec![];
        PlanConfig::all_fields(&mut v);
        v
    }
    pub fn from_legacy_env<'a, I>(env: I) -> Result<(PlanConfig, Unplaced)>
    where
        I: IntoIterator<Item = (&'a str, Option<&'a str>)>,
    {
        let mut c = PlanConfig::default();
        let mut rest = vec![];
        for (k, v) in env {
            if !c.set_env(k, v)? {
                rest.push(k.to_string());
            }
        }
        Ok((c, rest))
    }
    pub fn to_legacy_env(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        self.to_legacy_env_into(&mut out);
        out
    }
}
