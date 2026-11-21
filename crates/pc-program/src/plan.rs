//! A shipped plan directory: `block_<b>[_<variant>]_placement.json` (version 2), `PLAN_CMD.txt` (graph / recipe /
//! runtime lines), `pcmm_calib.json` (prefill PCMM sites), and the run `Config` a plan binds: the `config` stamp
//! when the placement carries one, else the legacy `capture_env` + the `runtime=` line through
//! `Config::from_legacy_env`; the planner's recipe knobs through `PlanConfig::from_legacy_env`.
use crate::error::{json, read, Error, Result};
use pc_config::{Config, PlanConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub node_op: String,
    #[serde(default)]
    pub target_var: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Rules {
    #[serde(default)]
    pub bootstrap_level: Option<u32>,
    #[serde(default)]
    pub max_level: Option<u32>,
    #[serde(default)]
    pub source_level: Option<u32>,
    #[serde(default)]
    pub cache_read_level: Option<u32>,
    #[serde(default)]
    pub cache_pin_level: Option<i64>,
    #[serde(default)]
    pub sparse_bts_slots: Vec<u32>,
    /// The level past which a block input the plan does not map seeds with a pending rescale (plan.seed_level; unset:
    /// `bootstrap_level`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_level: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    #[serde(default)]
    pub final_named_levels: BTreeMap<String, u32>,
    #[serde(default)]
    pub final_named_degs: BTreeMap<String, u8>,
    #[serde(default)]
    pub exit_var: Option<String>,
    #[serde(default)]
    pub exit_level: Option<u32>,
    #[serde(default)]
    pub exit_deg: Option<u8>,
    #[serde(default)]
    pub total_bootstraps: Option<u32>,
    #[serde(default)]
    pub num_placements: Option<u32>,
    #[serde(default)]
    pub num_hint_bootstraps: Option<u32>,
}

/// placement.json version 2 (the fields the runtime binds; `summary.bts_quality` & co. are planner reports).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlacementFile {
    #[serde(default)]
    pub version: Option<u32>,
    #[serde(default)]
    pub rules: Rules,
    #[serde(default)]
    pub placements: Vec<Placement>,
    #[serde(default)]
    pub weight_levels: BTreeMap<String, u32>,
    #[serde(default)]
    pub mask_levels: BTreeMap<String, Vec<u32>>,
    /// Key presence (even empty) binds every hint decision to the plan.
    #[serde(default)]
    pub hint_fire: Option<Vec<String>>,
    /// Key presence (even empty) binds every landing-realize decision to the plan.
    #[serde(default)]
    pub rescale_after: Option<Vec<String>>,
    #[serde(default)]
    pub sparse_slots: BTreeMap<String, u32>,
    #[serde(default)]
    pub correction_factor: BTreeMap<String, u32>,
    #[serde(default)]
    pub raise_drop: BTreeMap<String, u32>,
    #[serde(default)]
    pub prescale: BTreeMap<String, f64>,
    #[serde(default)]
    pub offset: BTreeMap<String, f64>,
    #[serde(default)]
    pub real_route: Vec<String>,
    #[serde(default)]
    pub shipped_order: Vec<String>,
    #[serde(default)]
    pub summary: Summary,
    /// Config stamp: `Config::load_bearing()` JSON. Replaces `capture_env` when present.
    #[serde(default)]
    pub config: Option<Value>,
    /// Legacy contract: `{env: {NAME: value}, session: {...}}`.
    #[serde(default)]
    pub capture_env: Option<Value>,
}

/// The plan of one block, normalized into sets for the compiler.
#[derive(Clone, Debug, Default)]
pub struct BlockPlan {
    pub path: String,
    pub file: PlacementFile,
    pub placed: BTreeSet<String>,
    pub hint_fire: BTreeSet<String>,
    pub hints_bound: bool,
    pub rescale_after: BTreeSet<String>,
    pub rescale_bound: bool,
    pub real_route: BTreeSet<String>,
    pub shipped_order: BTreeSet<String>,
}

impl BlockPlan {
    pub fn load(path: impl AsRef<Path>) -> Result<BlockPlan> {
        let p = path.as_ref();
        let file: PlacementFile = json(p)?;
        BlockPlan::from_file(file, &p.display().to_string())
    }

    pub fn from_file(file: PlacementFile, path: &str) -> Result<BlockPlan> {
        if let Some(v) = file.version {
            if v != 2 {
                return Err(Error::Plan { path: path.into(), msg: format!("placement version {v} (known: 2)") });
            }
        }
        let mut placed = BTreeSet::new();
        for pl in &file.placements {
            if pl.kind != "bootstrap_after_node" {
                return Err(Error::Plan { path: path.into(), msg: format!("placement type {:?} (known: bootstrap_after_node)", pl.kind) });
            }
            if !pl.target_var.is_empty() {
                placed.insert(pl.target_var.clone());
            }
        }
        let set = |v: &Option<Vec<String>>| v.iter().flatten().cloned().collect::<BTreeSet<_>>();
        Ok(BlockPlan {
            path: path.into(),
            placed,
            hint_fire: set(&file.hint_fire),
            hints_bound: file.hint_fire.is_some(),
            rescale_after: set(&file.rescale_after),
            rescale_bound: file.rescale_after.is_some(),
            real_route: file.real_route.iter().cloned().collect(),
            shipped_order: file.shipped_order.iter().cloned().collect(),
            file,
        })
    }
}

/// `PLAN_CMD.txt`: `graph=<dir> (md5 ..)`, `recipe=K=V ...`, `runtime=K=V ...` (later tokens win, as in the shell).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanCmd {
    pub graph: Option<String>,
    pub recipe: BTreeMap<String, String>,
    pub runtime: BTreeMap<String, String>,
}

fn kv_line(s: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for tok in s.split_whitespace() {
        if let Some((k, v)) = tok.split_once('=') {
            m.insert(k.to_string(), v.to_string());
        }
    }
    m
}

impl PlanCmd {
    pub fn parse(text: &str) -> PlanCmd {
        let mut c = PlanCmd::default();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("graph=") {
                c.graph = rest.split_whitespace().next().map(str::to_string);
            } else if let Some(rest) = line.strip_prefix("recipe=") {
                c.recipe.extend(kv_line(rest));
            } else if let Some(rest) = line.strip_prefix("runtime=") {
                c.runtime.extend(kv_line(rest));
            }
        }
        c
    }
}

/// A plan directory with its blocks (and variants such as `block_0_feedback`).
#[derive(Clone, Debug)]
pub struct PlanDir {
    pub dir: PathBuf,
    pub cmd: PlanCmd,
    /// `pcmm_calib.json`: PCMM site (`b<block>.<site>`) -> calibrated magnitude.
    pub pcmm_calib: BTreeMap<String, f64>,
    /// (block index, variant, path), blocks ascending, the plain block before its variants.
    pub blocks: Vec<(u32, Option<String>, PathBuf)>,
}

impl PlanDir {
    pub fn open(dir: impl AsRef<Path>) -> Result<PlanDir> {
        let dir = dir.as_ref().to_path_buf();
        let cmd_path = dir.join("PLAN_CMD.txt");
        let cmd = if cmd_path.is_file() {
            PlanCmd::parse(&String::from_utf8_lossy(&read(&cmd_path)?))
        } else {
            PlanCmd::default()
        };
        let calib_path = dir.join("pcmm_calib.json");
        let pcmm_calib = if calib_path.is_file() { json(&calib_path)? } else { BTreeMap::new() };
        let rd = std::fs::read_dir(&dir).map_err(|err| Error::Io { path: dir.display().to_string(), err })?;
        let mut blocks = vec![];
        for e in rd {
            let e = e.map_err(|err| Error::Io { path: dir.display().to_string(), err })?;
            let name = e.file_name().to_string_lossy().to_string();
            let Some(mid) = name.strip_prefix("block_").and_then(|r| r.strip_suffix("_placement.json")) else { continue };
            let (b, var) = match mid.split_once('_') {
                Some((b, v)) => (b, Some(v.to_string())),
                None => (mid, None),
            };
            let Ok(b) = b.parse::<u32>() else { continue };
            blocks.push((b, var, e.path()));
        }
        blocks.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        Ok(PlanDir { dir, cmd, pcmm_calib, blocks })
    }

    /// The `PlanConfig` of the recipe line (empty when the plan has none).
    pub fn plan_config(&self) -> Result<PlanConfig> {
        let (pc, _rest) = PlanConfig::from_legacy_env(self.cmd.recipe.iter().map(|(k, v)| (k.as_str(), Some(v.as_str()))))?;
        Ok(pc)
    }

    /// The run Config a block plan binds: its `config` stamp overlaid by `runtime=` (a stamp carries the capture's
    /// load-bearing fields, not the runtime levers PLAN_CMD.txt declares), else `capture_env.env` overlaid by
    /// `runtime=`.
    pub fn config(&self, plan: &BlockPlan) -> Result<Config> {
        if let Some(stamp) = &plan.file.config {
            let cfg: Config = serde_json::from_value(stamp.clone()).map_err(pc_config::Error::from)?;
            if self.cmd.runtime.is_empty() {
                return Ok(cfg);
            }
            let (rt, _rest) = Config::from_legacy_env(self.cmd.runtime.iter().map(|(k, v)| (k.as_str(), Some(v.as_str()))))?;
            return Ok(cfg.overlay(&rt));
        }
        let mut env: BTreeMap<String, Option<String>> = BTreeMap::new();
        if let Some(ce) = plan.file.capture_env.as_ref().and_then(|c| c.get("env")).and_then(Value::as_object) {
            for (k, v) in ce {
                let v = match v {
                    Value::Null => None,
                    Value::String(s) => Some(s.clone()),
                    other => Some(other.to_string()),
                };
                env.insert(k.clone(), v);
            }
        }
        for (k, v) in &self.cmd.runtime {
            env.insert(k.clone(), Some(v.clone()));
        }
        let (cfg, _rest) = Config::from_legacy_env(env.iter().map(|(k, v)| (k.as_str(), v.as_deref())))?;
        Ok(cfg)
    }

    /// The graph dir PLAN_CMD.txt names, resolved against `assets` (the dir holding `graphs/`).
    pub fn graph_dir(&self, assets: &Path) -> Option<PathBuf> {
        self.cmd.graph.as_ref().map(|g| assets.join(g))
    }
}
