//! CLAP model cez ONNX Runtime – rovnaké grafy ako Python verzia.
//! DirectML (Windows GPU) sa skúsi ako poskytovateľ, ak je zapnutá
//! vlastnosť `ort-directml`; inak CPU. Texty tokenizuje oficiálny HF
//! tokenizér (tokenizer.json) – identicky ako Python.

use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

pub const TEXT_TOKENS: usize = 32;
pub const TEXT_BATCH: usize = 32;

fn new_session(path: &std::path::Path) -> Result<Session> {
    // DirectML (GPU) s pádom späť na CPU – rovnako ako Python verzia
    #[cfg(feature = "ort-directml")]
    {
        let b = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
            ])?;
        match b.commit_from_file(path) {
            Ok(s) => return Ok(s),
            Err(e) => eprintln!("[CLAP] DirectML nedostupné ({e}) → pokračujem na CPU"),
        }
    }
    let b = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;
    Ok(b.commit_from_file(path)?)
}

pub struct ClapModel {
    pub audio: Session,
    pub text: Session,
    pub audio_out: String,
    pub text_out: String,
    pub tokenizer: tokenizers::Tokenizer,
    pub logit_scale: f32,
    pub backend_info: String,
}

impl ClapModel {
    /// `model_dir` obsahuje clap_audio.onnx / clap_text.onnx / export_meta.json
    /// (a prípadne tokenizer.json; inak hľadá v HF cache).
    pub fn load(model_dir: &std::path::Path) -> Result<Self> {
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("export_meta.json"))
                .context("export_meta.json – najprv raz spustite Python verziu (export modelu)")?,
        )
        .context("export_meta.json nie je platný JSON")?;
        let logit_scale = meta
            .get("logit_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(100.0) as f32;
        let tokenizer_path = Self::find_tokenizer(model_dir)?;

        let audio = new_session(&model_dir.join("clap_audio.onnx"))?;
        let text = new_session(&model_dir.join("clap_text.onnx"))?;
        let audio_out = audio.outputs[0].name.clone();
        let text_out = text.outputs[0].name.clone();

        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("tokenizér {}: {e}", tokenizer_path.display()))?;
        let (pad_id, pad_token) = tokenizer
            .get_padding()
            .map(|p| (p.pad_id, p.pad_token.clone()))
            .unwrap_or((0, "<pad>".to_string()));
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(TEXT_TOKENS),
            pad_id,
            pad_token,
            pad_type_id: 0,
            pad_to_multiple_of: None,
            direction: tokenizers::PaddingDirection::Right,
        }));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: TEXT_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("truncation: {e}"))?;

        Ok(Self {
            audio,
            text,
            audio_out,
            text_out,
            tokenizer,
            logit_scale,
            backend_info: "onnxruntime (Rust)".into(),
        })
    }

    fn find_tokenizer(model_dir: &std::path::Path) -> Result<std::path::PathBuf> {
        let local = model_dir.join("tokenizer.json");
        if local.exists() {
            return Ok(local);
        }
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".into());
        let snaps = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--laion--clap-htsat-unfused/snapshots");
        if let Ok(entries) = std::fs::read_dir(&snaps) {
            for snap in entries.flatten() {
                let cand = snap.path().join("tokenizer.json");
                if cand.exists() {
                    return Ok(cand);
                }
            }
        }
        anyhow::bail!(
            "tokenizer.json sa nenašiel v {} ani v HF cache – skopírujte ho do priečinka modelu",
            model_dir.display()
        )
    }

    /// Popisy → (N, 512) normalizované embeddingy (dávky po 32 ako Python).
    pub fn embed_texts(&mut self, descs: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(descs.len());
        for chunk in descs.chunks(TEXT_BATCH) {
            let enc = self
                .tokenizer
                .encode_batch(chunk.to_vec(), true)
                .map_err(|e| anyhow::anyhow!("tokenizácia: {e}"))?;
            let n = enc.len();
            let mut ids = vec![0i64; n * TEXT_TOKENS];
            let mut mask = vec![0i64; n * TEXT_TOKENS];
            for (i, e) in enc.iter().enumerate() {
                let toks = e.get_ids();
                let attn = e.get_attention_mask();
                for j in 0..toks.len().min(TEXT_TOKENS) {
                    ids[i * TEXT_TOKENS + j] = toks[j] as i64;
                    mask[i * TEXT_TOKENS + j] = attn[j] as i64;
                }
            }
            let ids_t = Tensor::from_array((vec![n as i64, TEXT_TOKENS as i64], ids))?;
            let mask_t = Tensor::from_array((vec![n as i64, TEXT_TOKENS as i64], mask))?;
            let outputs = self.text.run(ort::inputs![
                "input_ids" => ids_t,
                "attention_mask" => mask_t
            ])?;
            let (shape, data) = outputs[self.text_out.as_str()].try_extract_tensor::<f32>()?;
            let d = *shape.last().unwrap_or(&512) as usize;
            for row in data.chunks(d.max(1)) {
                out.push(l2norm(row));
            }
        }
        Ok(out)
    }

    /// Log-mel črty (N × 1001 × 64) → (N, 512) normalizované embeddingy.
    pub fn embed_audio(&mut self, feats: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = feats.len();
        // Rýchla cesta: všetky okná v jednom batchi (graf s dynamickým batch).
        if n > 1 && !self.audio_fixed_batch() {
            if let Ok(out) = self.embed_audio_batch(feats) {
                return Ok(out);
            }
            // starší graf môže vyžadovať batch=1 → skúsime po riadkoch
            eprintln!("[CLAP] graf odmietol batch {n} → okná sa pošlú osobitne");
        }
        // Pomalá cesta: jedno okno na volanie (graf s fixným batch = 1).
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut rows = self.embed_audio_batch(&feats[i..i + 1])?;
            out.push(rows.remove(0));
        }
        Ok(out)
    }

    /// Má audio graf fixne danú 0. dimenziu (batch) = 1?
    fn audio_fixed_batch(&self) -> bool {
        self.audio
            .inputs
            .first()
            .map(|i| match &i.input_type {
                ort::value::ValueType::Tensor { shape, .. } => shape.first() == Some(&1),
                _ => false,
            })
            .unwrap_or(false)
    }

    fn embed_audio_batch(&mut self, feats: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = feats.len();
        let flat: Vec<f32> = feats.concat();
        let t = Tensor::from_array((vec![n as i64, 1, 1001, 64], flat))?;
        let outputs = self.audio.run(ort::inputs!["input_features" => t])?;
        let (shape, data) = outputs[self.audio_out.as_str()].try_extract_tensor::<f32>()?;
        let d = (*shape.last().unwrap_or(&512)).max(1) as usize;
        let mut out = Vec::with_capacity(n);
        for row in data.chunks(d) {
            out.push(l2norm(row));
        }
        Ok(out)
    }
}

pub fn l2norm(v: &[f32]) -> Vec<f32> {
    let n = v
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    if n < 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|x| *x / n as f32).collect()
}

/// softmax presne ako Python (_softmax, 1-D).
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}
