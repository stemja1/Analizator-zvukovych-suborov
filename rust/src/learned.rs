//! Naučené dáta – KOMPATIBILNÉ s Python verziou:
//!  - naucene_spojenia.json (slová z názvov → popisy)
//!  - naucene_vzory.npz     (zvukové vzory: embeddingy + popisy)
//! npz čítame/zapisujeme ručne (zip + npy formát) kvôli <U dtype labelov.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::{Read, Write};

pub const PATTERN_MAX_PER_LABEL: usize = 30;
const F32_SIZE: usize = 4;

// ---------- npy helpery ------------------------------------------------------

pub fn write_npy_flat(path: &std::path::Path, shape: &[usize], data: &[f32]) -> anyhow::Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(path)?;
    let mut w = zip::ZipWriter::new(f);
    let opts: zip::write::SimpleFileOptions = Default::default();
    w.start_file("data.npy", opts)?;
    w.write_all(&npy_header("<f4", shape, 4))?;
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    w.finish()?;
    Ok(())
}

fn npy_header(descr: &str, shape: &[usize], itemsize: usize) -> Vec<u8> {
    let shape_str = shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape_str)
    } else {
        format!("({})", shape_str)
    };
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );
    let mut head = Vec::new();
    head.extend_from_slice(b"\x93NUMPY");
    head.extend_from_slice(&[1u8, 0]); // verzia 1.0
    let unpadded_len = 10 + dict.len() + 1;
    let total_len = (unpadded_len + 63) / 64 * 64; // zarovnanie na 64 bajtov
    head.extend_from_slice(&((total_len - 10) as u16).to_le_bytes());
    head.extend_from_slice(dict.as_bytes());
    while head.len() < total_len {
        head.push(b' ');
    }
    head[total_len - 1] = b'\n';
    debug_assert_eq!(head.len(), total_len);
    debug_assert_eq!(itemsize, itemsize); // itemsize sa počíta volajúcim
    head
}

fn parse_npy_header(r: &mut impl Read) -> Result<(String, Vec<usize>, usize)> {
    let mut magic = [0u8; 6];
    r.read_exact(&mut magic)?;
    if magic != *b"\x93NUMPY" {
        anyhow::bail!("nie je npy súbor");
    }
    let mut ver = [0u8; 2];
    r.read_exact(&mut ver)?;
    let hlen = if ver[0] == 1 {
        let mut b = [0u8; 2];
        r.read_exact(&mut b)?;
        u16::from_le_bytes(b) as usize
    } else {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        u32::from_le_bytes(b) as usize
    };
    let mut hdr = vec![0u8; hlen];
    r.read_exact(&mut hdr)?;
    let s = String::from_utf8_lossy(&hdr);
    // {'descr': '<f4', 'fortran_order': False, 'shape': (30, 512), }
    let descr = s
        .split("'descr': '")
        .nth(1)
        .and_then(|x| x.split('\'').next())
        .unwrap_or("<f4")
        .to_string();
    let shape_part = s
        .split("'shape': (")
        .nth(1)
        .and_then(|x| x.split(')').next())
        .unwrap_or("");
    let shape: Vec<usize> = shape_part
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect();
    let itemsize: usize = if let Some(rest) = descr.strip_prefix("<U") {
        rest.parse().unwrap_or(4) * 4
    } else if descr == "<f4" {
        4
    } else {
        4
    };
    Ok((descr, shape, itemsize))
}

fn read_npy_f32(r: &mut impl Read) -> Result<(Vec<usize>, Vec<f32>)> {
    let (descr, shape, _is) = parse_npy_header(r)?;
    if descr != "<f4" {
        anyhow::bail!("očakávané <f4, dostal {descr}");
    }
    let n: usize = shape.iter().product();
    let mut bytes = vec![0u8; n * F32_SIZE];
    r.read_exact(&mut bytes)?;
    let data = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok((shape, data))
}

fn read_npy_str(r: &mut impl Read) -> Result<Vec<String>> {
    let (descr, shape, itemsize) = parse_npy_header(r)?;
    if !descr.starts_with("<U") {
        anyhow::bail!("očakávané <U…, dostal {descr}");
    }
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut item = vec![0u8; itemsize];
    for _ in 0..n {
        r.read_exact(&mut item)?;
        // numpy <U formát: UTF-32LE znaky doplnené nulami (BEZ dĺžkovej predpony)
        let mut s = String::new();
        for i in 0..itemsize / 4 {
            let b = &item[i * 4..i * 4 + 4];
            let cp = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if cp == 0 {
                break; // výplň nulami = koniec reťazca
            }
            if let Some(c) = char::from_u32(cp) {
                s.push(c);
            }
        }
        out.push(s);
    }
    Ok(out)
}

// ---------- zvukové vzory (npz) ----------------------------------------------

pub struct Patterns {
    pub emb: Vec<f32>,   // (N, D) sploštené
    pub label: Vec<String>,
    pub dim: usize,
    pub path: std::path::PathBuf,
}

impl Patterns {
    pub fn load(path: &std::path::Path) -> Self {
        let mut p = Patterns { emb: Vec::new(), label: Vec::new(), dim: 0, path: path.to_path_buf() };
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return p,
        };
        let mut z = match zip::ZipArchive::new(std::io::BufReader::new(file)) {
            Ok(z) => z,
            Err(_) => return p,
        };
        let mut read_member = |name: &str| -> Option<Vec<u8>> {
            let mut f = z.by_name(name).ok()?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).ok()?;
            Some(buf)
        };
        if let Some(bytes) = read_member("emb.npy") {
            let mut cur = &bytes[..];
            if let Ok((shape, data)) = read_npy_f32(&mut cur) {
                if shape.len() == 2 && shape[0] > 0 {
                    p.dim = shape[1];
                    p.emb = data;
                }
            }
        }
        if let Some(bytes) = read_member("label.npy") {
            let mut cur = &bytes[..];
            if let Ok(labels) = read_npy_str(&mut cur) {
                p.label = labels;
            }
        }
        if p.label.len() * (if p.dim == 0 { 1 } else { p.dim }) != p.emb.len() {
            p.emb.clear();
            p.label.clear();
            p.dim = 0;
        }
        p
    }

    pub fn add(&mut self, v: &[f32], label: &str) {
        if v.is_empty() {
            return;
        }
        if self.dim == 0 || self.emb.len() != self.label.len() * self.dim {
            self.dim = v.len();
            self.emb = v.to_vec();
            self.label = vec![label.to_string()];
            return;
        }
        if v.len() != self.dim {
            return; // iná dimenzia (iný model) – ignorujeme
        }
        let same: Vec<usize> = self
            .label
            .iter()
            .enumerate()
            .filter(|(_, l)| l.as_str() == label)
            .map(|(i, _)| i)
            .collect();
        if same.len() >= PATTERN_MAX_PER_LABEL {
            let drop_i = same[0];
            self.label.remove(drop_i);
            let start = drop_i * self.dim;
            self.emb.drain(start..start + self.dim);
        }
        self.emb.extend_from_slice(v);
        self.label.push(label.to_string());
    }

    pub fn save(&self) -> Result<()> {
        if self.emb.is_empty() || self.dim == 0 {
            return Ok(());
        }
        let tmp = self.path.with_extension("npz.tmp");
        {
            let file = std::fs::File::create(&tmp)?;
            let mut w = zip::ZipWriter::new(std::io::BufWriter::new(file));
            let opts: zip::write::SimpleFileOptions = Default::default();
            w.start_file("emb.npy", opts)?;
            w.write_all(&npy_header("<f4", &[self.label.len(), self.dim], 4))?;
            for f in &self.emb {
                w.write_all(&f.to_le_bytes())?;
            }
            // numpy <U na disku: UTF-32LE znaky doplnené nulami, šírka = max dĺžka
            let max_len = self.label.iter().map(|l| l.chars().count()).max().unwrap_or(1).max(1);
            w.start_file("label.npy", opts)?;
            w.write_all(&npy_header(
                &format!("<U{}", max_len),
                &[self.label.len()],
                max_len * 4,
            ))?;
            for l in &self.label {
                let mut item = vec![0u8; max_len * 4];
                for (i, ch) in l.chars().take(max_len).enumerate() {
                    let b = (ch as u32).to_le_bytes();
                    item[i * 4..i * 4 + 4].copy_from_slice(&b);
                }
                w.write_all(&item)?;
            }
            w.finish()?;
        }
        std::fs::rename(&tmp, &self.path).context("premenovanie npz")?;
        Ok(())
    }

    /// Najpodobnejší vzor → (popis, kosínová podobnosť).
    pub fn find_similar(&self, v: &[f32]) -> (String, f32) {
        if self.dim == 0 || self.emb.is_empty() || v.len() != self.dim {
            return (String::new(), 0.0);
        }
        let vn = norm(v);
        let mut best = (String::new(), -1.0f32);
        for (i, label) in self.label.iter().enumerate() {
            let row = &self.emb[i * self.dim..(i + 1) * self.dim];
            let rn = norm(row);
            let mut dot = 0.0f32;
            for k in 0..self.dim {
                dot += row[k] * v[k];
            }
            let sim = dot / (rn * vn + 1e-9);
            if sim > best.1 {
                best.1 = sim;
                best.0 = label.clone();
            }
        }
        (best.0, best.1)
    }
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12)
}

// ---------- naučené spojenia (json) ------------------------------------------

pub type Words = serde_json::Map<String, serde_json::Value>;

pub fn load_words(path: &std::path::Path) -> Words {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

pub fn save_words(path: &std::path::Path, words: &Words) {
    if words.is_empty() {
        return;
    }
    if let Ok(s) = serde_json::to_string_pretty(words) {
        let _ = std::fs::write(path, s);
    }
}

/// Zapamätá spojenia slovo → popis (len slová, ktoré v popisoch nie sú).
pub fn learn_words(
    path: &str,
    desc: &str,
    descriptions: &[String],
    words: &mut Words,
) -> Vec<String> {
    let mut learned_now = Vec::new();
    for kw in crate::names::filename_keywords(path) {
        if descriptions
            .iter()
            .any(|d| crate::names::keyword_in_description(&kw, d))
        {
            continue;
        }
        let entry = words
            .entry(kw.clone())
            .or_insert_with(|| serde_json::json!({}));
        if let Some(counts) = entry.as_object_mut() {
            let c = counts.get(desc).and_then(|x| x.as_i64()).unwrap_or(0) + 1;
            counts.insert(desc.to_string(), serde_json::json!(c));
            // pamäť: max 3 najčastejšie popisy na slovo
            if counts.len() > 3 {
                let mut items: Vec<(String, i64)> = counts
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0)))
                    .collect();
                items.sort_by_key(|(_, c)| *c);
                let to_remove = items[0].0.clone();
                counts.remove(&to_remove);
            }
        }
        learned_now.push(kw);
    }
    learned_now
}

/// HashMap pre rýchle „bol už použitý?" (nepoužívané zatiaľ).
#[allow(dead_code)]
pub type WordIndex = HashMap<String, usize>;
