//! Analyzátor zvukových súborov – Rust verzia (knižnica).
//! CLI: src/main.rs, GUI: src/gui.rs – obe používajú túto knižnicu.

pub mod audio;
pub mod feat;
pub mod model;
pub mod names;
pub mod pipeline;
pub mod tags;
pub mod updater;

use std::path::{Path, PathBuf};

/// Rekurzívne nájde všetky zvukové súbory (wav/mp3/ogg/flac).
pub fn scan_audio(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            scan_audio(&p, out);
        } else if let Some(ext) = p.extension().and_then(|x| x.to_str()) {
            if matches!(ext.to_lowercase().as_str(), "wav" | "mp3" | "ogg" | "flac") {
                out.push(p);
            }
        }
    }
}

/// Súbor je zvukový? (podľa prípony)
pub fn is_audio_file(p: &Path) -> bool {
    p.extension()
        .and_then(|x| x.to_str())
        .map(|ext| matches!(ext.to_lowercase().as_str(), "wav" | "mp3" | "ogg" | "flac"))
        .unwrap_or(false)
}

/// Prirodzené triedenie (ako Python _natural_key): cislo2 < cislo10.
pub fn natural_key(p: &Path) -> Vec<(u8, u64)> {
    let s = p
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mut key = Vec::new();
    let mut cur = String::new();
    let mut in_num = false;
    for ch in s.chars() {
        let digit = ch.is_ascii_digit();
        if !cur.is_empty() && digit != in_num {
            flush_key(&mut cur, &mut key);
        }
        in_num = digit;
        cur.push(ch);
    }
    flush_key(&mut cur, &mut key);
    key
}
fn flush_key(cur: &mut String, key: &mut Vec<(u8, u64)>) {
    let s = std::mem::take(cur);
    if let Ok(n) = s.parse::<u64>() {
        key.push((1, n));
    } else if !s.is_empty() {
        let h = s
            .chars()
            .map(|c| c as u32 as u64)
            .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b));
        key.push((0, h));
    }
}

/// Usporiada zvukové súbory prirodzene (2 pred 10).
pub fn sort_natural(files: &mut Vec<PathBuf>) {
    files.sort_by(|a, b| natural_key(a).cmp(&natural_key(b)));
}

pub fn basename(p: &Path) -> String {
    p.file_name()
        .and_then(|x| x.to_str())
        .unwrap_or("?")
        .to_string()
}

/// Text so zoznamom popisov → čistý zoznam (prázdne riadky a # sekcie von).
pub fn parse_descriptions(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect()
}

/// npy hlavička (používa --dump-emb na ladiaci výpis).
fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
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
    head.extend_from_slice(&[1u8, 0]);
    let unpadded_len = 10 + dict.len() + 1;
    let total_len = (unpadded_len + 63) / 64 * 64;
    head.extend_from_slice(&((total_len - 10) as u16).to_le_bytes());
    head.extend_from_slice(dict.as_bytes());
    while head.len() < total_len {
        head.push(b' ');
    }
    head[total_len - 1] = b'\n';
    head
}

/// Zapíše (shape, f32 dáta) ako npz (zip s data.npy) – ladiaci --dump-emb.
pub fn write_npy_flat(path: &Path, shape: &[usize], data: &[f32]) -> anyhow::Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(path)?;
    let mut w = zip::ZipWriter::new(f);
    let opts: zip::write::SimpleFileOptions = Default::default();
    w.start_file("data.npy", opts)?;
    w.write_all(&npy_header("<f4", shape))?;
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    w.finish()?;
    Ok(())
}
