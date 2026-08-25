//! Analyzátor zvukových súborov – RUST verzia (CLI, na testovanie výkonu).
//! Rovnaký CLAP model, rovnaké pravidlá, rovnaké naučené dáta ako Python.
//!
//! Použitie:
//!   analyzator-rs <priecinok> [--popisy popisy.txt] [--segments 4]
//!                [--min-istota 50] [--model-dir models/clap_htsat_unfused_onnx]
//!                [--bez-preskocenia] [--istota-do-popisu] [--vlakien 4]

mod audio;
mod feat;
mod learned;
mod model;
mod names;
mod tags;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use rayon::prelude::*;

const AUDIO_SIM_MIN: f64 = 0.80;
const AUDIO_SIM_BOOST: f64 = 1.2;
const MULTI_RATIO: f64 = 0.4;
const MULTI_EXTRA_MAX: usize = 2;

struct Opts {
    folder: PathBuf,
    popisy: PathBuf,
    segments: usize,
    min_istota: f64,
    model_dir: PathBuf,
    skip_by_name: bool,
    istota_do_popisu: bool,
    vlakien: usize,
    dump_mel: Option<PathBuf>,
    debug_audio: bool,
    dump_emb: Option<PathBuf>,
    json_out: Option<PathBuf>,
}

fn parse_args() -> Result<Opts> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        std::process::exit(1);
    }
    if args.iter().any(|a| a == "--pomoc" || a == "--help" || a == "-h") {
        print_usage();
        std::process::exit(0);
    }
    let mut folder: Option<PathBuf> = None;
    let mut popisy = PathBuf::from("popisy.txt");
    let mut segments = 4usize;
    let mut min_istota = 50.0f64;
    let mut model_dir = PathBuf::from("models/clap_htsat_unfused_onnx");
    let mut skip_by_name = true;
    let mut istota_do_popisu = false;
    let mut vlakien = 4usize;
    let mut dump_mel: Option<PathBuf> = None;
    let mut debug_audio = false;
    let mut dump_emb: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        macro_rules! val { () => {{
            i += 1;
            if i >= args.len() { anyhow::bail!("chýba hodnota za {}", a); }
            args[i].clone()
        }}}
        match a {
            "--popisy" => popisy = PathBuf::from(val!()),
            "--segments" => segments = val!().parse()?,
            "--min-istota" => min_istota = val!().parse()?,
            "--model-dir" => model_dir = PathBuf::from(val!()),
            "--vlakien" => vlakien = val!().parse()?,
            "--bez-preskocenia" => skip_by_name = false,
            "--dump-mel" => dump_mel = Some(PathBuf::from(val!())),
            "--debug-audio" => debug_audio = true,
            "--dump-emb" => dump_emb = Some(PathBuf::from(val!())),
            "--json" => json_out = Some(PathBuf::from(val!())),
            "--istota-do-popisu" => istota_do_popisu = true,
            _ => {
                if a.starts_with("--") {
                    anyhow::bail!("neznámy parameter {a}");
                }
                folder = Some(PathBuf::from(a));
            }
        }
        i += 1;
    }
    Ok(Opts {
        folder: folder.context("chýba priečinok so zvukmi")?,
        popisy,
        segments,
        min_istota: min_istota.clamp(0.0, 95.0) / 100.0,
        model_dir,
        skip_by_name,
        istota_do_popisu,
        vlakien: vlakien.clamp(1, 16),
        dump_mel,
        debug_audio,
        dump_emb,
        json_out,
    })
}

fn print_usage() {
    println!("Analyzátor zvukových súborov – Rust verzia (rýchly test výkonu)");
    println!();
    println!("Použitie: analyzator-rs PRIECINOK [možnosti]");
    println!("  --popisy SUBOR        kandidátske popisy, jeden na riadok (default: popisy.txt)");
    println!("  --segments N          počet 10 s okien na súbor (default 4)");
    println!("  --min-istota P        popis pod P % sa nezapisuje (default 50)");
    println!("  --model-dir DIR       priečinok s ONNX modelmi (default models/clap_htsat_unfused_onnx)");
    println!("  --bez-preskocenia     vypne preskakovanie AI podľa názvu súboru");
    println!("  --istota-do-popisu    zapíše istotu do popisu (napr. ‘(istota 87 %)’)");
    println!("  --vlakien N           paralelné dekódovanie (default 4)");
}

fn load_descriptions(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("čítam {}", path.display()))?;
    let descs: Vec<String> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect();
    if descs.len() < 2 {
        anyhow::bail!("v {} musia byť aspoň 2 popisy (jeden na riadok)", path.display());
    }
    Ok(descs)
}

fn scan_audio(dir: &Path, out: &mut Vec<PathBuf>) {
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

/// Prirodzené triedenie (ako Python _natural_key): cislo2 < cislo10.
fn natural_key(p: &Path) -> Vec<(u8, u64)> {
    let s = p.file_name().and_then(|x| x.to_str()).unwrap_or("").to_lowercase();
    let mut key = Vec::new();
    let mut cur = String::new();
    let mut in_num = false;
    for ch in s.chars() {
        let digit = ch.is_ascii_digit();
        if !cur.is_empty() && digit != in_num {
            flush(&mut cur, &mut key);
        }
        in_num = digit;
        cur.push(ch);
    }
    flush(&mut cur, &mut key);
    key
}
fn flush(cur: &mut String, key: &mut Vec<(u8, u64)>) {
    let s = std::mem::take(cur);
    if let Ok(n) = s.parse::<u64>() {
        key.push((1, n));
    } else if !s.is_empty() {
        let h = s.chars().map(|c| c as u32 as u64).fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b));
        key.push((0, h));
    }
}

fn basename(p: &Path) -> String {
    p.file_name().and_then(|x| x.to_str()).unwrap_or("?").to_string()
}

fn main() -> Result<()> {
    let t_start = Instant::now();
    let opts = parse_args()?;
    let descriptions = load_descriptions(&opts.popisy)?;

    // --- súborová ani naučené dáta ---------------------------------------
    let mut files: Vec<PathBuf> = Vec::new();
    scan_audio(&opts.folder, &mut files);
    files.sort_by(|a, b| natural_key(a).cmp(&natural_key(b)));
    if files.is_empty() {
        anyhow::bail!("v {} nie sú žiadne zvukové súbory (wav/mp3/ogg/flac)", opts.folder.display());
    }
    println!("Súborov: {} | popisov: {} | okná: {} | prah istoty: {:.0} %",
             files.len(), descriptions.len(), opts.segments, opts.min_istota * 100.0);

    let words_path = std::env::current_dir()?.join("naucene_spojenia.json");
    let pats_path = std::env::current_dir()?.join("naucene_vzory.npz");
    let mut words = learned::load_words(&words_path);
    let mut patterns = learned::Patterns::load(&pats_path);

    // --- 1) jednoznačné názvy → preskoč AI --------------------------------
    let mut name_skips: Vec<Option<String>> = vec![None; files.len()];
    let need_ai;
    if opts.skip_by_name {
        for (i, f) in files.iter().enumerate() {
            if let Some(p) = f.to_str() {
                if let Some(d) = names::name_skip_description(p, &descriptions, &words) {
                    name_skips[i] = Some(d);
                }
            }
        }
        let n_skip = name_skips.iter().filter(|x| x.is_some()).count();
        need_ai = files.len() - n_skip;
        if n_skip > 0 {
            println!("⚡ {n_skip} z {} súborov má jednoznačný názov – pri nich sa AI nepoužije.",
                     files.len());
        }
    } else {
        need_ai = files.len();
    }

    // --- 2) model (iba ak je čo analyzovať) --------------------------------
    let mut mdl = None;
    let mut text_emb: Vec<Vec<f32>> = Vec::new();
    if need_ai > 0 {
        let t0 = Instant::now();
        mdl = Some(model::ClapModel::load(&opts.model_dir)?);
        let m = mdl.as_mut().unwrap();
        println!("Model načítaný za {:.1} s ({})", t0.elapsed().as_secs_f32(), m.backend_info);
        let t0 = Instant::now();
        text_emb = m.embed_texts(&descriptions)?;
        println!("📝 Embeddingy {} popisov za {:.1} s", descriptions.len(), t0.elapsed().as_secs_f32());
    } else {
        println!("⚡ Podľa názvov netreba AI – model sa nespúšťa.");
    }

    // --- 3) paralelná príprava (dekód + resample + mel) ----------------------
    //        GPU/CPU inference beží potom sekvenčne – ako Python pipeline.
    let ai_idx: Vec<usize> = (0..files.len())
        .filter(|i| name_skips[*i].is_none())
        .collect();
    let decode_ms = AtomicU32::new(0);
    let mel_ms = AtomicU32::new(0);
    let bank = feat::mel_filter_bank_slaney();
    let segments = opts.segments;
    let dump_mel_target = opts.dump_mel.clone();
    let debug_audio = opts.debug_audio;
    let dump_emb_target = opts.dump_emb.clone();
    // záznamy pre --json (hodnoty vo f64 ako Python)
    let mut json_records: Vec<serde_json::Value> = Vec::new();

    let prepared: Vec<anyhow::Result<(usize, Vec<Vec<f32>>)>> = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(opts.vlakien)
            .build()?;
        pool.install(|| {            ai_idx
                .into_par_iter()
                .map(|i| {
                    let p = files[i].to_str().context("cesta")?;
                    let t0 = Instant::now();
                    let (raw, sr0) = audio::decode_mono(p)?;
                    if debug_audio {
                        let rms = (raw.iter().map(|x| x * x).sum::<f32>()
                            / raw.len().max(1) as f32).sqrt();
                        eprintln!(
                            "[debug] {}: sr={} n={} rms={:.5} prvé={:?}",
                            basename(&files[i]), sr0, raw.len(), rms,
                            &raw[..raw.len().min(6)]
                        );
                    }
                    let y = audio::resample(&raw, sr0, audio::TARGET_SR);
                    let dur = y.len() as f64 / audio::TARGET_SR as f64;
                    let starts = audio::window_starts(dur, segments);
                    let wins = audio::cut_windows(&y, &starts);
                    let d_ms = t0.elapsed().as_millis() as u32;
                    let t1 = Instant::now();
                    let feats: Vec<Vec<f32>> = wins.iter().map(|w| feat::log_mel(w, &bank)).collect();
                    if i == 0 {
                        if let Some(out) = &dump_mel_target {
                            use std::io::Write;
                            let mut f = std::fs::File::create(out).unwrap();
                            let n = feats[0].len();
                            let head = format!(
                                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, 64), }}", 1001);
                            let total = (10 + head.len() + 1 + 63) / 64 * 64;
                            f.write_all(b"\x93NUMPY").unwrap();
                            f.write_all(&[1u8, 0]).unwrap();
                            f.write_all(&((total - 10) as u16).to_le_bytes()).unwrap();
                            f.write_all(head.as_bytes()).unwrap();
                            while f.metadata().unwrap().len() < total as u64 - 1 {
                                f.write_all(b" ").unwrap();
                            }
                            f.write_all(b"\n").unwrap();
                            for v in &feats[0] {
                                f.write_all(&v.to_le_bytes()).unwrap();
                            }
                            let _ = n;
                        }
                    }
                    decode_ms.fetch_add(d_ms, Ordering::Relaxed);
                    mel_ms.fetch_add(t1.elapsed().as_millis() as u32, Ordering::Relaxed);
                    Ok((i, feats))
                })
                .collect()
        })
    };

    // --- 4) inference + pravidlá + zápisy ------------------------------------
    let mut ok = 0u32;
    let mut low = 0u32;
    let mut names_cnt = 0u32;
    let mut err = 0u32;
    let mut infer_ms_sum = 0u128;
    let mut patterns_changed = false;
    let mut feats_by_idx: std::collections::HashMap<usize, Vec<Vec<f32>>> =
        prepared.into_iter().filter_map(|r| r.ok()).collect();

    for (i, f) in files.iter().enumerate() {
        let p = f.to_str().unwrap_or("?");
        let name = basename(f);
        // (a) jednoznačný názov
        if let Some(desc) = &name_skips[i] {
            match tags::write_description(p, desc) {
                Ok(_) => {
                    println!("⚡ {name} → ‘{desc}’ (názov jednoznačne sedí)");
                    names_cnt += 1;
                    for w in learned::learn_words(p, desc, &descriptions, &mut words) {
                        println!("🧠 Naučené spojenie: ‘{w}’ → ‘{desc}’");
                    }
                }
                Err(e) => {
                    println!("✖ {name}: {e}");
                    err += 1;
                }
            }
            continue;
        }
        // (b) AI analýza
        let Some(feats) = feats_by_idx.remove(&i) else {
            println!("✖ {name}: príprava zlyhala");
            err += 1;
            continue;
        };
        let m = match mdl.as_mut() {
            Some(m) => m,
            None => unreachable!(),
        };
        let t0 = Instant::now();
        let window_embs = match m.embed_audio(&feats) {
            Ok(e) => e,
            Err(e) => {
                println!("✖ {name}: {e}");
                err += 1;
                continue;
            }
        };
        infer_ms_sum += t0.elapsed().as_millis();
        if dump_emb_target.is_some() && json_records.is_empty() {
            let flat: Vec<f32> = window_embs.concat();
            let _ = learned::write_npy_flat(
                dump_emb_target.as_ref().unwrap(),
                &[window_embs.len(), window_embs[0].len()],
                &flat,
            );
        }

        // priemer okien + podobnosť s textami (rovnaká matematika ako Python)
        let d = window_embs[0].len();
        let mut mean = vec![0.0f32; d];
        for w in &window_embs {
            for k in 0..d {
                mean[k] += w[k];
            }
        }
        let audio_emb = model::l2norm(&mean);
        let scale = m.logit_scale;
        let logits: Vec<f32> = text_emb
            .iter()
            .map(|t| {
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += t[k] * audio_emb[k];
                }
                dot * scale
            })
            .collect();
        let probs = model::softmax(&logits);
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap());
        let best = order[0];

        // viac popisov (dlhšia nahrávka, viac zvukov)
        let mut additional: Vec<(String, f32)> = Vec::new();
        if window_embs.len() >= 2 {
            let winners: Vec<usize> = window_embs
                .iter()
                .map(|w| {
                    let lg: Vec<f32> = text_emb
                        .iter()
                        .map(|t| {
                            let mut dot = 0.0f32;
                            for k in 0..d {
                                dot += t[k] * w[k];
                            }
                            dot * scale
                        })
                        .collect();
                    let pr = model::softmax(&lg);
                    (0..pr.len()).reduce(|a, b| if pr[b] > pr[a] { b } else { a }).unwrap()
                })
                .collect();
            let mut extra: Vec<(usize, f32)> = (0..descriptions.len())
                .filter(|j| *j != best)
                .map(|j| (j, probs[j]))
                .filter(|(j, mp)| winners.iter().any(|w| w == j) && *mp as f64 >= MULTI_RATIO * probs[best] as f64)
                .collect();
            extra.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            additional = extra
                .into_iter()
                .take(MULTI_EXTRA_MAX)
                .map(|(j, mp)| (descriptions[j].clone(), mp))
                .collect();
        }

        // boosty: názov súboru + naučený zvukový vzor
        let mut conf = probs[best] as f64;
        let mut name_boosted = false;
        let mut pattern_boosted = false;
        if names::name_matches_description(p, &descriptions[best], &words) {
            conf = (conf * names::NAME_BOOST_FACTOR).min(names::NAME_BOOST_CAP);
            name_boosted = true;
        }
        {
            let (lbl, sim) = patterns.find_similar(&audio_emb);
            if lbl == descriptions[best] && sim as f64 >= AUDIO_SIM_MIN {
                conf = (conf * AUDIO_SIM_BOOST).min(names::NAME_BOOST_CAP);
                pattern_boosted = true;
            }
        }

        json_records.push(serde_json::json!({
            "file": name,
            "winner": descriptions[best],
            "p": probs[best] as f64,
            "conf": conf,
            "name_boosted": name_boosted,
            "pattern_boosted": pattern_boosted,
            "ranking": order.iter()
                .map(|&i| serde_json::json!([descriptions[i], probs[i] as f64]))
                .collect::<Vec<_>>(),
            "additional": additional.iter()
                .map(|(d2, mp)| serde_json::json!([d2, *mp as f64]))
                .collect::<Vec<_>>(),
        }));
        if conf < opts.min_istota {
            // nízká istota → nezapsať + zmazať starý "naš" popis
            let mut removed = String::new();
            let old = tags::read_description(p);
            if !old.is_empty() && old_is_ours(&old, &descriptions) {
                if let Ok(msg) = tags::remove_description(p) {
                    removed = format!(" | {msg}");
                }
            }
            println!("⚠ {name}: istota {:.0} % je pod prahom {:.0} % → popis nezapsaný{removed}. Najlepší tip: ‘{}’",
                     conf * 100.0, opts.min_istota * 100.0, descriptions[best]);
            low += 1;
            continue;
        }

        let text_out = if additional.is_empty() {
            descriptions[best].clone()
        } else {
            let mut t = descriptions[best].clone();
            for (d2, _) in &additional {
                t.push_str(" + ");
                t.push_str(d2);
            }
            t
        };
        let final_text = if opts.istota_do_popisu {
            format!("{} (istota {:.0} %)", text_out, conf * 100.0)
        } else {
            text_out.clone()
        };
        match tags::write_description(p, &final_text) {
            Ok(_) => {
                let mut notes = Vec::new();
                if name_boosted { notes.push("názov podporil".to_string()); }
                if pattern_boosted { notes.push("🧠 podobný naučenému zvuku".to_string()); }
                if !additional.is_empty() {
                    notes.push(format!("viac zvukov: {}", additional.iter()
                        .map(|(d2, mp)| format!("{d2} ({:.0} %)", mp * 100.0))
                        .collect::<Vec<_>>().join(", ")));
                }
                let note = if notes.is_empty() { String::new() } else { format!(" | {}", notes.join(" · ")) };
                println!("✔ {name} → ‘{text_out}’ ({:.0} %){note}", conf * 100.0);
                ok += 1;
                for w in learned::learn_words(p, &descriptions[best], &descriptions, &mut words) {
                    println!("🧠 Naučené spojenie: ‘{w}’ → ‘{}’", descriptions[best]);
                }
                if additional.is_empty() {
                    patterns.add(&audio_emb, &descriptions[best]);
                    patterns_changed = true;
                }
            }
            Err(e) => {
                println!("✖ {name}: {e}");
                err += 1;
            }
        }
    }

    // --- 5) uložiť naučené + súhrn -------------------------------------------
    learned::save_words(&words_path, &words);
    if patterns_changed {
        patterns.save()?;
    }
    let total = t_start.elapsed().as_secs_f32();
    println!();
    if let Some(p) = &opts.json_out {
        std::fs::write(p, serde_json::to_string_pretty(&json_records).unwrap_or_default())?;
    }
    println!("Dokončené: ✔ {ok} hotovo, ⚡ {names_cnt} podľa názvu, ⚠ {low} s nízkou istotou, ✖ {err} chýb");
    println!("Čas celkom: {total:.1} s | dekód: {:.1} s | mel: {:.1} s | inference: {:.1} s | vlákna: {}",
             decode_ms.load(Ordering::Relaxed) as f32 / 1000.0,
             mel_ms.load(Ordering::Relaxed) as f32 / 1000.0,
             infer_ms_sum as f32 / 1000.0,
             opts.vlakien);
    Ok(())
}

/// Vyzerá starý popis, ako keby ho napísala táto appka?
fn old_is_ours(old: &str, descriptions: &[String]) -> bool {
    let old = old.trim();
    if old.is_empty() { return false; }
    if old.contains("(istota") { return true; }
    let core = old.split("(istota").next().unwrap_or("").trim().to_lowercase();
    descriptions.iter().any(|d| {
        let dl = d.to_lowercase();
        dl == core || dl.contains(&core) || core.contains(&dl)
    })
}
