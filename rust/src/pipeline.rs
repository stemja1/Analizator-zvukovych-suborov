//! Spoločný analyzačný pipeline – používa ho CLI (main.rs) aj GUI (gui.rs).
//!
//! Celý beh beží v samostatnom vlákne; priebeh sa hlási cez `Event`
//! (GUI ich posiela do okna, CLI ich vypisuje). Beh možno kedykoľvek
//! zastaviť cez `AtomicBool`.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::model;
use crate::{audio, feat, names, tags};

pub const MULTI_RATIO: f64 = 0.4;
pub const MULTI_EXTRA_MAX: usize = 2;

/// Vstup celého behu (súbory aj popisy už vybrané a usporiadané).
pub struct RunOptions {
    pub files: Vec<PathBuf>,
    pub descriptions: Vec<String>,
    pub segments: usize,
    /// 0..1 – popis pod touto istotou sa nezapisuje
    pub min_istota: f64,
    pub model_dir: PathBuf,
    pub skip_by_name: bool,
    pub istota_do_popisu: bool,
    pub vlakien: usize,
    // ladiace voľby (GUI necháva None)
    pub dump_mel: Option<PathBuf>,
    pub debug_audio: bool,
    pub dump_emb: Option<PathBuf>,
    pub json_out: Option<PathBuf>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            files: Vec::new(),
            descriptions: Vec::new(),
            segments: 4,
            min_istota: 0.5,
            model_dir: PathBuf::from("models/clap_htsat_unfused_onnx"),
            skip_by_name: true,
            istota_do_popisu: false,
            vlakien: 4,
            dump_mel: None,
            debug_audio: false,
            dump_emb: None,
            json_out: None,
        }
    }
}

/// Správy o priebehu (idx = poradie súboru vo `files`).
pub enum Event {
    /// řádok do logu
    Info(String),
    /// AI sa preskočila – názov jednoznačne určil popis
    NameSkip { idx: usize, desc: String },
    /// hotovo – popis zapísaný
    FileDone { idx: usize, text: String, conf: f64, notes: String },
    /// istota pod prahom – popis nezapísaný
    FileLow { idx: usize, tip: String, conf: f64, detail: String },
    /// chyba súboru
    FileErr { idx: usize, msg: String },
    /// koniec behu
    Finished {
        ok: u32,
        names: u32,
        low: u32,
        err: u32,
        total_s: f32,
        decode_s: f32,
        mel_s: f32,
        infer_s: f32,
        cancelled: bool,
    },
}

/// Celý beh analýzy. Vráti Err len pri chybe, ktorá zhabe celý beh
/// (napr. chybný model); chyby jednotlivých súborov idú cez FileErr.
pub fn run(
    opts: RunOptions,
    cancel: &AtomicBool,
    emit: &(dyn Fn(Event) + Sync),
) -> Result<()> {
    let t_start = Instant::now();
    let files = &opts.files;
    let descriptions = &opts.descriptions;
    if files.is_empty() {
        anyhow::bail!("žiadne zvukové súbory (wav/mp3/ogg/flac)");
    }
    if descriptions.len() < 2 {
        anyhow::bail!("sú potrebné aspoň 2 popisy");
    }
    emit(Event::Info(format!(
        "Súborov: {} | popisov: {} | okná: {} | prah istoty: {:.0} %",
        files.len(),
        descriptions.len(),
        opts.segments,
        opts.min_istota * 100.0
    )));

    // --- 1) jednoznačné názvy → preskoč AI --------------------------------
    let mut name_skips: Vec<Option<String>> = vec![None; files.len()];
    let need_ai;
    if opts.skip_by_name {
        for (i, f) in files.iter().enumerate() {
            if let Some(p) = f.to_str() {
                if let Some(d) = names::name_skip_description(p, descriptions) {
                    name_skips[i] = Some(d);
                }
            }
        }
        let n_skip = name_skips.iter().filter(|x| x.is_some()).count();
        need_ai = files.len() - n_skip;
        if n_skip > 0 {
            emit(Event::Info(format!(
                "⚡ {n_skip} z {} súborov má jednoznačný názov – pri nich sa AI nepoužije.",
                files.len()
            )));
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
        emit(Event::Info(format!(
            "Model načítaný za {:.1} s ({})",
            t0.elapsed().as_secs_f32(),
            m.backend_info
        )));
        let t0 = Instant::now();
        text_emb = m.embed_texts(descriptions)?;
        emit(Event::Info(format!(
            "📝 Embeddingy {} popisov za {:.1} s",
            descriptions.len(),
            t0.elapsed().as_secs_f32()
        )));
    } else {
        emit(Event::Info("⚡ Podľa názvov netreba AI – model sa nespúšťa.".into()));
    }

    // --- 3) paralelná príprava (dekód + resample + mel) ---------------------
    let ai_idx: Vec<usize> = (0..files.len())
        .filter(|i| name_skips[*i].is_none())
        .collect();
    let decode_ms = AtomicU32::new(0);
    let mel_ms = AtomicU32::new(0);
    let bank = feat::mel_filter_bank_slaney();
    let segments = opts.segments;
    let dump_mel_target = opts.dump_mel.clone();
    let debug_audio = opts.debug_audio;
    let mut json_records: Vec<serde_json::Value> = Vec::new();

    let prepared: Vec<anyhow::Result<(usize, Vec<Vec<f32>>)>> = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(opts.vlakien)
            .build()?;
        pool.install(|| {
            ai_idx
                .into_par_iter()
                .map(|i| {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(anyhow::anyhow!("zrušené"));
                    }
                    let p = files[i].to_str().context("cesta")?;
                    let t0 = Instant::now();
                    let (raw, sr0) = audio::decode_mono(p)?;
                    if debug_audio {
                        let rms = (raw.iter().map(|x| x * x).sum::<f32>()
                            / raw.len().max(1) as f32)
                            .sqrt();
                        eprintln!(
                            "[debug] {}: sr={} n={} rms={:.5}",
                            crate::basename(&files[i]),
                            sr0,
                            raw.len(),
                            rms
                        );
                    }
                    let y = audio::resample(&raw, sr0, audio::TARGET_SR);
                    let dur = y.len() as f64 / audio::TARGET_SR as f64;
                    let starts = audio::window_starts(dur, segments);
                    let wins = audio::cut_windows(&y, &starts);
                    let d_ms = t0.elapsed().as_millis() as u32;
                    let t1 = Instant::now();
                    let feats: Vec<Vec<f32>> =
                        wins.iter().map(|w| feat::log_mel(w, &bank)).collect();
                    if i == 0 {
                        if let Some(out) = &dump_mel_target {
                            use std::io::Write;
                            let mut f = std::fs::File::create(out).unwrap();
                            let head = format!(
                                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, 64), }}",
                                1001
                            );
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
                        }
                    }
                    decode_ms.fetch_add(d_ms, Ordering::Relaxed);
                    mel_ms.fetch_add(t1.elapsed().as_millis() as u32, Ordering::Relaxed);
                    Ok((i, feats))
                })
                .collect()
        })
    };

    // --- 4) inference + pravidlá + zápisy -----------------------------------
    let mut ok = 0u32;
    let mut low = 0u32;
    let mut names_cnt = 0u32;
    let mut err = 0u32;
    let mut infer_ms_sum = 0u128;
    let mut feats_by_idx: std::collections::HashMap<usize, Vec<Vec<f32>>> =
        prepared.into_iter().filter_map(|r| r.ok()).collect();

    let mut cancelled = false;
    for (i, f) in files.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            cancelled = true;
            break;
        }
        let p = f.to_str().unwrap_or("?");
        let name = crate::basename(f);
        // (a) jednoznačný názov
        if let Some(desc) = &name_skips[i] {
            match tags::write_description(p, desc) {
                Ok(_) => {
                    emit(Event::Info(format!("⚡ {name} → ‘{desc}’ (názov jednoznačne sedí)")));
                    emit(Event::NameSkip { idx: i, desc: desc.clone() });
                    names_cnt += 1;
                }
                Err(e) => {
                    emit(Event::Info(format!("✖ {name}: {e}")));
                    emit(Event::FileErr { idx: i, msg: e.to_string() });
                    err += 1;
                }
            }
            continue;
        }
        // (b) AI analýza
        let Some(feats) = feats_by_idx.remove(&i) else {
            emit(Event::Info(format!("✖ {name}: príprava zlyhala")));
            emit(Event::FileErr { idx: i, msg: "príprava zlyhala".into() });
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
                emit(Event::Info(format!("✖ {name}: {e}")));
                emit(Event::FileErr { idx: i, msg: e.to_string() });
                err += 1;
                continue;
            }
        };
        infer_ms_sum += t0.elapsed().as_millis();
        if opts.dump_emb.is_some() && json_records.is_empty() {
            let flat: Vec<f32> = window_embs.concat();
            crate::write_npy_flat(
                opts.dump_emb.as_ref().unwrap(),
                &[window_embs.len(), window_embs[0].len()],
                &flat,
            )
            .ok();
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
                    (0..pr.len())
                        .reduce(|a, b| if pr[b] > pr[a] { b } else { a })
                        .unwrap()
                })
                .collect();
            let mut extra: Vec<(usize, f32)> = (0..descriptions.len())
                .filter(|j| *j != best)
                .map(|j| (j, probs[j]))
                .filter(|(j, mp)| {
                    winners.iter().any(|w| w == j)
                        && *mp as f64 >= MULTI_RATIO * probs[best] as f64
                })
                .collect();
            extra.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            additional = extra
                .into_iter()
                .take(MULTI_EXTRA_MAX)
                .map(|(j, mp)| (descriptions[j].clone(), mp))
                .collect();
        }

        // posilnenie názvom súboru (priama zhoda slova v popise)
        let mut conf = probs[best] as f64;
        let mut name_boosted = false;
        if names::name_matches_description(p, &descriptions[best]) {
            conf = (conf * names::NAME_BOOST_FACTOR).min(names::NAME_BOOST_CAP);
            name_boosted = true;
        }

        json_records.push(serde_json::json!({
            "file": name,
            "winner": descriptions[best],
            "p": probs[best] as f64,
            "conf": conf,
            "name_boosted": name_boosted,
            "ranking": order.iter()
                .map(|&i2| serde_json::json!([descriptions[i2], probs[i2] as f64]))
                .collect::<Vec<_>>(),
            "additional": additional.iter()
                .map(|(d2, mp)| serde_json::json!([d2, *mp as f64]))
                .collect::<Vec<_>>(),
        }));

        if conf < opts.min_istota {
            // nízká istota → nezapsať + zmazať starý "naš" popis
            let mut removed = String::new();
            let old = tags::read_description(p);
            if !old.is_empty() && old_is_ours(&old, descriptions) {
                if let Ok(msg) = tags::remove_description(p) {
                    removed = format!(" | {msg}");
                }
            }
            let detail = format!(
                "istota {:.0} % < {:.0} % → popis nezapísaný{}",
                conf * 100.0,
                opts.min_istota * 100.0,
                removed
            );
            emit(Event::Info(format!(
                "⚠ {name}: {detail}. Najlepší tip: ‘{}’",
                descriptions[best]
            )));
            emit(Event::FileLow {
                idx: i,
                tip: descriptions[best].clone(),
                conf,
                detail,
            });
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
                if name_boosted {
                    notes.push("názov podporil".to_string());
                }
                if !additional.is_empty() {
                    notes.push(format!(
                        "viac zvukov: {}",
                        additional
                            .iter()
                            .map(|(d2, mp)| format!("{d2} ({:.0} %)", mp * 100.0))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                let note = notes.join(" · ");
                emit(Event::Info(format!(
                    "✔ {name} → ‘{text_out}’ ({:.0} %){}",
                    conf * 100.0,
                    if note.is_empty() { String::new() } else { format!(" | {note}") }
                )));
                emit(Event::FileDone {
                    idx: i,
                    text: text_out,
                    conf,
                    notes: note,
                });
                ok += 1;
            }
            Err(e) => {
                emit(Event::Info(format!("✖ {name}: {e}")));
                emit(Event::FileErr { idx: i, msg: e.to_string() });
                err += 1;
            }
        }
    }

    // --- 5) súhrn --------------------------------------------------------------
    let total = t_start.elapsed().as_secs_f32();
    if let Some(p) = &opts.json_out {
        std::fs::write(p, serde_json::to_string_pretty(&json_records).unwrap_or_default())?;
    }
    let decode_s = decode_ms.load(Ordering::Relaxed) as f32 / 1000.0;
    let mel_s = mel_ms.load(Ordering::Relaxed) as f32 / 1000.0;
    let infer_s = infer_ms_sum as f32 / 1000.0;
    let mut summary = format!(
        "Dokončené: ✔ {ok} hotovo, ⚡ {names_cnt} podľa názvu, ⚠ {low} s nízkou istotou, ✖ {err} chýb"
    );
    if cancelled {
        summary.push_str(" (beh bol zrušený – zvyšok preskočený)");
    }
    emit(Event::Info(summary));
    emit(Event::Info(format!(
        "Čas celkom: {total:.1} s | dekód: {decode_s:.1} s | mel: {mel_s:.1} s | inference: {infer_s:.1} s | vlákna: {}",
        opts.vlakien
    )));
    emit(Event::Finished {
        ok,
        names: names_cnt,
        low,
        err,
        total_s: total,
        decode_s,
        mel_s,
        infer_s,
        cancelled,
    });
    Ok(())
}

/// Vyzerá starý popis, ako keby ho napísala táto appka?
pub fn old_is_ours(old: &str, descriptions: &[String]) -> bool {
    let old = old.trim();
    if old.is_empty() {
        return false;
    }
    if old.contains("(istota") {
        return true;
    }
    let core = old
        .split("(istota")
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    descriptions.iter().any(|d| {
        let dl = d.to_lowercase();
        dl == core || dl.contains(&core) || core.contains(&dl)
    })
}
