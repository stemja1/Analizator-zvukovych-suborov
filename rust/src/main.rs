//! Analyzátor zvukových súborov – Rust verzia, CLI (príkazový riadok).
//! GUI verzia: analyzator-gui (src/gui.rs). Obe zdieľajú knižnicu.
//!
//! Použitie:
//!   analyzator-rs <priecinok> [--popisy popisy.txt] [--segments 4]
//!                [--min-istota 50] [--model-dir models/clap_htsat_unfused_onnx]
//!                [--bez-preskocenia] [--istota-do-popisu] [--vlakien N]

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use analyzator_rs::pipeline::{self, Event, RunOptions};

/// Počet logických jadier stroja (auto-detekcia; záloha 4).
pub fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

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
    if args.iter().any(|a| a == "--aktualizacia") {
        run_update_check()?;
        std::process::exit(0);
    }
    let mut folder: Option<PathBuf> = None;
    let mut popisy = PathBuf::from("popisy.txt");
    let mut segments = 4usize;
    let mut min_istota = 50.0f64;
    let mut model_dir = PathBuf::from("models/clap_htsat_unfused_onnx");
    let mut skip_by_name = true;
    let mut istota_do_popisu = false;
    let mut vlakien = available_threads();
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
        vlakien: vlakien.clamp(1, 64),
        dump_mel,
        debug_audio,
        dump_emb,
        json_out,
    })
}

fn print_usage() {
    println!("Analyzátor zvukových súborov – Rust verzia {} (rýchly test výkonu)", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Použitie: analyzator-rs PRIECINOK [možnosti]");
    println!("  --popisy SUBOR        kandidátske popisy, jeden na riadok (default: popisy.txt)");
    println!("  --segments N          počet 10 s okien na súbor (default 4)");
    println!("  --min-istota P        popis pod P % sa nezapisuje (default 50)");
    println!("  --model-dir DIR       priečinok s ONNX modelmi (default models/clap_htsat_unfused_onnx)");
    println!("  --bez-preskocenia     vypne preskakovanie AI podľa názvu súboru");
    println!("  --istota-do-popisu    zapíše istotu do popisu (napr. ‘(istota 87 %)’)");
    println!("  --vlakien N           paralelné dekódovanie (default: všetky detekované jadrá = {})", available_threads());
    println!();
    println!("Grafická verzia: spustite analyzator-gui (okno ako Python verzia).");
}

/// Kontrola aktualizácie z príkazového riadku (rovnaká logika ako GUI).
fn run_update_check() -> Result<()> {
    let cur = env!("CARGO_PKG_VERSION");
    println!("Aktuálna verzia: {cur}");
    let rel = analyzator_rs::updater::latest_release()?;
    println!("Najnovšia na GitHube: {} ({})", rel.tag, rel.name);
    if !analyzator_rs::updater::is_newer(&rel.tag, cur) {
        println!("Máte najnovšiu verziu.");
        return Ok(());
    }
    println!("Nová verzia je k dispozícii – sťahujem {} ({:.1} MB)...",
             rel.url, rel.size as f64 / 1e6);
    let zip = std::path::PathBuf::from("_update.zip");
    analyzator_rs::updater::download_to(&rel.url, &zip, &|d, t| {
        if t > 0 && d % (5 * 1024 * 1024) < 256 * 1024 {
            println!("  {} / {} MB", d / 1_048_576, t / 1_048_576);
        }
    })?;
    let dest = std::path::PathBuf::from("_update_tmp");
    let inner = analyzator_rs::updater::extract_bundle(&zip, &dest)?;
    println!("Rozbalené do: {}", inner.display());
    println!("V GUI pokračujete tlačidlom Nainštalovať; tu iba testujeme.");
    Ok(())
}

fn load_descriptions_file(path: &std::path::Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("čítam {}", path.display()))?;
    let descs = analyzator_rs::parse_descriptions(&text);
    if descs.len() < 2 {
        anyhow::bail!(
            "v {} musia byť aspoň 2 popisy (jeden na riadok)",
            path.display()
        );
    }
    Ok(descs)
}

fn main() -> Result<()> {
    let opts = parse_args()?;
    let descriptions = load_descriptions_file(&opts.popisy)?;

    let mut files: Vec<PathBuf> = Vec::new();
    analyzator_rs::scan_audio(&opts.folder, &mut files);
    analyzator_rs::sort_natural(&mut files);
    if files.is_empty() {
        anyhow::bail!(
            "v {} nie sú žiadne zvukové súbory (wav/mp3/ogg/flac)",
            opts.folder.display()
        );
    }

    let run_opts = RunOptions {
        files,
        descriptions,
        segments: opts.segments,
        min_istota: opts.min_istota,
        model_dir: opts.model_dir,
        skip_by_name: opts.skip_by_name,
        istota_do_popisu: opts.istota_do_popisu,
        vlakien: opts.vlakien,
        dump_mel: opts.dump_mel,
        debug_audio: opts.debug_audio,
        dump_emb: opts.dump_emb,
        json_out: opts.json_out,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    // CLI vypisuje správy priamo do konzoly
    let res = pipeline::run(run_opts, &cancel, &|ev| match ev {
        Event::Info(s) => println!("{s}"),
        _ => {}
    });
    println!();
    res
}
