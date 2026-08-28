//! Analyzátor zvukových súborov – Rust verzia, GRAFICKÉ OKNO.
//! Rovnaká logika ako CLI (analyzator-rs), len s oknom ako Python GUI.
//!
//! Dizajn: jednotná tmavá paleta s akcentom, zaoblené prvky, prehľadná
//! tabuľka (egui_extras), sémantické farby stavov, veľké hlavné
//! tlačidlo, nápoveda pri prázdnom zozname a zvýraznenie pri
//! pretiahnutí súborov.
//!
//! Spustenie: dvojklik na analyzator-gui(.exe).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // bez čierneho okna

use eframe::egui;
use eframe::egui::{Align2, Color32, FontId, Rounding, Stroke};
use egui_extras::{Column, TableBuilder};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::Instant;

use analyzator_rs::pipeline::{self, Event, RunOptions};
use analyzator_rs::updater;
use analyzator_rs::{basename, is_audio_file, parse_descriptions, scan_audio, sort_natural};

/// Stav aktualizácie (GUI).
enum Upd {
    Idle,
    Checking,
    Latest,
    Avail {
        tag: String,
        url: String,
        size: u64,
        name: String,
    },
    Downloading {
        pct: f32,
    },
    Ready {
        inner: PathBuf,
        tag: String,
    },
    Err(String),
}

enum UpdMsg {
    State(Upd),
    Log(String),
}

/// Predvolených 69 popisov – rovnakých ako Python GUI.
const DEFAULT_POPISY: &str = include_str!("popisy_default.txt");

// ---- farebná paleta (tmavá „slate“ s modrým akcentom) -----------------------
mod pal {
    use eframe::egui::Color32;
    pub const BG: Color32 = Color32::from_rgb(0x16, 0x18, 0x1D); // panely
    pub const BG2: Color32 = Color32::from_rgb(0x1A, 0x1D, 0x23); // panely (sekc.)
    pub const INPUT: Color32 = Color32::from_rgb(0x0F, 0x11, 0x14); // vstupy
    pub const STRIPE: Color32 = Color32::from_rgb(0x1F, 0x23, 0x2A); // ryhy tabuľky
    pub const ACCENT: Color32 = Color32::from_rgb(0x53, 0xA8, 0xFF); // akcent
    pub const GO: Color32 = Color32::from_rgb(0x2F, 0xA4, 0x6A); // hlavné tlačidlo
    pub const STOP: Color32 = Color32::from_rgb(0xB0, 0x3A, 0x3A);
    pub const OK: Color32 = Color32::from_rgb(0x57, 0xC7, 0x76);
    pub const WARN: Color32 = Color32::from_rgb(0xE8, 0xA3, 0x3D);
    pub const ERR: Color32 = Color32::from_rgb(0xE5, 0x69, 0x6E);
    pub const SKIP: Color32 = Color32::from_rgb(0x7F, 0xB8, 0xFF);
    pub const TXT_WEAK: Color32 = Color32::from_rgb(0x9A, 0xA0, 0xAA);
}

fn install_style(ctx: &egui::Context) {
    let mut st = (*ctx.style()).clone();
    st.visuals = egui::Visuals::dark();
    st.visuals.panel_fill = pal::BG;
    st.visuals.extreme_bg_color = pal::INPUT;
    st.visuals.faint_bg_color = pal::STRIPE;
    st.visuals.window_fill = pal::BG2;
    st.visuals.selection.bg_fill = pal::ACCENT.linear_multiply(0.30);
    st.visuals.selection.stroke = Stroke::new(1.0_f32, pal::ACCENT);
    st.visuals.widgets.inactive.bg_fill = Color32::from_rgb(0x27, 0x2B, 0x33);
    st.visuals.widgets.inactive.fg_stroke.color = Color32::from_rgb(0xE2, 0xE6, 0xEC);
    st.visuals.widgets.hovered.bg_fill = Color32::from_rgb(0x33, 0x39, 0x44);
    st.visuals.widgets.hovered.fg_stroke.color = Color32::WHITE;
    st.visuals.widgets.active.bg_fill = Color32::from_rgb(0x3C, 0x43, 0x50);
    st.visuals.widgets.noninteractive.bg_fill = pal::BG2;
    st.visuals.widgets.noninteractive.fg_stroke.color = pal::TXT_WEAK;
    for w in [
        &mut st.visuals.widgets.noninteractive,
        &mut st.visuals.widgets.inactive,
        &mut st.visuals.widgets.hovered,
        &mut st.visuals.widgets.active,
        &mut st.visuals.widgets.open,
    ] {
        w.rounding = Rounding::same(6.0);
    }
    st.visuals.window_rounding = Rounding::same(10.0);
    st.visuals.menu_rounding = Rounding::same(8.0);
    st.spacing.button_padding = egui::vec2(12.0, 7.0);
    st.spacing.item_spacing = egui::vec2(8.0, 7.0);
    ctx.set_style(st);
}

#[derive(Clone, PartialEq)]
enum RowStatus {
    Pending,
    Done,
    Low,
    Err,
    NameSkip,
}

struct Row {
    status: RowStatus,
    name: String,
    desc: String,
    conf: Option<f64>,
    note: String,
}

struct App {
    max_vlakien: usize,
    path_text: String,
    files: Vec<PathBuf>,
    rows: Vec<Row>,
    desc_text: String,
    segments: usize,
    min_istota: f32,
    skip_by_name: bool,
    istota_do_popisu: bool,
    vlakien: usize,
    model_dir: PathBuf,
    model_ok: bool,
    log: String,
    running: bool,
    started: Option<Instant>,
    cancel: Arc<AtomicBool>,
    rx: Option<Receiver<Event>>,
    last_summary: String,
    upd: Upd,
    upd_rx: Option<Receiver<UpdMsg>>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_style(&cc.egui_ctx);

        // popisy: popisy.txt pri programe, inak vstavaný zoznam
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|x| x.to_path_buf()));
        let desc_text = exe_dir
            .as_ref()
            .and_then(|d| std::fs::read_to_string(d.join("popisy.txt")).ok())
            .unwrap_or_else(|| DEFAULT_POPISY.to_string());

        // model: hľadaj pri programe / o úroveň vyššie / v CWD
        let mut model_dir = PathBuf::from("models/clap_htsat_unfused_onnx");
        let mut model_ok = false;
        if let Some(ed) = &exe_dir {
            for cand in [
                ed.join("..").join("models").join("clap_htsat_unfused_onnx"),
                ed.join("models").join("clap_htsat_unfused_onnx"),
                ed.join("model"),
            ] {
                if cand.join("clap_audio.onnx").exists() {
                    model_dir = cand;
                    model_ok = true;
                    break;
                }
            }
        }
        if !model_ok && model_dir.join("clap_audio.onnx").exists() {
            model_ok = true;
        }

        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            max_vlakien: avail,
            path_text: String::new(),
            files: Vec::new(),
            rows: Vec::new(),
            desc_text,
            segments: 4,
            min_istota: 50.0,
            skip_by_name: true,
            istota_do_popisu: false,
            vlakien: avail,
            model_dir,
            model_ok,
            log: String::from("Analyzátor (Rust test) je pripravený.\n"),
            running: false,
            started: None,
            cancel: Arc::new(AtomicBool::new(false)),
            rx: None,
            last_summary: String::new(),
            upd: Upd::Idle,
            upd_rx: None,
        }
    }

    /// Na pozadí skontroluje novú verziu na GitHube (mlčky pri chybe).
    fn spawn_update_check(&mut self, ctx: &egui::Context, manual: bool) {
        if matches!(self.upd, Upd::Checking | Upd::Downloading { .. }) {
            return;
        }
        self.upd = Upd::Checking;
        let (tx, rx) = mpsc::channel::<UpdMsg>();
        let ctx2 = ctx.clone();
        self.upd_rx = Some(rx);
        std::thread::spawn(move || {
            let send = |st: Upd, log: Option<String>| {
                let _ = tx.send(UpdMsg::State(st));
                if let Some(l) = log {
                    let _ = tx.send(UpdMsg::Log(l));
                }
                ctx2.request_repaint();
            };
            match updater::latest_release() {
                Ok(rel) => {
                    if updater::is_newer(&rel.tag, env!("CARGO_PKG_VERSION")) {
                        send(
                            Upd::Avail {
                                tag: rel.tag.clone(),
                                url: rel.url,
                                size: rel.size,
                                name: rel.name.clone(),
                            },
                            Some(format!(
                                "🔄 Nová verzia {} je k dispozícii ({:.1} MB).",
                                rel.tag,
                                rel.size as f64 / 1e6
                            )),
                        );
                    } else if manual {
                        send(Upd::Latest, Some("Máte najnovšiu verziu.".into()));
                    } else {
                        send(Upd::Latest, None);
                    }
                }
                Err(e) => {
                    if manual {
                        send(Upd::Err(e.to_string()), Some(format!("Kontrola aktualizácií zlyhala: {e}")));
                    } else {
                        send(Upd::Idle, None); // tichý neúspech pri štarte
                    }
                }
            }
        });
    }

    /// Stiahne a rozbalí novú verziu na pozadí.
    fn spawn_update_download(&mut self, ctx: &egui::Context, url: String, size: u64, tag: String) {
        self.upd = Upd::Downloading { pct: 0.0 };
        let (tx, rx) = mpsc::channel::<UpdMsg>();
        let ctx2 = ctx.clone();
        self.upd_rx = Some(rx);
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|x| x.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        std::thread::spawn(move || {
            let send = |st: Upd, log: Option<String>| {
                let _ = tx.send(UpdMsg::State(st));
                if let Some(l) = log {
                    let _ = tx.send(UpdMsg::Log(l));
                }
                ctx2.request_repaint();
            };
            let zip = exe_dir.join("_update.zip");
            let last_pct = std::cell::Cell::new(-1.0f32);
            let res = updater::download_to(&url, &zip, &|done, total| {
                if total > 0 {
                    let pct = (done as f32 / total as f32 * 100.0).min(100.0);
                    if pct - last_pct.get() >= 5.0 {
                        last_pct.set(pct);
                        let _ = tx.send(UpdMsg::State(Upd::Downloading { pct }));
                        ctx2.request_repaint();
                    }
                }
            });
            if let Err(e) = res {
                send(Upd::Err(e.to_string()), Some(format!("Sťahovanie zlyhalo: {e}")));
                return;
            }
            let _ = tx.send(UpdMsg::Log(format!("Rozbaľujem {}...", tag)));
            ctx2.request_repaint();
            let dest = exe_dir.join("_update_tmp");
            let _ = std::fs::remove_dir_all(&dest);
            match updater::extract_bundle(&zip, &dest) {
                Ok(inner) => send(
                    Upd::Ready { inner, tag },
                    Some("✅ Nová verzia je stiahnutá – kliknite „Nainštalovať a reštartovať“.".into()),
                ),
                Err(e) => send(Upd::Err(e.to_string()), Some(format!("Rozbalenie zlyhalo: {e}"))),
            }
        });
    }

    fn rebuild_rows(&mut self) {
        self.rows = self
            .files
            .iter()
            .map(|f| Row {
                status: RowStatus::Pending,
                name: basename(f),
                desc: String::new(),
                conf: None,
                note: String::new(),
            })
            .collect();
    }

    fn load_folder(&mut self, dir: &PathBuf) {
        let mut files = Vec::new();
        scan_audio(dir, &mut files);
        sort_natural(&mut files);
        self.log.push_str(&format!(
            "Priečinok {}: nájdených {} zvukových súborov.\n",
            dir.display(),
            files.len()
        ));
        self.files = files;
        self.rebuild_rows();
    }

    fn add_paths(&mut self, paths: &[PathBuf]) {
        for p in paths {
            if p.is_dir() {
                scan_audio(p, &mut self.files);
            } else if is_audio_file(p) && !self.files.contains(p) {
                self.files.push(p.clone());
            }
        }
        sort_natural(&mut self.files);
        self.log.push_str(&format!(
            "Zoznam: {} zvukových súborov.\n",
            self.files.len()
        ));
        self.rebuild_rows();
    }

    fn apply_event(&mut self, ev: Event) {
        match ev {
            Event::Info(s) => {
                self.log.push_str(&s);
                self.log.push('\n');
                self.last_summary = s;
            }
            Event::NameSkip { idx, desc } => {
                if let Some(r) = self.rows.get_mut(idx) {
                    r.status = RowStatus::NameSkip;
                    r.desc = desc;
                }
            }
            Event::FileDone {
                idx,
                text,
                conf,
                notes,
            } => {
                if let Some(r) = self.rows.get_mut(idx) {
                    r.status = RowStatus::Done;
                    r.desc = text;
                    r.conf = Some(conf);
                    r.note = notes;
                }
            }
            Event::FileLow {
                idx,
                tip,
                conf,
                detail,
            } => {
                if let Some(r) = self.rows.get_mut(idx) {
                    r.status = RowStatus::Low;
                    r.desc = format!("(tip: {tip})");
                    r.conf = Some(conf);
                    r.note = detail;
                }
            }
            Event::FileErr { idx, msg } => {
                if let Some(r) = self.rows.get_mut(idx) {
                    r.status = RowStatus::Err;
                    r.note = msg;
                }
            }
            Event::Finished { .. } => {
                self.running = false;
                self.started = None;
            }
        }
    }

    fn start_run(&mut self, ctx: &egui::Context) {
        let descriptions = parse_descriptions(&self.desc_text);
        if descriptions.len() < 2 {
            self.log
                .push_str("✖ Zadajte aspoň dva kandidátske popisy (jeden na riadok).\n");
            return;
        }
        if self.files.is_empty() {
            self.log
                .push_str("✖ Najprv vyberte priečinok alebo potiahnite zvukové súbory do okna.\n");
            return;
        }
        if !self.model_ok {
            self.log.push_str(
                "✖ Nenašiel som AI model – nakopírujte tento priečinok do hlavného \
                 priečinka programu (k SPUSTI.bat a priečinku models).\n",
            );
            return;
        }

        self.cancel.store(false, Ordering::Relaxed);
        self.running = true;
        self.started = Some(Instant::now());
        self.last_summary.clear();
        self.rebuild_rows();
        self.log.push_str("▶ Spúšťam analýzu…\n");

        let opts = RunOptions {
            files: self.files.clone(),
            descriptions,
            segments: self.segments,
            min_istota: (self.min_istota.max(0.0) / 100.0) as f64,
            model_dir: self.model_dir.clone(),
            skip_by_name: self.skip_by_name,
            istota_do_popisu: self.istota_do_popisu,
            vlakien: self.vlakien,
            dump_mel: None,
            debug_audio: false,
            dump_emb: None,
            json_out: None,
        };
        let (tx, rx) = mpsc::channel::<Event>();
        let cancel = self.cancel.clone();
        let ctx2 = ctx.clone();
        self.rx = Some(rx);
        std::thread::spawn(move || {
            let ctx3 = ctx2.clone();
            let tx2 = tx.clone();
            let res = pipeline::run(opts, &cancel, &move |ev| {
                if tx2.send(ev).is_ok() {
                    ctx3.request_repaint();
                }
            });
            if let Err(e) = res {
                let _ = tx.send(Event::Info(format!("✖ Chyba behu: {e}")));
                ctx2.request_repaint();
            }
        });
    }

    fn status_style(r: &Row) -> (&'static str, Color32) {
        match r.status {
            RowStatus::Pending => ("●", pal::TXT_WEAK),
            RowStatus::Done => ("●", pal::OK),
            RowStatus::Low => ("●", pal::WARN),
            RowStatus::Err => ("●", pal::ERR),
            RowStatus::NameSkip => ("⚡", pal::SKIP),
        }
    }
}

impl App {
    /// Aplikuje stiahnutú aktualizáciu: bat počka na ukončenie,
    /// prekopíruje súbory a spustí novú verziu.
    fn install_update(&mut self) {
        let inner = match &self.upd {
            Upd::Ready { inner, .. } => inner.clone(),
            _ => return,
        };
        self.log
            .push_str("↻ Inštalujem aktualizáciu a reštartujem…\n");
        #[cfg(windows)]
        {
            match updater::install_and_restart(&inner, std::process::id()) {
                Ok(_) => std::process::exit(0),
                Err(e) => self.log.push_str(&format!("✖ Instalácia zlyhala: {e}\n")),
            }
        }
        #[cfg(not(windows))]
        {
            let _ = inner;
            self.log.push_str(
                "Na tomto systéme rozbaľte obsah priečinka _update_tmp/analyzator-rs-windows ručne.\n",
            );
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 1) prijaté správy z pracovného vlákna
        if let Some(rx) = self.rx.take() {
            loop {
                match rx.try_recv() {
                    Ok(ev) => self.apply_event(ev),
                    Err(_) => break,
                }
            }
            self.rx = Some(rx);
        }

        // 1b) správy o aktualizácii
        if let Some(rx) = self.upd_rx.take() {
            loop {
                match rx.try_recv() {
                    Ok(UpdMsg::State(st)) => self.upd = st,
                    Ok(UpdMsg::Log(l)) => {
                        self.log.push_str(&l);
                        self.log.push('\n');
                    }
                    Err(_) => break,
                }
            }
            self.upd_rx = Some(rx);
        }

        // 2) potiahnuté súbory/priečinky myšou
        if !self.running {
            let dropped: Vec<PathBuf> = ctx.input(|i| {
                i.raw
                    .dropped_files
                    .iter()
                    .filter_map(|f| f.path.clone())
                    .collect()
            });
            if !dropped.is_empty() {
                self.add_paths(&dropped);
                if let Some(first) = self.files.first() {
                    if let Some(dir) = first.parent() {
                        self.path_text = dir.display().to_string();
                    }
                }
            }
        }

        // Ctrl+Enter = Analyzovať
        if !self.running
            && !self.files.is_empty()
            && ctx.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl)
        {
            self.start_run(ctx);
        }

        // kontrola aktualizácií raz pri štarte (mlčky)
        if matches!(self.upd, Upd::Idle) {
            self.spawn_update_check(ctx, false);
        }

        let dragging = ctx.input(|i| !i.raw.hovered_files.is_empty());

        // 3) hlavička: titulok + cesta
        egui::TopBottomPanel::top("hlavicka")
            .frame(
                egui::Frame::default()
                    .fill(pal::BG2)
                    .inner_margin(egui::Margin::symmetric(14.0, 10.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("🎧 Analyzátor zvukových súborov")
                            .size(17.0)
                            .strong(),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        match &self.upd {
                            Upd::Avail { tag, .. } => {
                                let t = tag.clone();
                                if ui
                                    .add(
                                        egui::Button::new(
                                            egui::RichText::new(format!("🔄 Aktualizovať na {t}"))
                                                .small()
                                                .strong(),
                                        )
                                        .fill(pal::ACCENT.linear_multiply(0.25)),
                                    )
                                    .clicked()
                                {
                                    if let Upd::Avail { url, size, tag, .. } =
                                        std::mem::replace(&mut self.upd, Upd::Idle)
                                    {
                                        self.spawn_update_download(ctx, url, size, tag);
                                    }
                                }
                            }
                            Upd::Downloading { pct } => {
                                ui.label(
                                    egui::RichText::new(format!("⬇ sťahujem {pct:.0} %"))
                                        .small()
                                        .color(pal::ACCENT),
                                );
                            }
                            Upd::Ready { tag, .. } => {
                                let t = tag.clone();
                                if ui
                                    .add(
                                        egui::Button::new(
                                            egui::RichText::new("↻ Nainštalovať a reštartovať")
                                                .small()
                                                .strong(),
                                        )
                                        .fill(pal::GO),
                                    )
                                    .clicked()
                                {
                                    self.install_update();
                                }
                                ui.label(egui::RichText::new(format!("✅ {t} pripravená")).small());
                            }
                            Upd::Err(e) => {
                                ui.label(
                                    egui::RichText::new(format!("⚠ aktualizácia: {e}"))
                                        .small()
                                        .color(pal::WARN),
                                );
                            }
                            _ => {}
                        }
                        ui.label(
                            egui::RichText::new(format!(
                                "v{} · Rust test · rovnaké výsledky ako Python",
                                env!("CARGO_PKG_VERSION")
                            ))
                            .small()
                            .color(pal::TXT_WEAK),
                        );
                    });
                });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        !self.running,
                        egui::TextEdit::singleline(&mut self.path_text)
                            .hint_text("C:\\Users\\TvojeMeno\\Desktop\\Zvuky")
                            .desired_width(ui.available_width() - 330.0),
                    );
                    if ui
                        .add_enabled(
                            !self.running,
                            egui::Button::new("📂 Vybrať priečinok…"),
                        )
                        .clicked()
                    {
                        if let Some(dir) = rfd::FileDialog::new()
                            .set_title("Vybrať priečinok so zvukovými súbormi")
                            .pick_folder()
                        {
                            self.path_text = dir.display().to_string();
                        }
                    }
                    let want_load = ui
                        .add_enabled(!self.running, egui::Button::new("Načítať ▸"))
                        .clicked();
                    if (want_load || ui.input(|i| i.key_pressed(egui::Key::Enter)))
                        && !self.running
                        && !self.path_text.trim().is_empty()
                    {
                        let dir = PathBuf::from(self.path_text.trim());
                        if dir.is_dir() {
                            self.load_folder(&dir);
                        } else {
                            self.log.push_str(&format!(
                                "✖ Priečinok neexistuje: {}\n",
                                dir.display()
                            ));
                        }
                    }
                    if !self.running && !self.files.is_empty() {
                        if ui.button("✕ Zoznam").clicked() {
                            self.files.clear();
                            self.rebuild_rows();
                            self.log.push_str("Zoznam súborov vymazaný.\n");
                        }
                    }
                });
            });

        // 4) spodný panel: tlačidlá + priebeh + log
        egui::TopBottomPanel::bottom("ovladanie")
            .frame(
                egui::Frame::default()
                    .fill(pal::BG2)
                    .inner_margin(egui::Margin::symmetric(14.0, 10.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let can_start = !self.running
                        && !self.files.is_empty()
                        && parse_descriptions(&self.desc_text).len() >= 2;
                    let start_btn = egui::Button::new(
                        egui::RichText::new(format!("▶  Analyzovať  ({})", self.files.len()))
                            .size(16.0)
                            .strong(),
                    )
                    .min_size(egui::vec2(210.0, 38.0))
                    .fill(pal::GO);
                    if ui.add_enabled(can_start, start_btn).clicked() {
                        self.start_run(ctx);
                    }
                    if self.running {
                        let stop_btn = egui::Button::new(
                            egui::RichText::new("■  Zastaviť").size(15.0),
                        )
                        .min_size(egui::vec2(120.0, 38.0))
                        .fill(pal::STOP);
                        if ui.add(stop_btn).clicked() {
                            self.cancel.store(true, Ordering::Relaxed);
                            self.log
                                .push_str("Zastavujem… (dokončí sa aktuálny súbor)\n");
                        }
                    }

                    let total = self.rows.len();
                    let done = self
                        .rows
                        .iter()
                        .filter(|r| r.status != RowStatus::Pending)
                        .count();
                    let frac = if total > 0 {
                        done as f32 / total as f32
                    } else {
                        0.0
                    };
                    let mut bar_text = if total > 0 {
                        format!("{done} / {total}")
                    } else {
                        "—".into()
                    };
                    if self.running && done > 0 {
                        if let Some(t) = self.started {
                            let eta = t.elapsed().as_secs_f32() / done as f32
                                * (total - done) as f32;
                            bar_text = format!("{bar_text} · ostáva ~{eta:.0} s");
                        }
                    }
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .desired_width(ui.available_width() - 40.0)
                            .fill(pal::GO)
                            .text(bar_text),
                    );
                });
                ui.add_space(2.0);
                if !self.last_summary.is_empty() && !self.running {
                    ui.label(
                        egui::RichText::new(&self.last_summary)
                            .small()
                            .color(pal::TXT_WEAK),
                    );
                }
                ui.add_space(4.0);
                egui::CollapsingHeader::new(
                    egui::RichText::new("📜 Log").small().color(pal::TXT_WEAK),
                )
                .default_open(true)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .max_height(150.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut self.log.as_str())
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(f32::INFINITY)
                                    .interactive(false),
                            );
                        });
                });
            });

        // 5) pravý panel: popisy + nastavenia
        egui::SidePanel::right("nastavenia")
            .resizable(true)
            .default_width(360.0)
            .frame(
                egui::Frame::default()
                    .fill(pal::BG2)
                    .inner_margin(egui::Margin::same(12.0)),
            )
            .show(ctx, |ui| {
                ui.add_space(2.0);
                let n_popisov = parse_descriptions(&self.desc_text).len();
                egui::CollapsingHeader::new(
                    egui::RichText::new(format!("📝 Kandidátske popisy ({n_popisov})")).strong(),
                )
                .default_open(true)
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("jeden na riadok · ‘#’ = sekcia").small().color(pal::TXT_WEAK),
                    );
                    ui.add(
                        egui::TextEdit::multiline(&mut self.desc_text)
                            .desired_rows(12)
                            .font(egui::TextStyle::Small),
                    );
                });
                ui.add_space(4.0);
                egui::CollapsingHeader::new(egui::RichText::new("⚙ Nastavenia analýzy").strong())
                    .default_open(true)
                    .show(ui, |ui| {
                        let seg_txt = format!("okien (10 s) na súbor: {}", self.segments);
                        ui.add(egui::Slider::new(&mut self.segments, 1..=8).text(seg_txt))
                        .on_hover_text("Dlhšie nahrávky sa analyzujú vo viacerich 10-sekundových oknách; ich výsledky sa spriemerujú.");
                        let mi_txt = format!("prah istoty: {} %", self.min_istota as i32);
                        ui.add(
                            egui::Slider::new(&mut self.min_istota, 0.0..=95.0)
                                .step_by(1.0)
                                .text(mi_txt),
                        )
                        .on_hover_text("Popis s istotou pod touto hodnotou sa do súboru nezapíše (a starý náš popis sa zmaže).");
                        ui.checkbox(
                            &mut self.skip_by_name,
                            "AI preskočiť pri jednoznačnom názve",
                        )
                        .on_hover_text("Keď slová z názvu priamo a jednoznačne sedia na jediný popis, AI sa nepoužije.");
                        ui.checkbox(
                            &mut self.istota_do_popisu,
                            "zapísať istotu do popisu (napr. 87 %)",
                        );
                    });
                ui.add_space(4.0);
                egui::CollapsingHeader::new(egui::RichText::new("⚡ Výkon").strong())
                    .default_open(true)
                    .show(ui, |ui| {
                        let vl_txt = format!("vlákien dekódu: {}", self.vlakien);
                        ui.add(
                            egui::Slider::new(&mut self.vlakien, 1..=self.max_vlakien.max(4))
                                .text(vl_txt),
                        )
                        .on_hover_text("Dekódovanie a predspracovanie zvuku beží paralelne; AI beží rovnako.");
                        ui.label(
                            egui::RichText::new(format!(
                                "Počet jadier nájdených v počítači: {} (prednastavené automaticky)",
                                self.max_vlakien
                            ))
                            .small()
                            .color(pal::TXT_WEAK),
                        );
                    });
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.separator();
                    ui.horizontal(|ui| {
                        let checking = matches!(self.upd, Upd::Checking | Upd::Downloading { .. });
                        if ui
                            .add_enabled(!checking, egui::Button::new("🔄 Skontrolovať aktualizácie"))
                            .on_hover_text("Zistí, či je na GitHube novšia verzia programu, a ponúkne jej stiahnutie.")
                            .clicked()
                        {
                            self.spawn_update_check(ctx, true);
                        }
                        if matches!(self.upd, Upd::Checking) {
                            ui.label(egui::RichText::new("kontrolujem…").small().color(pal::TXT_WEAK));
                        } else if matches!(self.upd, Upd::Latest) {
                            ui.label(egui::RichText::new("máte najnovšiu").small().color(pal::OK));
                        }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if self.model_ok {
                            ui.label(
                                egui::RichText::new("●").color(pal::OK),
                            );
                            ui.label(egui::RichText::new("AI model nájdený").strong());
                        } else {
                            ui.label(egui::RichText::new("●").color(pal::ERR));
                            ui.label(
                                egui::RichText::new("AI model nenájdený!").strong(),
                            );
                        }
                    });
                    if self.model_ok {
                        ui.label(
                            egui::RichText::new(self.model_dir.display().to_string())
                                .small()
                                .color(pal::TXT_WEAK),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(
                                "Nakopírujte tento priečinok do hlavného priečinka programu \
                                 (k SPUSTI.bat, tam kde je priečinok „models“) a spustite znovu.",
                            )
                            .small()
                            .color(pal::TXT_WEAK),
                        );
                    }
                });
            });

        // 6) hlavný panel: tabuľka súborov
        egui::CentralPanel::default()
            .frame(egui::Frame::default().inner_margin(egui::Margin::same(12.0)))
            .show(ctx, |ui| {
                if self.files.is_empty() {
                    ui.centered_and_justified(|ui| {
                        ui.vertical(|ui| {
                            ui.label(
                                egui::RichText::new("📂").size(42.0),
                            );
                            ui.add_space(8.0);
                            ui.label(
                                egui::RichText::new("Žiadne zvukové súbory").size(19.0).strong(),
                            );
                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new(
                                    "Potiahnite sem priečinok so zvukmi (wav/mp3/ogg/flac)\nalebo použite „📂 Vybrať priečinok…“ hore.",
                                )
                                .color(pal::TXT_WEAK),
                            );
                        });
                    });
                } else {
                    let table = TableBuilder::new(ui)
                        .striped(true)
                        .resizable(true)
                        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                        .column(Column::exact(36.0))
                        .column(Column::initial(300.0).at_least(140.0).clip(true))
                        .column(Column::remainder().clip(true))
                        .column(Column::exact(72.0))
                        .header(26.0, |mut h| {
                            h.col(|_| {});
                            h.col(|ui| {
                                ui.label(egui::RichText::new("SÚBOR").small().color(pal::TXT_WEAK));
                            });
                            h.col(|ui| {
                                ui.label(egui::RichText::new("POPIS").small().color(pal::TXT_WEAK));
                            });
                            h.col(|ui| {
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            egui::RichText::new("ISTOTA")
                                                .small()
                                                .color(pal::TXT_WEAK),
                                        );
                                    },
                                );
                            });
                        });
                    table.body(|mut body| {
                        for r in &self.rows {
                            let (dot, color) = Self::status_style(r);
                            let h = if r.note.is_empty() { 30.0 } else { 44.0 };
                            body.row(h, |mut row| {
                                row.col(|ui| {
                                    ui.label(
                                        egui::RichText::new(dot).color(color).size(13.0),
                                    );
                                });
                                row.col(|ui| {
                                    ui.label(egui::RichText::new(&r.name).strong());
                                });
                                row.col(|ui| {
                                    let inner = ui.vertical(|ui| {
                                        ui.label(&r.desc);
                                        if !r.note.is_empty() {
                                            ui.label(
                                                egui::RichText::new(&r.note)
                                                    .small()
                                                    .color(pal::TXT_WEAK),
                                            );
                                        }
                                    });
                                    inner.response.on_hover_text(format!("{}\n{}", r.name, r.note));
                                });
                                row.col(|ui| {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if let Some(c) = r.conf {
                                                ui.label(
                                                    egui::RichText::new(format!(
                                                        "{:.0} %",
                                                        c * 100.0
                                                    ))
                                                    .strong()
                                                    .color(color),
                                                );
                                            }
                                        },
                                    );
                                });
                            });
                        }
                    });
                }

                // 7) zvýraznenie pri pretiahnutí súborov
                if dragging {
                    let rect = ui.max_rect();
                    ui.painter().rect_filled(
                        rect,
                        Rounding::same(12.0),
                        pal::ACCENT.linear_multiply(0.07),
                    );
                    ui.painter().rect_stroke(
                        rect.shrink(2.0),
                        Rounding::same(12.0),
                        Stroke::new(2.0_f32, pal::ACCENT),
                    );
                    ui.painter().text(
                        rect.center(),
                        Align2::CENTER_CENTER,
                        "↧  Pusťte súbory alebo priečinok sem",
                        FontId::proportional(20.0),
                        pal::ACCENT,
                    );
                }
            });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 780.0])
            .with_min_inner_size([900.0, 580.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Analyzátor zvukových súborov (Rust test)",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
