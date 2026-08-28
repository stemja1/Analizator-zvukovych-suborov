//! Analyzátor zvukových súborov – Rust verzia, GRAFICKÉ OKNO.
//! Rovnaká logika ako CLI (analyzator-rs), len s oknom ako Python GUI.
//!
//! Spustenie: dvojklik na analyzator-gui(.exe).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // bez čierneho okna

use eframe::egui;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use analyzator_rs::pipeline::{self, Event, RunOptions};
use analyzator_rs::{basename, is_audio_file, parse_descriptions, scan_audio, sort_natural};

/// Predvolených 69 popisov – rovnakých ako Python GUI.
const DEFAULT_POPISY: &str = include_str!("popisy_default.txt");

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
    cancel: Arc<AtomicBool>,
    rx: Option<Receiver<Event>>,
    last_summary: String,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

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
            log: String::from("Analyzátor (Rust test) spustený.\n"),
            running: false,
            cancel: Arc::new(AtomicBool::new(false)),
            rx: None,
            last_summary: String::new(),
        }
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
        self.log
            .push_str(&format!("Priečinok {}: nájdených {} zvukových súborov.\n", dir.display(), files.len()));
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
        self.log.push_str(&format!("Zoznam: {} zvukových súborov.\n", self.files.len()));
        self.rebuild_rows();
    }

    fn apply_event(&mut self, ev: Event) {
        match ev {
            Event::Info(s) => {
                self.log.push_str(&s);
                self.log.push('\n');
                self.last_summary = s; // posledná správa = súhrn behu
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
        self.last_summary.clear();
        self.rebuild_rows();
        self.log
            .push_str("▶ Spúšťam analýzu…\n");

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

        // 3) spodný panel: tlačidlá + priebeh + log
        egui::TopBottomPanel::bottom("ovladanie")
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let can_start = !self.running
                        && !self.files.is_empty()
                        && parse_descriptions(&self.desc_text).len() >= 2;
                    if ui
                        .add_enabled(can_start, egui::Button::new("▶  Analyzovať"))
                        .clicked()
                    {
                        self.start_run(ctx);
                    }
                    if self.running
                        && ui.button("■  Zastaviť").clicked()
                    {
                        self.cancel.store(true, Ordering::Relaxed);
                        self.log.push_str("Zastavujem… (dokončí sa aktuálny súbor)\n");
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
                    let txt = if self.running || done > 0 {
                        format!("{done}/{total}")
                    } else {
                        "—".into()
                    };
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .show_percentage()
                            .text(txt),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak(if self.running {
                            "beží…".to_string()
                        } else {
                            self.last_summary.clone()
                        });
                    });
                });
                ui.add_space(4.0);
                ui.add(
                    egui::TextEdit::multiline(&mut self.log)
                        .font(egui::TextStyle::Monospace)
                        .desired_rows(8)
                        .interactive(false),
                );
            });

        // 4) pravý panel: popisy + nastavenia
        egui::SidePanel::right("nastavenia")
            .resizable(false)
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.heading("Kandidátske popisy");
                ui.label(egui::RichText::new("jeden na riadok, ‘#’ = sekcia").weak());
                ui.add(
                    egui::TextEdit::multiline(&mut self.desc_text)
                        .desired_rows(16)
                        .font(egui::TextStyle::Small),
                );
                ui.add_space(8.0);
                ui.separator();
                ui.add(
                    egui::Slider::new(&mut self.segments, 1..=8)
                        .text("okien (10 s) na súbor"),
                );
                ui.add(
                    egui::Slider::new(&mut self.min_istota, 0.0..=95.0)
                        .step_by(1.0)
                        .text("prah istoty %"),
                );
                ui.add(
                    egui::Slider::new(&mut self.vlakien, 1..=self.max_vlakien.max(4))
                        .text(format!("vlákien dekódu (auto: {})", self.max_vlakien)),
                );
                ui.label(
                    egui::RichText::new(format!("Detekovaných logických jadier: {}", self.max_vlakien))
                        .small()
                        .weak(),
                );
                ui.checkbox(
                    &mut self.skip_by_name,
                    "AI preskočiť pri jednoznačnom názve",
                );
                ui.checkbox(
                    &mut self.istota_do_popisu,
                    "zapísať istotu do popisu (napr. 87 %)",
                );
                ui.add_space(8.0);
                ui.separator();
                if self.model_ok {
                    ui.label(
                        egui::RichText::new("✔ AI model nájdený").color(egui::Color32::from_rgb(90, 200, 110)),
                    );
                    ui.weak(self.model_dir.display().to_string());
                } else {
                    ui.label(
                        egui::RichText::new("✖ AI model nenájdený – nakopírujte tento priečinok k SPUSTI.bat (k priečinku models)")
                            .color(egui::Color32::from_rgb(230, 90, 90)),
                    );
                }
            });

        // 5) hlavný panel: súbory
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let edit = egui::TextEdit::singleline(&mut self.path_text)
                    .hint_text("C:\\Users\\TvojeMeno\\Desktop\\Zvuky")
                    .desired_width(ui.available_width() - 210.0);
                ui.add(edit);
                if ui.button("Vybrať priečinok…").clicked() {
                    if let Some(dir) = rfd::FileDialog::new()
                        .set_title("Vybrať priečinok so zvukovými súbormi")
                        .pick_folder()
                    {
                        self.path_text = dir.display().to_string();
                    }
                }
                let load_clicked = ui.button("Načítať").clicked();
                if (load_clicked || ui.input(|i| i.key_pressed(egui::Key::Enter)))
                    && !self.path_text.trim().is_empty()
                {
                    let dir = PathBuf::from(self.path_text.trim());
                    if dir.is_dir() {
                        self.load_folder(&dir);
                    } else {
                        self.log
                            .push_str(&format!("✖ Priečinok neexistuje: {}\n", dir.display()));
                    }
                }
            });
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new("Tip: priečinok alebo súbory môžete potiahnuť myskou priamo do okna.")
                    .weak(),
            );
            ui.add_space(4.0);
            ui.separator();

            let n = self.files.len();
            ui.heading(if n == 0 {
                "Žiadne súbory – vyberte priečinok".to_string()
            } else {
                format!("Zvukové súbory ({n})")
            });
            ui.add_space(4.0);

            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                egui::Grid::new("subory")
                    .striped(true)
                    .num_columns(4)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        for r in &self.rows {
                            let (icon, color) = match r.status {
                                RowStatus::Pending => ("…", egui::Color32::GRAY),
                                RowStatus::Done => ("✔", egui::Color32::from_rgb(90, 200, 110)),
                                RowStatus::Low => ("⚠", egui::Color32::from_rgb(224, 138, 0)),
                                RowStatus::Err => ("✖", egui::Color32::from_rgb(230, 90, 90)),
                                RowStatus::NameSkip => ("⚡", egui::Color32::from_rgb(120, 170, 255)),
                            };
                            ui.label(egui::RichText::new(icon).color(color).strong());
                            ui.label(egui::RichText::new(&r.name).strong());
                            ui.label(&r.desc);
                            ui.label(
                                egui::RichText::new(match r.conf {
                                    Some(c) => format!("{:.0} %", c * 100.0),
                                    None => String::new(),
                                })
                                .color(color),
                            );
                            ui.end_row();
                            if !r.note.is_empty() {
                                ui.label("");
                                ui.label(
                                    egui::RichText::new(&r.note).small().weak(),
                                )
                                .on_hover_text(&r.note);
                                ui.label("");
                                ui.label("");
                                ui.end_row();
                            }
                        }
                    });
            });
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1150.0, 750.0])
            .with_min_inner_size([880.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Analyzátor zvukových súborov (Rust test)",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
