"""
main_gui.py
===========
PyQt6 rozhranie aplikácie „Analyzátor zvukových súborov".

* Výber súborov alebo celého priečinka (WAV / MP3 / OGG / FLAC) + drag&drop
* Tabuľka so stavom každého súboru (Čaká / Spracováva sa / Hotovo / Chyba)
* QTextEdit so zoznamom kandidátskych popisov (jeden na riadok)
* Progress bar + spracovanie na pozadí (QThread), GUI nikdy nezamrzne
* AI beží cez core_analyzer.ClapAnalyzer (ONNX Runtime / DirectML)

Spustenie:  python main_gui.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import traceback

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from core_analyzer import (DEFAULT_SEGMENTS, SUPPORTED_EXTENSIONS, ClapAnalyzer,
                           write_metadata)

# --- stavy súborov v tabuľke -----------------------------------------------
ST_WAITING = "Čaká"
ST_RUNNING = "Spracováva sa…"
ST_DONE = "Hotovo"
ST_ERROR = "Chyba"
ST_SKIPPED = "Preskočené"

ST_COLORS = {
    ST_WAITING: QColor("#8a8a8a"),
    ST_RUNNING: QColor("#e08a00"),
    ST_DONE: QColor("#2e9e44"),
    ST_ERROR: QColor("#d03a3a"),
    ST_SKIPPED: QColor("#8a8a8a"),
}

FILE_FILTER = "Zvukové súbory (*.wav *.mp3 *.ogg *.flac);;Všetky súbory (*.*);;WAV (*.wav);;MP3 (*.mp3);;OGG (*.ogg);;FLAC (*.flac)"

DEFAULT_DESCRIPTIONS = "\n".join([
    # tranzície / whoosh
    "a fast whoosh sound effect",
    "a deep cinematic whoosh",
    "a metal whoosh transition sound",
    "a swipe transition sound effect",
    "an air whoosh passing by",
    # impacty / hity
    "a heavy cinematic impact hit",
    "a deep bass impact hit",
    "a dramatic braam impact sound",
    "a punch hit sound effect",
    "a metal clang impact sound",
    # risery / napätie
    "a rising tension riser sound",
    "a suspenseful build-up riser",
    "a horror tension sting sound",
    # sub / bass
    "a deep sub bass drop",
    "a low frequency bass rumble",
    # glitch / digitálne
    "a digital glitch sound effect",
    "a glitchy stutter sound",
    "a data error glitch beep",
    "an electronic malfunction sound",
    # kamera / foto
    "a camera shutter click",
    "a camera flash sound",
    "an old film camera shutter sound",
    # UI / tech / interface
    "a computer mouse click sound",
    "a mechanical keyboard typing sound",
    "a keyboard key press sound",
    "a notification beep sound",
    "a user interface click sound",
    "a phone ringtone sound",
    "a radio tuning static sound",
    "a radio dial adjustment sound",
    "a robotic beeping sound",
    "a sci-fi interface sound effect",
    "a spaceship engine hum",
    # výbuchy / zbrane
    "an explosion sound effect",
    "a gunshot sound effect",
    "a sword clashing sound",
    "an arrow whoosh sound",
    "a laser blast sound effect",
    "fireworks exploding",
    # vozidlá
    "car engine revving",
    "a car door closing",
    "car tires screeching",
    "an airplane flying overhead",
    "a helicopter blade whoosh",
    "a train passing by",
    # foley / mechanika
    "footsteps on a wood floor",
    "footsteps on gravel",
    "a door creaking open",
    "glass breaking sound",
    "metal chains rattling",
    "a clock ticking sound",
    "an alarm siren sound",
    "cloth rustling sound",
    "a paper page turning sound",
    # príroda / počasie / ambient
    "heavy rain falling on roof",
    "thunder rumbling in the distance",
    "wind blowing strongly",
    "fire crackling sound",
    "water splashing sound",
    "waves crashing on the shore",
    "birds singing in the forest",
    # ľudia / dav (bez reči)
    "people talking in a room",
    "a crowd cheering and applauding",
    "children laughing",
    "a heartbeat sound effect",
    # zvieratá
    "dog barking outdoors",
    "a cat meowing",
    # mágia / fantasy / sci-fi
    "a magic spell casting sound",
    "an energy charging sound effect",
])

# --- ďalšie predvoľby (combo v GUI); '#' riadky = sekcie, do analýzy nezahrňujeme ---
PRESET_QUICK = """whistle sound in midnight
heavy rain falling on roof
car engine revving
dog barking outdoors
birds singing in the forest
people talking in a room"""

PRESET_SFX_BIG = """# ===== POČASIE A PRÍRODA =====
heavy rain pouring on a roof
rain drops on a window
distant thunder and rainstorm
strong wind howling
gentle breeze in leaves
birds singing in a forest
crickets and night insects
ocean waves on a beach
flowing river stream
waterfall
campfire crackling
walking on snow
waves crashing on the shore
water splashing sound
fire crackling sound

# ===== MESTO A DOPRAVA =====
busy city street with traffic
car engine revving
car passing by at speed
car tires screeching
car horn honking
a car door closing
police siren
ambulance siren
an alarm siren sound
subway train arriving
a train passing by
train rolling on tracks
an airplane flying overhead
a helicopter blade whoosh
motorcycle passing by
tram bell ringing
ship horn

# ===== ĽUDIA A DOMÁCNOSŤ =====
crowd cheering and applause
a crowd cheering and applauding
children playing on a playground
children laughing
restaurant crowd chatter
people whispering
people talking in a room
man laughing
woman screaming
footsteps on gravel
footsteps on a wood floor
footsteps on creaky wood floor
door slamming
a door creaking open
knocking on a door
typing on a keyboard
a mechanical keyboard typing sound
writing on paper
a paper page turning sound
dishes and cutlery clattering
glass breaking on the floor
glass breaking sound
pouring water into a glass
shower running
clock ticking
a clock ticking sound
an old telephone ringing
a heartbeat sound effect

# ===== FILMOVÉ EFEKTY A PRECHODY =====
a fast whoosh sound effect
a deep cinematic whoosh
whoosh transition effect
a swipe transition sound effect
an air whoosh passing by
a metal whoosh transition sound
a rising tension riser sound
a suspenseful build-up riser
a horror tension sting sound
a dramatic braam impact sound
a heavy cinematic impact hit
a deep bass impact hit
a deep sub bass drop
a low frequency bass rumble
a punch hit sound effect
a metal clang impact sound
metal chains rattling
sword unsheathing
a sword clashing sound
an arrow whoosh sound
an explosion sound effect
fireworks exploding
a magic spell casting sound
an energy charging sound effect
magic sparkle glitter
shimmering time freeze effect
reverse cymbal swell
tape stop effect
vinyl scratch
a digital glitch sound effect
a glitchy stutter sound
an electronic malfunction sound
static noise
vinyl crackle
cloth rustling sound

# ===== SCI-FI A TECHNOLÓGIE =====
a spaceship engine hum
a laser blast sound effect
lightsaber hum
robot servo motors moving
a robotic beeping sound
computer beeps
a sci-fi interface sound effect
futuristic interface clicks
sci-fi door hissing open
electric zap buzzing
energy power up
power down shutdown
sci-fi alarm
sonar ping
a radio tuning static sound
a data error glitch beep
a notification beep sound
a user interface click sound
a phone ringtone sound
a camera shutter click
a camera flash sound
an old film camera shutter sound
film projector
typewriter
cash register
coin dropping

# ===== ZVIERATÁ =====
dog barking outdoors
dog growling
a cat meowing
cat purring
horse galloping
cow mooing
sheep bleating
chicken clucking
rooster crowing
crow cawing
seagulls calling
owl hooting
wolf howling
lion roar
elephant trumpeting
snake hissing
bees buzzing
frog croaking
whale song

# ===== HUDBA A NÁSTROJE =====
solo piano melody
acoustic guitar strumming
electric guitar riff
bass guitar groove
funky drum kit groove
snare drum roll
cymbal crash
jazz saxophone melody
trumpet fanfare
violin emotional melody
cello swell
harp glissando
angelic choir singing
female vocal humming
orchestral epic swell
epic strings ostinato

# ===== AMBIENT A PROSTREDIA =====
quiet room tone
office ambience with keyboards
cafe ambience
crowded shopping mall
stadium crowd roaring
forest ambience at night
jungle insects
desert wind
cave water dripping
reverberant tunnel
air conditioner hum
wind blowing strongly
wind chimes
fountain splashing"""

PRESETS = [
    ("🎬 SFX pre film/reklamu – plný zoznam (~150)", PRESET_SFX_BIG),
    ("🎚 Pôvodný SFX zoznam (~75)", DEFAULT_DESCRIPTIONS),
    ("🧪 Rýchly štart (6)", PRESET_QUICK),
]


# ---------------------------------------------------------------------------
# Worker – analýza na pozadí
# ---------------------------------------------------------------------------
class AnalysisWorker(QThread):
    """Všetku AI prácu vykonáva mimo GUI vlákna."""

    row_status = pyqtSignal(int, str, str, str)   # riadok, stav, popis, detail
    value = pyqtSignal(int)                       # hodnota progress baru
    log_line = pyqtSignal(str)                    # riadok do logu
    phase = pyqtSignal(str)                       # 'model' | 'batch'
    backend_ready = pyqtSignal(str)               # info o použitom backendu
    finished_batch = pyqtSignal(int, int, bool)   # ok, chyby, zrušené

    def __init__(self, files: list[str], descriptions: list[str],
                 include_score: bool = False,
                 segments: int = DEFAULT_SEGMENTS,
                 analyzer: ClapAnalyzer | None = None, parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.descriptions = list(descriptions)
        self.include_score = include_score
        self.segments = max(1, int(segments))  # počet 10 s okien na súbor
        self.analyzer = analyzer  # znovupoužitý z predchádzajúceho behu, ak je hotový
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    # ------------------------------------------------------------------------
    def run(self) -> None:  # noqa: D102 – beží v samostatnom vlákne
        ok = err = 0

        # Model sa načíta len raz – ak už máme hotový analyzer z predošlého
        # behu (audio_session != None), znovu ho nenačítavame, len prepojíme
        # log na aktuálny worker.
        if self.analyzer is None or self.analyzer.audio_session is None:
            self.phase.emit("model")
            if self.analyzer is None:
                self.analyzer = ClapAnalyzer(log=self.log_line.emit)
            else:
                self.analyzer._log = self.log_line.emit
            try:
                self.analyzer.load()
            except Exception as exc:  # model sa nepodarilo pripraviť
                self.log_line.emit(f"✖ Chyba pri príprave modelu: {exc}")
                self.log_line.emit(traceback.format_exc())
                for i in range(len(self.files)):
                    self.row_status.emit(i, ST_ERROR, "", "Model sa nepodarilo pripraviť")
                self.finished_batch.emit(0, len(self.files), False)
                return
        else:
            self.analyzer._log = self.log_line.emit

        analyzer = self.analyzer
        self.backend_ready.emit(analyzer.backend_info)
        self.phase.emit("batch")

        # predvýpočet textových embeddingov (jednorazovo, cacheované pre celý
        # beh) – prvý súbor potom nie je "nafúknutý" o čas prípravy popisov
        try:
            t_txt = time.time()
            analyzer.embed_texts(self.descriptions)
            self.log_line.emit(f"📝 Embeddingy {len(self.descriptions)} popisov "
                               f"hotové za {time.time() - t_txt:.1f} s (cache).")
        except Exception as exc:
            err_text = str(exc).strip() or type(exc).__name__
            self.log_line.emit(f"✖ Chyba pri embeddingoch popisov: {err_text}")
            for i in range(len(self.files)):
                self.row_status.emit(i, ST_ERROR, "", err_text)
            self.finished_batch.emit(0, len(self.files), False)
            return

        for i, path in enumerate(self.files):
            if self._cancelled:
                self.row_status.emit(i, ST_SKIPPED, "", "")
                continue

            self.row_status.emit(i, ST_RUNNING, "", "")
            try:
                result = analyzer.analyze_file(path, self.descriptions,
                                               segments=self.segments)
                conf = result.confidence if self.include_score else None
                tag_msg = write_metadata(path, result.best_description, conf)

                top3 = "; ".join(f"{d} ({p * 100:.0f} %)"
                                 for d, p in result.ranking[:3])
                detail = (f"istota {result.confidence * 100:.0f} % | "
                          f"náskok +{result.margin * 100:.0f} % | "
                          f"{result.segments_used}× okno | {tag_msg} | "
                          f"{result.elapsed:.1f} s")
                self.row_status.emit(i, ST_DONE, result.best_description, detail)
                self.log_line.emit(
                    f"✔ {os.path.basename(path)} → ‘{result.best_description}’ "
                    f"({result.confidence * 100:.0f} %, náskok +{result.margin * 100:.0f} %) | {top3}")
                ok += 1
            except Exception as exc:
                err_text = str(exc).strip() or type(exc).__name__
                self.row_status.emit(i, ST_ERROR, "", err_text)
                self.log_line.emit(f"✖ {os.path.basename(path)}: {err_text}")
                err += 1

            self.value.emit(i + 1)

        self.finished_batch.emit(ok, err, self._cancelled)


# ---------------------------------------------------------------------------
# Hlavné okno
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyzátor zvukových súborov – LAION-CLAP (DirectML)")
        self.resize(1150, 700)

        self.file_paths: list[str] = []
        self._path_set: set[str] = set()
        self.worker: AnalysisWorker | None = None
        self.analyzer: ClapAnalyzer | None = None  # cache – znovupoužitý naprieč behmi
        self._batch_start_ts: float | None = None

        self._build_ui()
        self._connect_actions()
        self.log("Aplikácia spustená. Pridajte súbory (tlačidlom alebo presunte myšou) "
                 "a definujte zoznam popisov vpravo.")

    # --- stavanie UI ---------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ==== horná časť: tabuľka | panel popisov ====
        top = QHBoxLayout()
        root.addLayout(top, stretch=1)

        # ---- ľavý panel: súbory ----
        left = QVBoxLayout()
        top.addLayout(left, stretch=3)

        row1 = QHBoxLayout()
        self.btn_add_files = QPushButton("＋ Pridať súbory…")
        self.btn_add_dir = QPushButton("📂 Pridať priečinok…")
        self.btn_auto_dir = QPushButton("⚡ Automaticky spracovať priečinok…")
        self.btn_auto_dir.setToolTip(
            "Vyberie priečinok, nahradí ním aktuálny zoznam súborov\n"
            "a rovno spustí AI analýzu všetkých podporovaných súborov v ňom.")
        row1.addWidget(self.btn_add_files)
        row1.addWidget(self.btn_add_dir)
        row1.addWidget(self.btn_auto_dir)
        left.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_remove = QPushButton("✖ Odobrať vybraté")
        self.btn_clear = QPushButton("🗑 Vyčistiť zoznam")
        row2.addWidget(self.btn_remove)
        row2.addWidget(self.btn_clear)
        row2.addStretch(1)
        self.lbl_count = QLabel("Súborov: 0")
        row2.addWidget(self.lbl_count)
        left.addLayout(row2)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Súbor", "Stav", "Priradený popis", "Detail"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 330)
        self.table.setColumnWidth(3, 380)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setAcceptDrops(True)
        left.addWidget(self.table, stretch=1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        left.addWidget(self.progress)

        self.lbl_eta = QLabel("")
        self.lbl_eta.setStyleSheet("color: #666; font-size: 11px;")
        left.addWidget(self.lbl_eta)

        row3 = QHBoxLayout()
        self.lbl_backend = QLabel("Backend: ešte neznámy (spustite analýzu)")
        self.lbl_backend.setStyleSheet("color: #666;")
        self.btn_start = QPushButton("▶ Spustiť AI analýzu")
        self.btn_start.setMinimumHeight(34)
        self.btn_cancel = QPushButton("■ Zrušiť")
        self.btn_cancel.setEnabled(False)
        row3.addWidget(self.lbl_backend, stretch=1)
        row3.addWidget(self.btn_cancel)
        row3.addWidget(self.btn_start)
        left.addLayout(row3)

        # ---- pravý panel: kandidátske popisy ----
        right = QVBoxLayout()
        top.addLayout(right, stretch=2)

        lbl = QLabel("Kandidátske popisy (jeden na riadok, ‘#’ = sekcia):")
        right.addWidget(lbl)

        self.cmb_presets = QComboBox()
        for name, _text in PRESETS:
            self.cmb_presets.addItem(name)
        self.cmb_presets.setToolTip(
            "Vyberte predvoľbu – naplní pole popisov. Pole môžete ďalej "
            "ľubovoľne upravovať (maľovať riadky, pridávať vlastné).\n"
            "Tip: čím viac rozličných a konkrétnych popisov, tým presnejšie "
            "triedenie.")
        right.addWidget(self.cmb_presets)

        self.txt_descriptions = QTextEdit()
        self.txt_descriptions.setPlainText(PRESETS[0][1])
        self.txt_descriptions.setPlaceholderText(
            "Každý riadok = jeden kandidátny popis,\n"
            "napr. ‘heavy rain falling on roof’")
        right.addWidget(self.txt_descriptions, stretch=1)

        self.chk_score = QCheckBox("Zapísať aj istotu do popisu "
                                   "(napr. ‘dog barking (istota 87 %)’)")
        right.addWidget(self.chk_score)

        seg_row = QHBoxLayout()
        seg_lbl = QLabel("Úseky na súbor (presnosť):")
        self.spin_segments = QSpinBox()
        self.spin_segments.setRange(1, 8)
        self.spin_segments.setValue(DEFAULT_SEGMENTS)
        self.spin_segments.setToolTip(
            "Koľko 10-sekundových okien sa v každom súbore analyzuje\n"
            "a spriemeruje (embeddingy).\n"
            "1 = rýchle, len stred súboru (staré správanie)\n"
            "4 = odporúčané – zachytí celý obsah aj dlhších súborov\n"
            "8 = maximum presnosti (2× pomalšie oproti 4)")
        seg_row.addWidget(seg_lbl)
        seg_row.addWidget(self.spin_segments)
        seg_row.addStretch(1)
        right.addLayout(seg_row)

        # ==== spodná časť: log ====
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setMaximumHeight(130)
        self.log_view.setPlaceholderText("Log spracovania…")
        root.addWidget(self.log_view)

        self.setAcceptDrops(True)

    # --- pomocné -------------------------------------------------------------
    def log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def descriptions(self) -> list[str]:
        return [ln.strip() for ln in
                self.txt_descriptions.toPlainText().splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

    def _set_busy(self, busy: bool) -> None:
        for btn in (self.btn_add_files, self.btn_add_dir, self.btn_auto_dir,
                    self.btn_remove, self.btn_clear, self.btn_start,
                    self.txt_descriptions, self.chk_score,
                    self.cmb_presets, self.spin_segments):
            btn.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)

    # --- pridávanie / odoberanie súborov -------------------------------------
    def _add_paths(self, paths: list[str]) -> None:
        added = 0
        skipped = 0
        for p in paths:
            p = os.path.abspath(p)
            if not os.path.isfile(p) or \
                    os.path.splitext(p)[1].lower() not in SUPPORTED_EXTENSIONS:
                skipped += 1
                continue
            if p in self._path_set:
                continue
            self._path_set.add(p)
            self.file_paths.append(p)

            row = self.table.rowCount()
            self.table.insertRow(row)
            self._fill_row(row, ST_WAITING, "", "")
            self.table.setItem(row, 0, QTableWidgetItem(os.path.basename(p)))
            self.table.item(row, 0).setToolTip(p)
            added += 1
        if added:
            msg = f"Pridaných {added} súborov."
            if skipped:
                msg += f" ({skipped} preskočených – neexistujú alebo nepodporovaný formát)"
            self.log(msg)
            self.lbl_count.setText(f"Súborov: {len(self.file_paths)}")

    def _fill_row(self, row: int, status: str, tag: str, detail: str) -> None:
        st_item = QTableWidgetItem(status)
        st_item.setForeground(ST_COLORS.get(status, QColor("#000000")))
        self.table.setItem(row, 1, st_item)
        self.table.setItem(row, 2, QTableWidgetItem(tag))
        self.table.setItem(row, 3, QTableWidgetItem(detail))

    def add_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Vyberte zvukové súbory", "", FILE_FILTER)
        if paths:
            self._add_paths(paths)

    # --- skenovanie priečinkov ------------------------------------------------
    @staticmethod
    def _natural_key(s: str):
        """Prirodzené triedenie: file2 < file10 (nie file10 < file2)."""
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", s)]

    @staticmethod
    def _scan_audio_files(root: str) -> tuple[list[str], dict[str, int]]:
        """REKURZÍVNE nájde zvukové súbory v priečinku a všetkých podpriečinkoch.

        Vráti (zoznam zvukových súborov, štatistika všetkých prípon).
        Skryté priečinky (začínajúce '.') a systémové priečinky sa preskakujú.
        """
        audio: list[str] = []
        ext_stats: dict[str, int] = {}
        skip_dirs = {"$RECYCLE.BIN", "System Volume Information",
                     "__pycache__", "node_modules"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if not d.startswith(".") and d not in skip_dirs)
            for n in filenames:
                ext = os.path.splitext(n)[1].lower().strip()
                if ext:
                    ext_stats[ext] = ext_stats.get(ext, 0) + 1
                if ext in SUPPORTED_EXTENSIONS:
                    audio.append(os.path.join(dirpath, n))
        audio.sort(key=MainWindow._natural_key)
        return audio, ext_stats

    def _log_scan_result(self, root: str, audio: list[str],
                         ext_stats: dict[str, int]) -> None:
        """Diagnostika do logu – hlavný dôvod, prečo sa 'nič nenačítalo'."""
        if audio:
            n_dirs = len({os.path.dirname(p) for p in audio})
            self.log(f"🔑 Nájdených {len(audio)} zvukových súborov "
                     f"v {n_dirs} priečinkoch (vrátane podpriečinkov).")
        else:
            hint = ""
            if ext_stats:
                top = sorted(ext_stats.items(), key=lambda kv: -kv[1])[:6]
                hint = (" Iné prípony v priečinku: "
                        + ", ".join(f"{e}×{c}" for e, c in top) + ".")
            self.log(f"⚠ V priečinku „{root}“ (ani v podpriečinkoch) neboli "
                     f"nájdené žiadne podporované zvuky "
                     f"(WAV/MP3/OGG/FLAC).{hint}")

    def add_dir_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Vyberte priečinok so zvukmi")
        if not d:
            return
        audio, ext_stats = self._scan_audio_files(d)
        self._log_scan_result(d, audio, ext_stats)
        self._add_paths(audio)

    def auto_process_folder(self) -> None:
        """Vyberie priečinok, nahradí ním zoznam a rovno spustí analýzu."""
        if self.worker and self.worker.isRunning():
            return
        d = QFileDialog.getExistingDirectory(
            self, "Vyberte priečinok na automatické spracovanie")
        if not d:
            return
        audio, ext_stats = self._scan_audio_files(d)
        if not audio:
            hint = ""
            if ext_stats:
                top = sorted(ext_stats.items(), key=lambda kv: -kv[1])[:6]
                hint = "\nNájdené iné prípony: " + \
                    ", ".join(f"{e}×{c}" for e, c in top)
            QMessageBox.information(
                self, "Žiadne súbory",
                f"V priečinku „{d}“ (ani v podpriečinkoch) neboli nájdené "
                f"žiadne podporované zvukové súbory (WAV/MP3/OGG/FLAC).{hint}")
            return

        self.clear_all()
        self._add_paths(audio)

        if not self.descriptions():
            self.txt_descriptions.setPlainText(DEFAULT_DESCRIPTIONS)
            self.log("Zoznam popisov bol prázdny – použil sa predvolený zoznam.")

        self.log(f"⚡ Automatické spracovanie: {len(audio)} súborov "
                 f"z „{d}“.")
        self.start_analysis()

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            self.table.removeRow(r)
            path = self.file_paths.pop(r)
            self._path_set.discard(path)
        self.lbl_count.setText(f"Súborov: {len(self.file_paths)}")

    def clear_all(self) -> None:
        self.table.setRowCount(0)
        self.file_paths.clear()
        self._path_set.clear()
        self.lbl_count.setText("Súborov: 0")

    # --- spustenie analýzy ----------------------------------------------------
    def start_analysis(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        if not self.file_paths:
            QMessageBox.information(self, "Žiadne súbory",
                                    "Najprv pridajte zvukové súbory do zoznamu.")
            return
        descs = self.descriptions()
        if len(descs) < 2:
            QMessageBox.warning(
                self, "Málo popisov",
                "Zadajte aspoň dva kandidátske popisy (jeden na riadok),\n"
                "aby malo s čím AI porovnávať.")
            self.txt_descriptions.setFocus()
            return

        missing = [p for p in self.file_paths if not os.path.isfile(p)]
        if missing:
            self._add_paths([])  # no-op, len pre prehľadnosť
            self.log(f"⚠ Počas behu zmizne {len(missing)} súborov – budú chybovať.")

        # reset stavov
        for r in range(self.table.rowCount()):
            self._fill_row(r, ST_WAITING, "", "")

        self._set_busy(True)
        self.progress.setRange(0, 0)  # "model sa pripravuje" – animovaný pás
        self.log("=" * 70)
        self.log(f"Štartujem analýzu {len(self.file_paths)} súborov "
                 f"s {len(descs)} popismi…")

        self.log(f"Analýza: {self.spin_segments.value()}× okno/súbor, "
                 f"{len(descs)} popisov.")
        self.worker = AnalysisWorker(self.file_paths, descs,
                                     self.chk_score.isChecked(),
                                     segments=self.spin_segments.value(),
                                     analyzer=self.analyzer)
        self.worker.row_status.connect(self.on_row_status)
        self.worker.value.connect(self.on_progress_value)
        self.worker.log_line.connect(self.log)
        self.worker.phase.connect(self.on_phase)
        self.worker.backend_ready.connect(self.on_backend)
        self.worker.finished_batch.connect(self.on_finished)
        self.worker.start()

    def cancel_analysis(self) -> None:
        if self.worker and self.worker.isRunning():
            self.log("⏳ Ruším… (dokončí sa aktuálny súbor)")
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)

    # --- sloty z workera ------------------------------------------------------
    def on_preset_changed(self, index: int) -> None:
        """Vybraná predvoľba → naplní pole popisov."""
        if 0 <= index < len(PRESETS):
            self.txt_descriptions.setPlainText(PRESETS[index][1])
            n = len([ln for ln in PRESETS[index][1].splitlines()
                     if ln.strip() and not ln.strip().startswith("#")])
            self.log(f"Predvoľba ‘{PRESETS[index][0]}’ → {n} popisov.")

    def on_row_status(self, row: int, status: str, tag: str, detail: str) -> None:
        if 0 <= row < self.table.rowCount():
            self._fill_row(row, status, tag, detail)
            self.table.scrollToItem(self.table.item(row, 1),
                                    QAbstractItemView.ScrollHint.EnsureVisible)

    def on_progress_value(self, value: int) -> None:
        self.progress.setValue(value)
        total = len(self.file_paths)
        if self._batch_start_ts is None or value <= 0 or total <= 0:
            return
        elapsed = time.time() - self._batch_start_ts
        avg = elapsed / value
        remaining = avg * (total - value)
        self.lbl_eta.setText(
            f"Odhad do konca: {self._fmt_duration(remaining)}  "
            f"(priemer {avg:.1f} s/súbor, spracované {value}/{total})")

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h} h {m} min"
        if m:
            return f"{m} min {s} s"
        return f"{s} s"

    def on_phase(self, phase: str) -> None:
        if phase == "model":
            self.progress.setRange(0, 0)
            self.progress.setFormat("Pripravujem model (prvýkrát: download + export)…")
            self._batch_start_ts = None
            self.lbl_eta.setText(
                "Prvé spustenie môže trvať niekoľko minút (stiahnutie modelu "
                "+ jednorazový export do ONNX). Ďalšie spustenia appky budú rýchle.")
        else:
            self.progress.setRange(0, len(self.file_paths))
            self.progress.setFormat("%p%  (%v / %m súborov)")
            self._batch_start_ts = time.time()
            self.lbl_eta.setText("Odhad do konca: počítam…")

    def on_backend(self, info: str) -> None:
        self.lbl_backend.setText(f"Backend: {info}")
        self.lbl_backend.setStyleSheet(
            "color: #2e9e44; font-weight: bold;" if "DirectML" in info
            else "color: #e08a00; font-weight: bold;")
        self.log(f"⚙ {info}")

    def on_finished(self, ok: int, err: int, cancelled: bool) -> None:
        self._set_busy(False)
        self.progress.setRange(0, 1)
        self.progress.setFormat("")
        self._batch_start_ts = None
        self.lbl_eta.setText("")
        if self.worker:
            self.analyzer = self.worker.analyzer  # cache pre ďalší beh
            self.worker.deleteLater()
            self.worker = None

        msg = (f"Dokončené: ✔ {ok} hotovo, ✖ {err} chýb")
        if cancelled:
            msg += " (beh bol zrušený – zvyšok preskočený)"
        self.log(msg)
        QMessageBox.information(self, "Analýza dokončená", msg + ".")

    # --- drag & drop -----------------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt konvencia)
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = url.toLocalFile()
                if os.path.isdir(p) or (
                        os.path.isfile(p) and
                        os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt konvencia)
        paths: list[str] = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isdir(p):
                found, _ext = self._scan_audio_files(p)
                self._log_scan_result(p, found, _ext)
                paths += found
            else:
                paths.append(p)
        self._add_paths(paths)
        event.acceptProposedAction()

    # --- zatvorenie okna --------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt konvencia)
        if self.worker and self.worker.isRunning():
            answer = QMessageBox.question(
                self, "Analýza beží",
                "Analýza stále beží. Naozaj chcete aplikáciu ukončiť?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self.worker.cancel()
            self.log("⏳ Čakám na dokončenie aktuálneho súboru pred ukončením…")
            self._set_busy(True)
            # Aktívne čakáme, kým vlákno skutočne dobehne – vlákno sa
            # nesmie zničiť, kým beží (spôsobilo by pád aplikácie).
            # GUI zostáva odozvané vďaka processEvents().
            while self.worker is not None and self.worker.isRunning():
                self.worker.wait(200)
                QApplication.processEvents()

        event.accept()

    # --- signály --------------------------------------------------------------
    def _connect_actions(self) -> None:
        self.btn_add_files.clicked.connect(self.add_files_dialog)
        self.btn_add_dir.clicked.connect(self.add_dir_dialog)
        self.btn_auto_dir.clicked.connect(self.auto_process_folder)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_cancel.clicked.connect(self.cancel_analysis)
        self.cmb_presets.currentIndexChanged.connect(self.on_preset_changed)


# ---------------------------------------------------------------------------
def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Analyzátor zvukových súborov")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
