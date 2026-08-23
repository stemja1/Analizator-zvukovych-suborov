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
import sys
import traceback

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

from core_analyzer import SUPPORTED_EXTENSIONS, ClapAnalyzer, write_metadata

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

DEFAULT_DESCRIPTIONS = (
    "whistle sound in midnight\n"
    "heavy rain falling on roof\n"
    "car engine revving\n"
    "dog barking outdoors\n"
    "birds singing in the forest\n"
    "people talking in a room"
)


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
                 include_score: bool = False, parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.descriptions = list(descriptions)
        self.include_score = include_score
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    # ------------------------------------------------------------------------
    def run(self) -> None:  # noqa: D102 – beží v samostatnom vlákne
        ok = err = 0

        self.phase.emit("model")
        analyzer = ClapAnalyzer(log=self.log_line.emit)
        try:
            analyzer.load()
        except Exception as exc:  # model sa nepodarilo pripraviť
            self.log_line.emit(f"✖ Chyba pri príprave modelu: {exc}")
            self.log_line.emit(traceback.format_exc())
            for i in range(len(self.files)):
                self.row_status.emit(i, ST_ERROR, "", "Model sa nepodarilo pripraviť")
            self.finished_batch.emit(0, len(self.files), False)
            return

        self.backend_ready.emit(analyzer.backend_info)
        self.phase.emit("batch")

        for i, path in enumerate(self.files):
            if self._cancelled:
                self.row_status.emit(i, ST_SKIPPED, "", "")
                continue

            self.row_status.emit(i, ST_RUNNING, "", "")
            try:
                result = analyzer.analyze_file(path, self.descriptions)
                conf = result.confidence if self.include_score else None
                tag_msg = write_metadata(path, result.best_description, conf)

                top3 = "; ".join(f"{d} ({p * 100:.0f} %)"
                                 for d, p in result.ranking[:3])
                detail = f"istota {result.confidence * 100:.0f} % | {tag_msg} | {result.elapsed:.1f} s"
                self.row_status.emit(i, ST_DONE, result.best_description, detail)
                self.log_line.emit(
                    f"✔ {os.path.basename(path)} → ‘{result.best_description}’ "
                    f"({result.confidence * 100:.0f} %) | {top3}")
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

        self._build_ui()
        self._connect_actions()
        self.log(f"Aplikácia spustená. Pridajte súbory (tlačidlom alebo presunte myšou) "
                 f"a definujte zoznam popisov vpravo.")

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
        row1.addWidget(self.btn_add_files)
        row1.addWidget(self.btn_add_dir)
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

        lbl = QLabel("Kandidátske popisy (jeden na riadok):")
        right.addWidget(lbl)

        self.txt_descriptions = QTextEdit()
        self.txt_descriptions.setPlainText(DEFAULT_DESCRIPTIONS)
        self.txt_descriptions.setPlaceholderText(
            "Každý riadok = jeden kandidátny popis,\n"
            "napr. ‘heavy rain falling on roof’")
        right.addWidget(self.txt_descriptions, stretch=1)

        self.chk_score = QCheckBox("Zapísať aj istotu do popisu "
                                   "(napr. ‘dog barking (istota 87 %)’)")
        right.addWidget(self.chk_score)

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
                if ln.strip()]

    def _set_busy(self, busy: bool) -> None:
        for btn in (self.btn_add_files, self.btn_add_dir, self.btn_remove,
                    self.btn_clear, self.btn_start, self.txt_descriptions,
                    self.chk_score):
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

    def add_dir_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Vyberte priečinok so zvukmi")
        if not d:
            return
        names = [os.path.join(d, n) for n in sorted(os.listdir(d))]
        audio = [p for p in names
                 if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS]
        self._add_paths(audio)
        if not audio:
            self.log(f"V priečinku {d} neboli nájdené žiadne podporované súbory.")

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

        self.worker = AnalysisWorker(self.file_paths, descs,
                                     self.chk_score.isChecked())
        self.worker.row_status.connect(self.on_row_status)
        self.worker.value.connect(self.progress.setValue)
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
    def on_row_status(self, row: int, status: str, tag: str, detail: str) -> None:
        if 0 <= row < self.table.rowCount():
            self._fill_row(row, status, tag, detail)
            self.table.scrollToItem(self.table.item(row, 1),
                                    QAbstractItemView.ScrollHint.EnsureVisible)

    def on_phase(self, phase: str) -> None:
        if phase == "model":
            self.progress.setRange(0, 0)
            self.progress.setFormat("Pripravujem model (prvýkrát: download + export)…")
        else:
            self.progress.setRange(0, len(self.file_paths))
            self.progress.setFormat("%v / %m")

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
        if self.worker:
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
                paths += [os.path.join(p, n) for n in sorted(os.listdir(p))]
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
            self.worker.wait(5000)
        event.accept()

    # --- signály --------------------------------------------------------------
    def _connect_actions(self) -> None:
        self.btn_add_files.clicked.connect(self.add_files_dialog)
        self.btn_add_dir.clicked.connect(self.add_dir_dialog)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_cancel.clicked.connect(self.cancel_analysis)


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
