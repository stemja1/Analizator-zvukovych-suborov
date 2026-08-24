"""
core_analyzer.py
================
AI engine + zápis metadát pre aplikáciu „Analyzátor zvukových súborov".

Čo tento modul robí:
  * Načítava zvuk (WAV / MP3 / OGG / FLAC) cez librosa – 48 kHz, mono,
    stredných 10 sekúnd (na tejto dĺžke bol trénovaný LAION-CLAP).
  * LAION-CLAP (laion/clap-htsat-unfused) beží cez ONNX Runtime:
      - na Windows s AMD GPU používa DirectML (DmlExecutionProvider),
      - pri zlyhaní / neprítomnosti Automaticky prepadne na CPU.
  * Počíta cosine similarity medzi zvukom a kandidátskymi popismi,
    vyberá najlepší zápis a softmaxom počíta istotu.
  * Najlepší popis zapíše do metadát súboru (mutagen):
      MP3  -> ID3 tag COMM (Comment)
      OGG  -> Vorbis Comment 'DESCRIPTION'
      FLAC -> Vorbis Comment 'DESCRIPTION'
      WAV  -> RIFF INFO 'ICMT'

Prvé spustenie:
  1. stiahne procesor/tokenizér a váhy modelu z HuggingFace (~600 MB),
  2. jednorazovo vyexportuje ONNX grafy do priečinka `models/`,
  3. potom už len načítava existujúce ONNX súbory (rýchly štart).

Malý test bez GUI:
    python core_analyzer.py zvuk.mp3 "heavy rain on roof" "dog barking"
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from mutagen.flac import FLAC
from mutagen.id3 import COMM, ID3, ID3NoHeaderError
from mutagen.oggvorbis import OggVorbis

# ---------------------------------------------------------------------------
# Konštanty
# ---------------------------------------------------------------------------
MODEL_ID = "laion/clap-htsat-unfused"
# Pripnutá revízia (git commit na HF Hube) – bráni tichej zámene váh modelu
# pod tým istým názvom. Overené cez https://huggingface.co/api/models/laion/clap-htsat-unfused
MODEL_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"

TARGET_SR = 48_000      # Hz – CLAP pracuje s 48 kHz audiom
CLIP_SECONDS = 10       # CLAP bol trénovaný na 10-sekundových klipoch
DEFAULT_SEGMENTS = 4    # počet 10 s okien na súbor (viac = presnejšie, pomalšie)
MAX_FULL_DECODE_SECONDS = 1200   # nad 20 min sa celý súbor nenačítava (RAM)

# --- názov súboru ako pomôcka pri triedení -----------------------------------
# Slová z názvu (napr. „whoosh_final_2.wav“ → „whoosh“) sa porovnávajú
# s kandidátnymi popismi. Ak slovo na víťazný popis sedí, ide o dôkaz –
# istota sa zvýši NÁSOBENÍM (nikdy nad 99 %): 60 % → 78 %, 70 % → 91 %.
NAME_MIN_WORD_LEN = 4        # kratšie slová sú väčšinou šum (max, ver, cut…)
NAME_SKIP_MIN_WORD_LEN = 5   # 1 dlhé slovo stačí na preskočenie AI
NAME_BOOST_FACTOR = 1.3      # istota × 1,3 keď názov podporí víťaza
NAME_BOOST_CAP = 0.99        # strop – 100 % nikdy neukazujeme
AUDIO_SIM_MIN = 0.80         # podobnosť zvuku s naučeným vzorom = dôkaz
AUDIO_SIM_BOOST = 1.2        # istota × 1,2 keď zvuk sedí na naučený vzor
MULTI_RATIO = 0.4            # 2. popis sa zapíše, ak má ≥ 40 % priemeru víťaza
MULTI_EXTRA_MAX = 2          # max počet ďalších popisov (spolu max 3)
PATTERN_MAX_PER_LABEL = 30   # koľko zvukových vzorov si pamätať na popis

# všeobecné slová, ktoré o obsahu nič nehovoria – pri triedení sa ignorujú
NAME_STOPWORDS = {
    "sound", "sounds", "audio", "zvuk", "zvuky", "zvukova", "zvukove",
    "nahravka", "nahravky", "rec", "recording", "record", "file", "subor",
    "subory", "final", "finalna", "finalne", "mix", "demo", "test",
    "testovaci", "novy", "nova", "nove", "stary", "stara", "kopie",
    "kopija", "copy", "track", "sample", "samples", "edit", "uprava",
    "upraveny", "cut", "rez", "video", "wav", "mp3", "flac", "ogg",
    "the", "and", "with", "from", "this", "new", "old", "max", "min",
    "ver", "version", "hlas", "song", "klip", "full", "free", "best",
    "good", "diag", "diagnoza", "vyrok", "siec", "sit", "train",
}

# --- paralelné dekódovanie na pozadí (pipelining s GPU) ----------------------
PREFETCH_WORKERS = 2         # vlákna na dekódovanie (CPU beží paralelne s GPU)
PREFETCH_DEPTH = 3           # koľko súborov max. vopred v zásobníku

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac"}

_HERE = os.path.dirname(os.path.abspath(__file__))
TEXT_BATCH = 32          # veľkosť dávky pri embedovaní textov (ohraničenie RAM)

DEFAULT_ONNX_DIR = os.path.join(_HERE, "models", "clap_htsat_unfused_onnx")
AUDIO_ONNX_NAME = "clap_audio.onnx"
TEXT_ONNX_NAME = "clap_text.onnx"
META_NAME = "export_meta.json"

LogFn = Callable[[str], None]

# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------
def _log_to_console(msg: str) -> None:
    print(f"[CLAP] {msg}")


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return x / norm


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def load_audio_window(file_path: str, center_time: float,
                      clip_seconds: int = CLIP_SECONDS,
                      sr: int = TARGET_SR) -> np.ndarray:
    """Načíta 10-sekundové okno so stredom na `center_time` (mono @48 kHz).

    Okno sa oreže na rozsah súboru; kratší súbor sa doplní nulami na presnú
    dĺžku – vďaka tomu má ONNX vždy fixný tvar vstupu.
    """
    import librosa  # lazy import – prvý súbor chvíľu trvá, ale GUI štartuje rýchlo

    total = float(librosa.get_duration(path=file_path))
    need = float(clip_seconds)

    if total > need:
        start = max(0.0, min(center_time - need / 2.0, total - need))
        y, _ = librosa.load(file_path, sr=sr, mono=True,
                            offset=start, duration=need)
    else:
        y, _ = librosa.load(file_path, sr=sr, mono=True)

    target = clip_seconds * sr
    if y.shape[0] < target:                      # doplnenie nulami na 10 s
        y = np.pad(y, (0, target - y.shape[0]))
    elif y.shape[0] > target:                    # (od-zaokrúhlenia) orez
        y = y[:target]
    return y.astype(np.float32, copy=False)


def load_audio_center(file_path: str, clip_seconds: int = CLIP_SECONDS,
                      sr: int = TARGET_SR) -> np.ndarray:
    """Stredný klip – zachované pre spätnú kompatibilitu (CLI a pod.)."""
    import librosa
    total = float(librosa.get_duration(path=file_path))
    return load_audio_window(file_path, total / 2.0, clip_seconds, sr)


# ---------------------------------------------------------------------------
# Zápis metadát (mutagen)
# ---------------------------------------------------------------------------
def write_metadata(file_path: str, description: str,
                   confidence: float | None = None) -> str:
    """Zapíše popis do metadát podľa prípony. Vráti krátku správu do logu.

    `confidence` (0..1) – ak je zadané, pripojí sa k popisu ako istota,
    napr. „heavy rain on roof (istota 87 %)".
    """
    ext = os.path.splitext(file_path)[1].lower()
    text = description if confidence is None else \
        f"{description} (istota {confidence * 100:.0f} %)"

    if ext == ".mp3":
        # ID3 v2.4, COMM frame – pri opakovanom zápise sa nahradí
        try:
            tags = ID3(file_path)
        except ID3NoHeaderError:
            tags = ID3()
        tags.delall("COMM")
        tags.add(COMM(encoding=3, lang="eng", desc="Description", text=text))
        # v1=0: legacy ID3v1 tag nepodporuje UTF-8 (len Latin-1) a mangloval
        # by diakritiku – primárny je ID3v2 COMM vyššie, ktorý UTF-8 zapisuje správne.
        tags.save(file_path, v1=0)
        return "ID3 COMM"

    if ext == ".flac":
        tags = FLAC(file_path)
        tags["DESCRIPTION"] = [text]
        tags.save()
        return "FLAC Vorbis 'DESCRIPTION'"

    if ext == ".ogg":
        tags = OggVorbis(file_path)
        tags["DESCRIPTION"] = [text]
        tags.save()
        return "OGG Vorbis 'DESCRIPTION'"

    if ext == ".wav":
        _write_wav_icmt(file_path, text)
        return "RIFF INFO 'ICMT'"

    raise ValueError(f"Nepodporovaný typ súboru: {ext} ({file_path})")


# ---------------------------------------------------------------------------
# WAV: zápis do RIFF INFO (ICMT)
#
# Pozn.: mutagen >= 1.46 píše tagy do WAV ako ID3 chunk, ale štandardom
# Windows / RIFF INFO je práve LIST INFO s položkou ICMT – preto vlastný,
# minimálny a bezpečný writer (súbory prepisuje chunk-po-chunku).
# ---------------------------------------------------------------------------
def read_description(file_path: str) -> str:
    """Prečíta popis, ktorý do súboru zapísala táto appka (inak prázdne)."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".mp3":
            try:
                tags = ID3(file_path)
            except ID3NoHeaderError:
                return ""
            for frame in tags.getall("COMM"):
                if frame.desc == "Description":
                    return str(frame.text[0]) if frame.text else ""
            return ""
        if ext in (".flac", ".ogg"):
            tags = FLAC(file_path) if ext == ".flac" else OggVorbis(file_path)
            val = tags.get("DESCRIPTION")
            return str(val[0]) if val else ""
        if ext == ".wav":
            return _read_wav_icmt(file_path)
    except Exception:
        return ""
    return ""


def remove_description(file_path: str) -> str:
    """ZMAŽE popis z vlastností súboru. Vráti krátku správu do logu.

    Použitie: keď nová istota klesne pod prah, radšej žiaden popis
    ako nezmysel – a starý (nízko istotný) popis sa odstráni.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mp3":
        try:
            tags = ID3(file_path)
        except ID3NoHeaderError:
            return "už bolo prázdne"
        tags.delall("COMM")
        tags.save(file_path, v1=0)
        return "starý popis zmazaný (ID3)"
    if ext in (".flac", ".ogg"):
        tags = FLAC(file_path) if ext == ".flac" else OggVorbis(file_path)
        if "DESCRIPTION" in tags:
            del tags["DESCRIPTION"]
            tags.save()
            return "starý popis zmazaný (Vorbis DESCRIPTION)"
        return "už bolo prázdne"
    if ext == ".wav":
        return _remove_wav_icmt(file_path)
    raise ValueError(f"Nepodporovaný typ súboru: {ext} ({file_path})")


def _remove_wav_icmt(file_path: str) -> str:
    """Zmaže LIST/INFO chunk s popisom z WAV súboru."""
    import struct
    with open(file_path, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{file_path}: nie je platný WAV (RIFF/WAVE) súbor")
    chunks: list[tuple[bytes, bytes]] = []
    pos, removed = 12, False
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if pos + 8 + size > len(data):
            raise ValueError(f"{file_path}: poškodený WAV chunk {cid!r}")
        payload = data[pos + 8:pos + 8 + size]
        if cid == b"LIST" and payload[0:4] == b"INFO":
            removed = True                 # celý INFO blok píšeme len my
        else:
            chunks.append((cid, payload))
        pos += 8 + size + (size & 1)
    if not removed:
        return "už bolo prázdne"
    body = bytearray(b"WAVE")
    for cid, payload in chunks:
        body += cid + struct.pack("<I", len(payload)) + payload
        if len(payload) & 1:
            body += b"\x00"
    out = b"RIFF" + struct.pack("<I", len(body)) + bytes(body)
    with open(file_path, "wb") as f:
        f.write(out)
    return "starý popis zmazaný (RIFF INFO)"


def _write_wav_icmt(file_path: str, text: str) -> None:
    import struct

    with open(file_path, "rb") as f:
        data = f.read()

    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{file_path}: nie je platný WAV (RIFF/WAVE) súbor")

    # 1) rozobrať existujúce chunky
    chunks: list[tuple[bytes, bytes]] = []
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if pos + 8 + size > len(data):
            raise ValueError(f"{file_path}: poškodený WAV chunk {cid!r}")
        chunks.append((cid, data[pos + 8:pos + 8 + size]))
        pos += 8 + size + (size & 1)          # nepárne chunky majú pad byte

    # 2) existujúci LIST INFO vyhodiť (nahradíme novým), ostatné nechať
    chunks = [(cid, payload) for cid, payload in chunks
              if not (cid == b"LIST" and payload[0:4] == b"INFO")]

    # 3) nový LIST INFO s ICMT: <ID><DWORD veľkosť vrátane NUL><text+NUL><pad>
    value = text.encode("utf-8") + b"\x00"
    icmt = b"ICMT" + struct.pack("<I", len(value)) + value
    if len(icmt) & 1:
        icmt += b"\x00"
    chunks.append((b"LIST", b"INFO" + icmt))

    # 4) zostaviť a zapísať celý súbor (aktualizovaná RIFF veľkosť)
    body = bytearray(b"WAVE")
    for cid, payload in chunks:
        body += cid + struct.pack("<I", len(payload)) + payload
        if len(payload) & 1:
            body += b"\x00"
    out = b"RIFF" + struct.pack("<I", len(body)) + bytes(body)

    with open(file_path, "wb") as f:
        f.write(out)


def _read_wav_icmt(file_path: str) -> str:
    """Na čítanie späť (testy / ladenie)."""
    import struct
    with open(file_path, "rb") as f:
        data = f.read()
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        payload = data[pos + 8:pos + 8 + size]
        if cid == b"LIST" and payload[0:4] == b"INFO":
            p = 4
            while p + 8 <= len(payload):
                iid = payload[p:p + 4]
                isz = struct.unpack("<I", payload[p + 4:p + 8])[0]
                ival = payload[p + 8:p + 8 + isz]
                if iid == b"ICMT":
                    return ival.split(b"\x00", 1)[0].decode("utf-8", "replace")
                p += 8 + isz + (isz & 1)
        pos += 8 + size + (size & 1)
    return ""


# ---------------------------------------------------------------------------
# Názov súboru ako pomôcka + „učenie“ (slová aj zvukové vzory)
# ---------------------------------------------------------------------------
LEARNED_FILE = os.path.join(_HERE, "naucene_spojenia.json")
PATTERNS_FILE = os.path.join(_HERE, "naucene_vzory.npz")


def filename_keywords(file_path: str) -> list[str]:
    """Významové ANGLICKÉ slová z názvu súboru (bez čísel a všeobecných slov).

    „whoosh_final_2.wav“ → [„whoosh“]; „dog_barking_03.mp3“ → [„dog“, „barking“]
    Slová s diakritikou (napr. „zvonenie“, „hodín“) sa IGNORUJÚ – učenie a
    párovanie s popismi funguje len na anglické názvy súborov (popisy
    CLAP modelu sú anglické, slovenské názvy by len znížili presnosť).
    """
    base = os.path.splitext(os.path.basename(file_path))[0].lower()
    words = [w for w in re.findall(r"[^\W\d_]+", base, re.UNICODE)
             if len(w) >= NAME_MIN_WORD_LEN and w.isascii()
             and w not in NAME_STOPWORDS]
    out: list[str] = []
    for w in words:
        if w not in out:
            out.append(w)
    return out


def keyword_in_description(keyword: str, description: str) -> bool:
    """Sedí kľúčové slovo na popis? Podreťazec; „birds“ sedí aj na „bird“."""
    d = description.lower()
    if keyword in d:
        return True
    if keyword.endswith("s") and len(keyword) > NAME_MIN_WORD_LEN:
        return keyword[:-1] in d
    return False


def name_matches_description(file_path: str, description: str,
                             learned: dict | None = None) -> bool:
    """Sedí nejaké slovo z názvu na popis (textovo alebo naučene)?"""
    for k in filename_keywords(file_path):
        if keyword_in_description(k, description):
            return True
        assoc = (learned or {}).get(k)
        if assoc and assoc.get(description, 0) >= 1:
            return True
    return False


def name_skip_description(file_path: str, descriptions: list[str],
                          learned: dict | None = None) -> str | None:
    """Popis, ak NÁZOV súboru jednoznačne určuje práve jeden popis.

    Konzervatívne podmienky (AI sa nesmie preskočiť len tak):
    * ≥ 2 slová z názvu sedia na ten istý popis a nikto iný nemá toľko,
      ALEBO 1 dlhé slovo (≥ 5 znakov) sedí len na jeden popis,
      ALEBO naučené spojenie: slovo už 2× viedlo k rovnakému popisu.
    Pri remíze → None (beží AI). Vrátený popis sa zapíše bez AI.
    """
    kws = filename_keywords(file_path)
    if not kws:
        return None
    hits = [sum(1 for k in kws if keyword_in_description(k, d))
            for d in descriptions]
    best_i = max(range(len(descriptions)), key=lambda i: hits[i])
    best = hits[best_i]
    if best > 0 and hits.count(best) == 1:
        strong = best >= 2 or any(
            len(k) >= NAME_SKIP_MIN_WORD_LEN
            and keyword_in_description(k, descriptions[best_i]) for k in kws)
        if strong:
            return descriptions[best_i]
    if learned:
        for k in kws:
            assoc = learned.get(k)
            if not assoc:
                continue
            tops = [d for d, n in assoc.items() if n >= 2]
            if len(tops) == 1 and assoc[tops[0]] >= 2 \
                    and tops[0] in descriptions:
                return tops[0]
    return None


def load_learned() -> dict:
    """Naučené spojenia slovo → {popis: počet} (naucene_spojenia.json)."""
    try:
        with open(LEARNED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_learned(learned: dict) -> None:
    try:
        with open(LEARNED_FILE, "w", encoding="utf-8") as f:
            json.dump(learned, f, ensure_ascii=False, indent=1)
    except Exception:
        pass                                # bez učenia sa beží ďalej


def learn_words(file_path: str, description: str,
                descriptions: list[str],
                learned: dict | None = None) -> list[str]:
    """Zapamätá si spojenia slovo z názvu → priradený popis.

    Učí sa len slová, ktoré v žiadnom popise nie sú (inak nič nové
    nenaučí). Mení `learned` v pamäti; ukladať treba cez save_learned().
    Vráti zoznam novo zapamätaných slov (na log).
    """
    learned_now: list[str] = []
    learned = learned if learned is not None else load_learned()
    for k in filename_keywords(file_path):
        if any(keyword_in_description(k, d) for d in descriptions):
            continue
        counts = learned.setdefault(k, {})
        counts[description] = counts.get(description, 0) + 1
        if len(counts) > 3:                 # pamäť: max 3 popisy na slovo
            for d, _ in sorted(counts.items(), key=lambda kv: kv[1])[:-3]:
                del counts[d]
        learned_now.append(k)
    return learned_now


def load_patterns() -> dict:
    """Naučené zvukové vzory: {'emb': (N, D), 'label': [str]}.

    Každý istý výsledok (popis + embedding zvuku) si program pamätá ako
    „vzor“ – podľa podobnosti zvuku potom vie radšej tipnúť aj súbory
    s nevravným názvom.
    """
    try:
        z = np.load(PATTERNS_FILE, allow_pickle=False)
        emb = z["emb"].astype(np.float32)
        label = [str(x) for x in z["label"]]
        if emb.ndim == 2 and len(label) == emb.shape[0] and emb.shape[0]:
            return {"emb": emb, "label": label}
    except Exception:
        pass
    return {"emb": np.zeros((0, 1), np.float32), "label": []}


def save_patterns(patterns: dict) -> None:
    try:
        emb = np.asarray(patterns.get("emb"), np.float32)
        if emb.ndim != 2 or emb.shape[0] == 0:
            return
        np.savez_compressed(PATTERNS_FILE, emb=emb,
                            label=np.array(patterns.get("label", [])))
    except Exception:
        pass


def add_pattern(patterns: dict, embedding: np.ndarray, label: str) -> None:
    """Pridá zvukový vzor (embedding → popis) do pamäti vzorov."""
    v = np.asarray(embedding, np.float32).reshape(1, -1)
    labels = patterns.get("label", [])
    emb = patterns.get("emb")
    if not labels or emb is None or emb.shape[0] == 0 \
            or emb.shape[1] != v.shape[1]:
        patterns["emb"], patterns["label"] = v, [label]
        return
    same = [i for i, l in enumerate(labels) if l == label]
    if len(same) >= PATTERN_MAX_PER_LABEL:  # zahodiť najstarší vzor popisu
        drop = same[0]
        keep = [j for j in range(len(labels)) if j != drop]
        patterns["emb"] = emb[keep]
        patterns["label"] = [labels[j] for j in keep]
    patterns["emb"] = np.vstack([patterns["emb"], v])
    patterns["label"].append(label)


def find_similar_pattern(embedding: np.ndarray,
                         patterns: dict) -> tuple[str, float]:
    """Najpodobnejší naučený zvukový vzor → (popis, podobnosť 0..1)."""
    emb, labels = patterns.get("emb"), patterns.get("label", [])
    if emb is None or not labels or emb.shape[0] != len(labels) \
            or emb.shape[0] == 0:
        return ("", 0.0)
    v = np.asarray(embedding, np.float32).ravel()
    if emb.shape[1] != v.shape[0]:
        return ("", 0.0)
    norm_e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    sims = norm_e @ v
    i = int(np.argmax(sims))
    return (labels[i], float(sims[i]))


# ---------------------------------------------------------------------------
# Výsledok analýzy
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    file_path: str
    best_description: str
    confidence: float                     # 0..1
    ranking: list[tuple[str, float]]      # zostupne zoradené (popis, p)
    margin: float                         # náskok pred 2. kandidátom (0..1)
    segments_used: int                    # počet analyzovaných 10 s okien
    decode_time: float = 0.0              # načítanie/dekódovanie súboru (s)
    infer_time: float = 0.0               # GPU/CPU inference (s)
    name_boosted: bool = False            # názov súboru podporil výber
    pattern_boosted: bool = False         # naučený zvukový vzor podporil výber
    skipped_by_name: bool = False         # AI preskočená (jednoznačný názov)
    additional: list = None               # ďalšie popisy [(popis, podiel)]
    audio_embedding: object = None        # spriemerovaný CLAP embedding zvuku
    backend: str = ""
    elapsed: float = 0.0

    def __post_init__(self):
        if self.additional is None:
            self.additional = []


# ---------------------------------------------------------------------------
# Analyzátor (CLAP + ONNX Runtime / DirectML)
# ---------------------------------------------------------------------------
class ClapAnalyzer:
    """LAION-CLAP zero-shot klasifikátor zvuku s DirectML akceleráciou."""

    def __init__(self, model_id: str = MODEL_ID,
                 onnx_dir: str = DEFAULT_ONNX_DIR,
                 revision: str | None = MODEL_REVISION,
                 log: LogFn | None = None):
        self.model_id = model_id
        self.onnx_dir = onnx_dir
        self.revision = revision
        self._log: LogFn = log or _log_to_console

        self.processor = None          # transformers ClapProcessor
        self.feature_extractor = None
        self.tokenizer = None
        self.audio_session = None      # onnxruntime InferenceSession
        self.text_session = None       # onnxruntime session ALEBO None → torch fallback
        self._text_torch_model = None  # ClapTextModelWithProjection (fallback)
        self.providers_used: list[str] = []
        self.backend_info: str = "neinicializované"
        self._meta: dict = {}
        self._text_cache: dict[tuple, np.ndarray] = {}
        self._prefetches: dict[str, object] = {}  # path -> future (max 2 položky)
        self._pool = None                    # ThreadPoolExecutor (lazy)
        self._stats = {"decodes": 0, "gpu_calls": 0}  # telemetria výkonu

    # -- verejné API --------------------------------------------------------
    def load(self) -> None:
        """Pripraví model (download/export pri prvom spustení) a ORT sessions."""
        t0 = time.time()
        self._log(f"Načítavam procesor a tokenizér ‘{self.model_id}’…")
        from transformers import AutoProcessor  # lazy import (ťažký)

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, revision=self.revision)
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

        meta_path = os.path.join(self.onnx_dir, META_NAME)
        audio_path = os.path.join(self.onnx_dir, AUDIO_ONNX_NAME)
        text_path = os.path.join(self.onnx_dir, TEXT_ONNX_NAME)

        if not os.path.isfile(meta_path) or not os.path.isfile(audio_path):
            self._export_onnx(audio_path, text_path, meta_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

        self._log("Vytváram ONNX Runtime session…")
        import onnxruntime as ort  # lazy import

        self.audio_session = self._make_session(ort, audio_path)

        # text: ONNX ak existuje a funguje; inak torch fallback (rovnaké výsledky)
        if os.path.isfile(text_path) and \
                self._meta.get("text_backend", "onnx") == "onnx":
            try:
                self.text_session = self._make_session(ort, text_path)
            except Exception as exc:
                self._log(f"⚠ text ONNX session zlyhala ({exc}) → torch fallback")
        if self.text_session is None:
            self._init_torch_text_backend()

        dml = "DmlExecutionProvider" in self.providers_used
        audio_desc = "audio: ONNX/" + ("DirectML" if dml else "CPU")
        text_desc = "text: ONNX" if self.text_session is not None else "text: torch/CPU"
        self.backend_info = f"{audio_desc}, {text_desc} | onnxruntime {ort.__version__}"
        self._log(f"Model pripravený za {time.time() - t0:.1f} s → {self.backend_info}")

    def _init_torch_text_backend(self) -> None:
        """Fallback: textové embeddingy priamo cez torch (CPU).

        Používa sa, ak ONNX export/session textového encodera zlyhá
        (napr. málo RAM) – výsledky sú identické (overené cosine = 1.0).
        """
        import torch  # noqa: F401 (transformers ho majú aj tak)
        from transformers import ClapTextModelWithProjection

        self._log("Načítavam text encoder (torch, CPU)…")
        self._text_torch_model = \
            ClapTextModelWithProjection.from_pretrained(
                self.model_id, revision=self.revision).eval()

    def analyze_file(self, file_path: str,
                     candidate_descriptions: list[str],
                     segments: int = DEFAULT_SEGMENTS,
                     use_name_hint: bool = True,
                     learned: dict | None = None,
                     patterns: dict | None = None) -> AnalysisResult:
        """Vráti najlepší popis pre daný zvukový súbor + celé poradie.

        Presnosť: súbor sa analyzuje v `segments` rovnomerne rozmiestených
        10-sekundových oknách (embeddingy sa spriemerujú). Súbor dlhší ako
        10 s tak zachytí celý svoj obsah, nielen stred. Pri krátkom súbore
        stačí jediné okno (doplnené nulami).
        """
        if self.audio_session is None:
            raise RuntimeError("Model nie je načítaný – zavolajte najprv load().")
        if not candidate_descriptions:
            raise ValueError("Chýba zoznam kandidátskych popisov.")

        t0 = time.time()

        # okná: z prefetchu (ak GUI predložilo) alebo vypočítaj teraz;
        # súbor sa dekóduje JEDNÝM volaním a okná sa vykroja z pamäte
        windows, n_windows, decode_time = self._take_windows(file_path, segments)

        t_inf = time.time()
        embs = self.embed_audio_batch(windows)          # 1 GPU volanie (batch)
        audio_emb = _l2norm(np.mean(embs, axis=0, keepdims=True))[0]
        infer_time = time.time() - t_inf

        text_emb = self.embed_texts(candidate_descriptions)  # (N, D)

        logits = (text_emb @ audio_emb) * float(self._meta.get("logit_scale", 100.0))
        probs = _softmax(logits.astype(np.float64))
        order = np.argsort(-probs)
        best = int(order[0])
        second = float(probs[int(order[1])]) if len(order) > 1 else 0.0

        # --- viac popisov, ak nahrávka obsahuje viac typov zvuku ----------
        # Popis sa pridá, ak aspoň v jednom okne VYHRÁ a jeho priemerná
        # pravdepodobnosť má ≥ MULTI_RATIO priemeru víťaza (je výrazne
        # prítomný v nahrávke, nie len náhoda v jednom okne).
        additional: list[tuple[str, float]] = []
        if n_windows >= 2:
            w_logits = (text_emb @ embs.T) * float(
                self._meta.get("logit_scale", 100.0))
            w_probs = np.stack(
                [_softmax(w_logits[:, j].astype(np.float64))
                 for j in range(n_windows)], axis=1)      # (D, n_windows)
            winners = np.argmax(w_probs, axis=0)
            best_mean = float(probs[best])
            extra = []
            for j in range(len(candidate_descriptions)):
                if j == best:
                    continue
                mean_p = float(probs[j])
                if int((winners == j).sum()) >= 1 \
                        and mean_p >= MULTI_RATIO * best_mean:
                    extra.append((j, mean_p))
            extra.sort(key=lambda t: -t[1])
            additional = [(candidate_descriptions[j], mean_p)
                          for j, mean_p in extra[:MULTI_EXTRA_MAX]]

        # --- istota: posilnenie názvom súboru a naučeným zvukovým vzorom --
        conf = float(probs[best])
        name_boosted = pattern_boosted = False
        if use_name_hint and name_matches_description(
                file_path, candidate_descriptions[best], learned):
            conf = min(NAME_BOOST_CAP, conf * NAME_BOOST_FACTOR)
            name_boosted = True
        if patterns is not None:
            sim_label, sim = find_similar_pattern(audio_emb, patterns)
            if sim_label == candidate_descriptions[best] \
                    and sim >= AUDIO_SIM_MIN:
                conf = min(NAME_BOOST_CAP, conf * AUDIO_SIM_BOOST)
                pattern_boosted = True

        return AnalysisResult(
            file_path=file_path,
            best_description=candidate_descriptions[best],
            confidence=conf,
            ranking=[(candidate_descriptions[i], float(probs[i])) for i in order],
            margin=float(probs[best]) - second,
            segments_used=n_windows,
            decode_time=decode_time,
            infer_time=infer_time,
            name_boosted=name_boosted,
            pattern_boosted=pattern_boosted,
            additional=additional,
            audio_embedding=audio_emb,
            backend=self.backend_info,
            elapsed=time.time() - t0,
        )

    # -- príprava okien (1 dekód na súbor) + prefetch -------------------------
    def _prepare_windows(self, file_path: str,
                         segments: int) -> tuple[list[np.ndarray], int, float]:
        """Vráti (zoznam 10 s okien, počet, čas dekódovania).

        Súbor sa dekóduje JEDNÝM volaním librosa.load a okná sa vykroja
        z pamäte (predtým N dekódovaní — MP3 s offsetom sa dekódovalo
        vždy od začiatku súboru). Extrémne dlhé súbor (> 20 min) sa z
        dôvodu RAM načítavajú postupne po oknách.
        """
        t0 = time.time()
        import librosa

        total = float(librosa.get_duration(path=file_path))
        segments = max(1, int(segments))
        clip_n = CLIP_SECONDS * TARGET_SR

        if total <= CLIP_SECONDS + 0.5:
            starts = [0.0]
        else:
            centers = [(i + 0.5) / segments * total for i in range(segments)]
            starts, seen = [], set()
            for c in centers:
                s = round(max(0.0, min(c - CLIP_SECONDS / 2.0,
                                       total - CLIP_SECONDS)), 1)
                if s not in seen:
                    seen.add(s)
                    starts.append(s)

        windows: list[np.ndarray] = []
        if total <= MAX_FULL_DECODE_SECONDS:
            y, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
            self._stats["decodes"] += 1
            y = np.asarray(y, dtype=np.float32)
            for s in starts:
                i0 = int(round(s * TARGET_SR))
                w = y[i0:i0 + clip_n]
                if w.shape[0] < clip_n:
                    w = np.pad(w, (0, clip_n - w.shape[0]))
                windows.append(np.ascontiguousarray(w))
        else:
            for s in starts:
                windows.append(load_audio_window(
                    file_path, s + CLIP_SECONDS / 2.0))
                self._stats["decodes"] += 1

        return windows, len(starts), time.time() - t0

    def _take_windows(self, file_path: str,
                      segments: int) -> tuple[list[np.ndarray], int, float]:
        """Použije preddekódované okná z prefetchu, ak sedia; inak počíta.

        Prefetch pre INÝ súbor sa zahodí len pri spotrebovaní vlastného
        súboru – cudzí future zostáva v zásobníku (použije sa neskôr),
        takže sa nič zbytočne nedekóduje dvakrát.
        """
        fut = self._prefetches.pop(file_path, None)
        if fut is not None:
            return fut.result()
        return self._prepare_windows(file_path, segments)

    def preload(self, file_path: str, segments: int = DEFAULT_SEGMENTS) -> None:
        """Preddekóduje súbor vo vlákne na pozadí.

        Využitie: kým GPU analyzuje súbor N, CPU medzitým dekóduje súbor
        N+1 (prekrytie CPU a GPU práce → vyššie využitie GPU). Uchová sa
        max. 2 súbory vopred (aktuálny + ďalší), aby nevyrástla pamäť.
        """
        import concurrent.futures
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=PREFETCH_WORKERS, thread_name_prefix="clap-preload")
        if file_path in self._prefetches:
            return
        while len(self._prefetches) >= PREFETCH_DEPTH:
            old_path, old_fut = next(iter(self._prefetches.items()))
            old_fut.cancel()
            del self._prefetches[old_path]
        self._prefetches[file_path] = self._pool.submit(
            self._prepare_windows, file_path, segments)

    def close(self) -> None:
        """Uvoľní prefetch vlákno."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        self._prefetches = {}

    # -- embedovanie ----------------------------------------------------------
    def embed_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Waveform (48 kHz, mono, 10 s) -> normalizovaný CLAP embedding."""
        return self.embed_audio_batch([waveform])[0]

    def embed_audio_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Všetky okná v JEDNOM GPU volaní -> (N, D) normalizované embeddingy.

        Audio graf má dynamický batch (export od v0.4.0); pri staršom grafe
        s fixným batch=1 prebehne fallback po riadkoch (výsledky rovnaké).
        """
        feats = self.feature_extractor(
            [np.asarray(w, dtype=np.float32) for w in waveforms],
            sampling_rate=TARGET_SR, return_tensors="np",
        )
        x = self._fit_audio_shape(
            np.ascontiguousarray(feats["input_features"], dtype=np.float32))
        try:
            out = self.audio_session.run(None, {"input_features": x})[0]
            self._stats["gpu_calls"] += 1
            return _l2norm(np.asarray(out, dtype=np.float32)
                           .reshape(len(waveforms), -1))
        except Exception:
            rows = []
            for i in range(x.shape[0]):
                out = self.audio_session.run(
                    None, {"input_features": x[i:i + 1]})[0]
                self._stats["gpu_calls"] += 1
                rows.append(np.asarray(out, dtype=np.float32).reshape(1, -1))
            return _l2norm(np.concatenate(rows, axis=0))

    def embed_texts(self, descriptions: list[str]) -> np.ndarray:
        """Zoznam popisov -> (N, D) normalizované embeddingy (s cache)."""
        key = tuple(descriptions)
        if key in self._text_cache:
            return self._text_cache[key]

        enc = self.tokenizer(
            list(descriptions),
            padding="max_length", truncation=True,
            max_length=int(self._meta.get("text_tokens", 32)),
            return_tensors="np",
        )
        input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(enc["attention_mask"], dtype=np.int64)

        if self.text_session is not None:            # ONNX cesta
            n = input_ids.shape[0]
            try:
                # graf s dynamickým batchom → dávky po 32 (ohraničená RAM)
                rows = []
                for i in range(0, n, TEXT_BATCH):
                    out = self.text_session.run(None, {
                        "input_ids": input_ids[i:i + TEXT_BATCH],
                        "attention_mask": attention_mask[i:i + TEXT_BATCH],
                    })[0]
                    rows.append(np.asarray(out, dtype=np.float32))
                emb = np.concatenate(rows, axis=0) if rows else \
                    np.zeros((n, int(self._meta.get("embedding_dim", 512))),
                             dtype=np.float32)
            except Exception:
                # starší graf s fixným batch=1 → po riadkoch (výsledky rovnaké)
                rows = []
                for i in range(n):
                    out = self.text_session.run(None, {
                        "input_ids": input_ids[i:i + 1],
                        "attention_mask": attention_mask[i:i + 1],
                    })[0]
                    rows.append(np.asarray(out, dtype=np.float32).reshape(1, -1))
                emb = np.concatenate(rows, axis=0)
        else:                                        # torch fallback (CPU)
            import torch
            with torch.no_grad():
                out = self._text_torch_model(
                    input_ids=torch.from_numpy(input_ids),
                    attention_mask=torch.from_numpy(attention_mask),
                )
                emb = out.text_embeds.numpy()

        emb = _l2norm(emb.reshape(len(descriptions), -1))
        self._text_cache[key] = emb
        return emb

    # -- ONNX Runtime ---------------------------------------------------------
    def _make_session(self, ort, onnx_path: str):
        """Session s DirectML; pri zlyhaní automatický fallback na CPU."""
        available = list(ort.get_available_providers())
        wanted: list[str] = []
        if sys.platform == "win32" and "DmlExecutionProvider" in available:
            wanted.append("DmlExecutionProvider")   # AMD RX 7700 XT
        if "CPUExecutionProvider" in available:
            wanted.append("CPUExecutionProvider")
        if not wanted:
            wanted = available[:1] or ["CPUExecutionProvider"]

        providers_arg: list = []
        for p in wanted:
            if p == "DmlExecutionProvider":
                providers_arg.append((p, {"device_id": 0}))
            else:
                providers_arg.append(p)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                        providers=providers_arg)
        except Exception as exc:
            if "DmlExecutionProvider" in wanted:
                self._log(f"⚠ DirectML zlyhalo ({type(exc).__name__}: {exc}) → "
                          f"prehádzam na CPU.")
                sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            else:
                raise

        self.providers_used = list(sess.get_providers())
        return sess

    # -- jednorazový ONNX export (vyžaduje torch, len pri prvom spustení) -----
    # Každý graf sa exportuje v SAMOSTATNOM subprocesse – šetrí RAM
    # (model sa nenačítava dvakrát do jedného procesu) a zlyhanie
    # jedného kroku nepoškodí druhý.
    def _export_onnx(self, audio_path: str, text_path: str, meta_path: str) -> None:
        self._log("ONNX grafy sa nenašli – robím JEDNORAZOVÝ export "
                  "(prvýkrát to môže trvať aj pár minút)…")
        os.makedirs(self.onnx_dir, exist_ok=True)

        self._log("Exportujem audio encoder (HTSAT)…")
        self._export_with_retry("--_export-audio", "audio")

        text_backend = "onnx"
        try:
            self._log("Exportujem text encoder…")
            self._export_with_retry("--_export-text", "text")
        except Exception as exc:
            self._log(f"⚠ ONNX export textu zlyhal ({type(exc).__name__}) – "
                      "textové embeddingy pôjdu cez torch (CPU), výsledky "
                      "sú identické.")
            text_backend = "torch"

        # zlúčiť častice metadát od subprocessov
        merged: dict = {}
        audio_part = os.path.join(self.onnx_dir, "_meta_audio.json")
        with open(audio_part, "r", encoding="utf-8") as f:
            merged.update(json.load(f))
        os.remove(audio_part)

        text_part = os.path.join(self.onnx_dir, "_meta_text.json")
        if os.path.isfile(text_part):
            with open(text_part, "r", encoding="utf-8") as f:
                merged.update(json.load(f))
            os.remove(text_part)
        else:  # doplniť text_tokens z tokenizéra (už je načítaný)
            tokens = 32
            for cand in (getattr(self.tokenizer, "model_max_length", None), 32):
                if isinstance(cand, int) and 8 <= cand <= 128:
                    tokens = min(cand, 128)
                    break
            merged["text_tokens"] = tokens
        merged["text_backend"] = text_backend

        merged.update({"model_id": self.model_id, "opset": 17})
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        self._log(f"Export hotový → {self.onnx_dir}")

    def _export_with_retry(self, flag: str, graph: str) -> None:
        """TorchScript export; pri zlyhaní (aj OOM-kill subprocessu) retry
        s novým torch.export/dynamo exporterom (nižšia spotreba RAM)."""
        try:
            self._run_subexport(flag)
        except Exception as first_error:
            self._log(f"⚠ {graph}: TorchScript export zlyhal "
                      f"({type(first_error).__name__}) – skúšam dynamo exporter…")
            try:
                self._run_subexport(flag + "-dynamo")
            except Exception:
                raise first_error

    def _run_subexport(self, flag: str) -> None:
        import subprocess
        cmd = [sys.executable, os.path.abspath(__file__),
               flag, self.onnx_dir]
        self._log(f"   subprocess: {os.path.basename(cmd[1])} {flag}")
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.stdout:
            for line in proc.stdout.strip().splitlines()[-6:]:
                self._log(f"   │ {line}")
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-2000:]
            raise RuntimeError(f"ONNX export zlyhal ({flag}):\n{tail}")

    @staticmethod
    def _torch_export(module, args, path: str, **kwargs):
        """torch.onnx.export s kompatibilitou pre novšie verzie torch."""
        import torch
        try:
            torch.onnx.export(module, args, path, opset_version=17,
                              do_constant_folding=False, dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(module, args, path, opset_version=17,
                              do_constant_folding=False, **kwargs)

    def _fit_audio_shape(self, x: np.ndarray) -> np.ndarray:
        """Upraví tvary na hodnoty z exportu (os 0 = voľný batch)."""
        expected = self._meta.get("audio_input_shape")
        if not expected:
            return np.ascontiguousarray(x, dtype=np.float32)

        axis = int(self._meta.get("frames_axis", len(expected) - 2))
        want = int(self._meta.get("frames", expected[axis]))
        got = int(x.shape[axis]) if axis < x.ndim else -1
        if got != want:
            if got > want:
                sl = [slice(None)] * x.ndim
                sl[axis] = slice(0, want)
                x = x[tuple(sl)]
            else:
                pad = [(0, 0)] * x.ndim
                pad[axis] = (0, want - got)
                x = np.pad(x, pad)
        if x.ndim != len(expected):     # normalizuj na (batch, *expected[1:])
            x = x.reshape((x.shape[0], *expected[1:]))
        return np.ascontiguousarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Interná fáza exportu – spúšťaná ako samostatný subprocess
# (používa sa menej RAM: každý graf má vlastný proces)
# ---------------------------------------------------------------------------
def _internal_export(mode: str, onnx_dir: str) -> int:
    import gc

    import torch
    from transformers import AutoProcessor

    dynamo = mode.endswith("-dynamo")     # nový torch.export exporter
    graph = mode.split("-")[0]            # "audio" | "text"
    if dynamo:
        print("(používam dynamo/torch.export exporter)")

    print("Načítavam procesor…")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    os.makedirs(onnx_dir, exist_ok=True)

    def _do_export(module, args, path, dynamic_axes=None, dynamic_shapes=None, **kw):
        if dynamo:
            # nový exporter: dynamické tvary cez dynamic_shapes
            kw.pop("do_constant_folding", None) if "do_constant_folding" in kw else None
            torch.onnx.export(module, args, path,
                              dynamic_shapes=dynamic_shapes, **kw)
        else:
            # TorchScript exporter: dynamické tvary cez dynamic_axes
            ClapAnalyzer._torch_export(module, args, path,
                                       dynamic_axes=dynamic_axes, **kw)

    def _pooler_or_tensor(out):
        """transformers v5 vracia ModelOutput, v4 priamo tensor."""
        return out.pooler_output if hasattr(out, "pooler_output") else out

    class _AudioFn(torch.nn.Module):
        """Plný ClapModel.get_audio_features (vrátane projekcie)."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_features):
            return _pooler_or_tensor(
                self.m.get_audio_features(input_features=input_features))

    class _TextFn(torch.nn.Module):
        """Plný ClapModel.get_text_features (vrátane projekcie)."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return _pooler_or_tensor(self.m.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask))

    class _TextProjFn(torch.nn.Module):
        """Ľahká cesta: len text encoder + projekcia (~1/3 RAM).
        Overené: výsledky identické s plným modelom (cosine = 1.0)."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids,
                          attention_mask=attention_mask).text_embeds

    def _load_full():
        from transformers import ClapModel
        try:
            m = ClapModel.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, attn_implementation="eager")
        except TypeError:  # staršie transformers bez attn_implementation
            m = ClapModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        return m.eval()

    def _free(model, unneeded_attrs):
        # uvoľniť časť modelu, ktorú teraz nepotrebujeme (šetrí RAM)
        for attr in unneeded_attrs:
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                    gc.collect()
                except Exception:
                    pass

    if graph == "audio":
        model = _load_full()
        _free(model, ("clap_text", "text_model"))

        dummy_wave = np.zeros(CLIP_SECONDS * TARGET_SR, dtype=np.float32)
        feats = processor.feature_extractor(
            dummy_wave, sampling_rate=TARGET_SR, return_tensors="pt")
        input_features = feats["input_features"]

        # os „času" = tá s väčším rozmerom z posledných dvoch (snímky ~1000 vs. mel 64)
        frames_axis = int(np.argmax(input_features.shape[-2:])) + \
            len(input_features.shape) - 2

        audio_dyn_axes = {"input_features": {0: "batch"},
                          "audio_embeds": {0: "batch"}}

        print(f"Exportujem clap_audio.onnx (vstup {tuple(input_features.shape)})…")
        with torch.no_grad():
            _do_export(
                _AudioFn(model), (input_features,),
                os.path.join(onnx_dir, AUDIO_ONNX_NAME),
                input_names=["input_features"], output_names=["audio_embeds"],
                dynamic_axes=audio_dyn_axes)

        logit_scale = 100.0
        try:
            logit_scale = float(model.logit_scale_a.exp().item())
        except Exception:
            try:
                sd = model.state_dict()
                key = next(k for k in sd if "logit_scale_a" in k)
                logit_scale = float(sd[key].exp().item())
            except Exception:
                print("⚠ logit_scale sa nepodarilo prečítať, použijem 100.0")
        with torch.no_grad():
            out = model.get_audio_features(input_features=input_features)
            dim = int(_pooler_or_tensor(out).shape[-1])

        meta = {
            "audio_input_shape": [int(v) for v in input_features.shape],
            "frames": int(input_features.shape[frames_axis]),
            "frames_axis": frames_axis,
            "logit_scale": logit_scale,
            "embedding_dim": dim,
        }
        with open(os.path.join(onnx_dir, "_meta_audio.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"clap_audio.onnx hotový (embedding_dim={dim}, "
              f"logit_scale={logit_scale:.2f})")

    elif graph == "text":
        tokenizer = processor.tokenizer
        text_tokens = 32
        for cand in (getattr(tokenizer, "model_max_length", None), 32):
            if isinstance(cand, int) and 8 <= cand <= 128:
                text_tokens = min(cand, 128)
                break

        enc = tokenizer(["a sound"], padding="max_length", truncation=True,
                        max_length=text_tokens, return_tensors="pt")

        print(f"Exportujem clap_text.onnx (vstup {tuple(enc['input_ids'].shape)})…")

        light_model = None
        try:
            from transformers import ClapTextModelWithProjection
            light_model = ClapTextModelWithProjection.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION).eval()
        except Exception as exc:
            print(f"⚠ ľahký text model nedostupný ({exc}) – "
                  f"použijem plný ClapModel")

        # dynamický batch: starý exporter = stringy, dynamo = objekty Dim
        text_dyn_axes = {"input_ids": {0: "batch"},
                         "attention_mask": {0: "batch"},
                         "text_embeds": {0: "batch"}}
        text_dyn_shapes = None
        try:
            from torch.export import Dim
            _b = Dim("batch")
            text_dyn_shapes = {"input_ids": {0: _b},
                               "attention_mask": {0: _b}}
        except Exception:
            pass  # veľmi starý torch → fixný batch → embed_texts použije slučku

        with torch.no_grad():
            if light_model is not None:
                _do_export(
                    _TextProjFn(light_model),
                    (enc["input_ids"], enc["attention_mask"]),
                    os.path.join(onnx_dir, TEXT_ONNX_NAME),
                    input_names=["input_ids", "attention_mask"],
                    output_names=["text_embeds"],
                    dynamic_axes=text_dyn_axes,
                    dynamic_shapes=text_dyn_shapes)
            else:
                model = _load_full()
                _free(model, ("clap_audio", "audio_model"))
                _do_export(
                    _TextFn(model),
                    (enc["input_ids"], enc["attention_mask"]),
                    os.path.join(onnx_dir, TEXT_ONNX_NAME),
                    input_names=["input_ids", "attention_mask"],
                    output_names=["text_embeds"])

        with open(os.path.join(onnx_dir, "_meta_text.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"text_tokens": text_tokens}, f, indent=2)
        print(f"clap_text.onnx hotový (text_tokens={text_tokens})")

    else:
        raise ValueError(f"Neznámy režim exportu: {mode}")
    return 0


# ---------------------------------------------------------------------------
# CLI mini-test (bez GUI):  python core_analyzer.py subor.mp3 "popis 1" "popis 2"
# ---------------------------------------------------------------------------
def _cli() -> int:
    args = sys.argv[1:]

    # interné fázy exportu (volá sám seba ako subprocess)
    export_flags = ("--_export-audio", "--_export-text",
                    "--_export-audio-dynamo", "--_export-text-dynamo")
    if len(args) == 2 and args[0] in export_flags:
        mode = args[0][len("--_export-"):]          # audio | text | audio-dynamo | text-dynamo
        return _internal_export(mode, args[1])

    if len(args) < 3:
        print(__doc__)
        return 1
    path, descs = sys.argv[1], sys.argv[2:]
    if not os.path.isfile(path):
        print(f"Súbor neexistuje: {path}")
        return 1

    analyzer = ClapAnalyzer()
    analyzer.load()
    result = analyzer.analyze_file(path, descs)

    print(f"\nSúbor:  {path}")
    print(f"Backend: {result.backend}")
    print(f"Analyzovaných okien: {result.segments_used} × {CLIP_SECONDS} s")
    print("Poradie kandidátov:")
    for desc, p in result.ranking:
        marker = "  ← NAJLEPŠIE" if desc == result.best_description else ""
        print(f"  {p * 100:5.1f} %  {desc}{marker}")
    margin_note = ("spoľahlivé" if result.margin > 0.15
                   else "nejednoznačné – skúste viac/lepšie popisy")
    print(f"\nNáskok pred 2. kandidátom: {result.margin * 100:.1f} %bodov ({margin_note})")

    msg = write_metadata(path, result.best_description, result.confidence)
    print(f"\nZapísané do metadát ({msg}): ‘{result.best_description}’")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
