# Changelog — Analyzátor zvukových súborov

Všetky významnejšie zmeny sú tu zdokumentované **čo sa zmenilo a prečo**.
Formát podľa [Keep a Changelog](https://keepachangelog.com), verzie SEMVER.

---

## [0.5.0] – 2026-08-24 – Prehrávanie súborov + oprava prefetchu

Požiadavka používateľa: pri kontrole výsledkov chýhal spôsob, ako súbor
rýchlo vypočuť — človek musel riadok ručne otvárať v Exploreri.

### Pridané (main_gui.py)
- **Prehrávanie: dvojklik na riadok v tabuľke prehrá súbor** (od začiatku).
  Pod tabuľkou svieti štítok s pozíciou `[mm:ss / mm:ss]` a názvom súboru,
  po pauze sa zmení ▶/⏸.
- **Klávesové skratky** (aktívne, keď je kurzor v tabuľke súborov —
  aby medzerník nerušil písanie popisov):
  - `Medzerník` = prehrať/pauza (nič nehrá → prehrá vybratý riadok)
  - `Ctrl+←/→` alebo `,` / `.` = posun −5 s / +5 s
  - `R` = prehrať od začiatku, `Esc` = stop
- `QtMultimedia` sa importuje „opatrne“ — ak modul chýba, appka beží
  normálne, len neprehráva (analytické funkcie ostávajú intact).
- Mazanie riadkov počas prehrávania udržuje správny riadok (posun indexov);
  zatvorenie appky prehrávanie najprv zastaví.

### Opravené (core_analyzer.py — prefetch z 0.4.0)
- Prefetch mal jednu „priehradku“: keď worker predložil súbor N+1 ešte
  pred analyzou súboru N, hocijaký čakajúci future sa zahodil → súbor sa
  dekódoval 2× (overené testom: 12 dekódov na 6 súborov). Teraz: zásobník
  max. 2 prefetchov (aktuálny + ďalší), `_take_windows` odoberá len
  vlastný súbor. Worker volá `preload(N+1)` PRED analysou N — dekód N+1
  beží počas GPU práce na N (skutočné prekrytie CPU/GPU).
- Test: 6 súborov × 4 okná = presne 6 dekódovaní (1/súbor) ✓.
- Audio ONNX graf s dynamickým batchom overený: batch(4) vs 4× per-row —
  cosine podobnosť embeddingov 1.000000 ✓.

---

## [0.4.0] – 2026-08-24 – Výkon: GPU pipeline (preload + dávkovanie)

Problém: používateľ hlásil nízke využitie GPU pri spracovaní súborov.
Príčina: na 1 súbor približne 300–600 ms CPU práce (dekód, resampling,
spektrogram) vs. len ~30–60 ms GPU inference — GPU čakal na CPU.

### Zmenené
- **`core_analyzer.py` — analyzuj súbor JEDNYM dekódovaním.**
  Predtým: N úsekov = N volaní `librosa.load` (MP3 s offsetom sa pri každom
  volaní dekódoval od začiatku súboru!). Teraz: súbor sa raz načíta celý
  (mono 48 kHz) a okná sa vykroja z pamäte. Pre extrémne dlhé súbor
  (> 20 min) fallback na postupné načítanie okien (kvôli RAM).
- **`embed_audio_batch` — všetky úseky súboru v JEDNOM GPU volaní.**
  Audio ONNX graf má teraz dynamický batch (rovnako ako textový).
  Predtým 4 úseky = 4 GPU volania, teraz 1 volanie s batch=4.
  Menšie preklady CPU↔GPU, vyššie využitie GPU.
- **Preload/prefetch (pipelining):** kým GPU analyzuje súbor N, vlákno
  na pozadí dopredu dekóduje súbor N+1 → CPU a GPU práca sa prekrýva.
- `_fit_audio_shape` prepísané pre podporu batch dimenzie.
- DirectML provider má explicitné `device_id: 0`.

### Pridané
- `AnalysisResult.decode_time` / `infer_time` — rozpad času v detaile
  („dekód 0,4 s | GPU 0,1 s") + počítadlá dekódovaní a GPU volaní.
- Tento `CHANGELOG.md` (požiadavka používateľa).

### Zamietnuté nápady (zamietnuté a prečo)
- **Dekódovanie audia na GPU** — pre samostatné zvukové súbory neexistuje
  praktická cesta: hardvérové dekodéry (AMD VCN, NVIDIA NVDEC) dekódujú
  iba audio stopu vo video kontajneroch; ffmpeg/librosa/soundfile dekódujú
  vždy na CPU. Dosiahnutý efekt: 1 dekód/súbor + prekrytie s GPU behom.
- **Spektrogram (STFT) cez `torch-directml`** — balík existuje, ale jeho
  pokrytie torch operácií (najmä `torch.stft`) je nespoľahlivé naprieč
  verziami driverov; riziko pádov prevyšuje zisk (STFT je ~30 ms/okno).

---

## [0.3.1] – 2026-08-24 – .bat skripty pre netechnikov

### Pridané
- `SPUSTI.bat` — spustenie aplikácie na dvojklik (pri prvom spustení sám
  vytvorí venv a nainštaluje závislosti). Dôvod: používateľ nechce písať
  príkazy do cmd.
- `AKTUALIZUJ.bat` — `git pull` na dvojklik.

## [0.3.0] – 2026-08-24 – Presnosť, ETA, predvoľby + oprava priečinkov

### Opravené
- **Bug (hlásil používateľ): výber priečinka nenačítal zvuky ani v
  podpriečinkoch.** Príčina: `os.listdir` (iba najvyššia úroveň) na všetkých
  troch miestach (tlačidlo, auto-spracovanie, drag&drop). Oprava: spoločný
  rekurzívny sken `os.walk` — vrátane veľkých/malých prípon (.WAV),
  diakritiky v názvoch, prirodzeného triedenia (file2 < file10),
  preskakovania skrytých a systémových priečinkov (`$RECYCLE.BIN`…).
  Keď sa nič nenájde, log vypíše, aké prípony v priečinku sú (napr. .m4a).

### Pridané (prečo: používateľ hlásil nepresnú analýzu)
- **Multi-oknová analýza (1–8 úsekov, default 4):** každý súbor sa analyzuje
  v N rovnomerne rozmiestených 10 s oknách a embeddingy sa spriemerujú.
  Predtzel dlhší súbor zachytil len svoj stred.
- **Náskok (margin)** pred 2. kandidátom v detaile — malý náskok = výsledok
  je nejednoznačný a zoznam popisov treba spresniť.
- **Predvoľby kandidátskych popisov:** 🎬 SFX pre film/reklamu (~170 popisov
  v 8 sekciách), 🎚 pôvodný zoznam (~69), 🧪 rýchly štart. Riadky `#` sú
  sekcie a do analýzy sa nezapočítavajú.
- **Progress: percentá + ETA** (odhad zostávajúceho času).
- Predvýpočet textových embeddingov na začiatku behu (cache — prvému
  súboru nebude nafúknutý čas prípravy 170 popisov).

### Zmenené
- Textový ONNX graf: dynamický batch + embedovanie v dávkach po 32
  (dôvod: jednorazový beh batch=170 zabil proces na strojoch s málo RAM).

## [0.2.0] – 2026-08-23 – Robustnosť

### Opravené
- Prázdne chybové hlášky: výnimky ako `NoBackendError` majú prázdny text —
  detail v tabuľke teraz zobrazí aspoň názov výnimky.
- Log informuje o preskočených súboroch pri pridávaní (neexistujú/nepodporované).
- WAV metadata: mutagen ≥ 1.46 píše do WAV ID3 namiesto RIFF INFO — pridaný
  vlastný bezpečný RIFF INFO (ICMT) chunk writer (zachová audio dáta,
  idempotentný zápis, prepisuje len LIST INFO chunk).

## [0.1.0] – 2026-08-23 – Prvá verzia

- LAION-CLAP (`laion/clap-htsat-unfused`) cez ONNX Runtime s DirectML
  (`DmlExecutionProvider`) a fallbackom na CPU (pre AMD GPU bez CUDA).
- Jednorazový ONNX export v samostatných subprocessoch (nízka RAM),
  pri zlyhaní TorchScript exportu automatický retry cez dynamo exporter,
  pri zlyhaní textového ONNX fallback na torch (identické výsledky,
  overené cosine = 1.000000).
- PyQt6 GUI: tabuľka stavov, QThread worker (GUI nezamrzá), drag&drop,
  progress, log; CLI test bez GUI.
- Zápis metadát: MP3 → ID3 COMM, OGG/FLAC → Vorbis DESCRIPTION,
  WAV → RIFF INFO ICMT.
