# Analyzátor zvukových súborov 🎧🤖

Desktopová aplikácia (Python + PyQt6), ktorá **automaticky rozpozná obsah zvukových
súborov a zapíše AI popis priamo do ich metadát** — určené pre **AMD GPU
(Radeon RX 7700 XT)** s akceleráciou cez **DirectML**.

Aplikácia porovná každý súbor so zoznamom kandidátskych popisov (vy definujete,
jeden na riadok) pomocou modelu **LAION-CLAP** (`laion/clap-htsat-unfused`),
vyberie najlepšiu zhodu a zapíše ju do tagov:

| Formát | Kam sa popis zapíše |
|--------|---------------------|
| MP3    | ID3 tag **COMM** (Comment) |
| OGG    | Vorbis Comment **DESCRIPTION** |
| FLAC   | Vorbis Comment **DESCRIPTION** |
| WAV    | RIFF INFO **ICMT** |

## Ako to funguje

1. **Načítanie audia** – `librosa` prevedie zvuk na 48 kHz mono a vezme
   stredných 10 sekúnd (na tejto dĺžke bol CLAP trénovaný).
2. **AI inference** – audio aj textové popisy sa premenia na embeddingy
   (vektory). Bežia cez **ONNX Runtime**:
   - na Windows s AMD GPU: `DmlExecutionProvider` (**DirectML**),
   - automatický fallback na CPU, ak DirectML zlyhá alebo chýba.
3. **Skórovanie** – cosine similarity medzi zvukom a každým popisom →
   softmax → najlepší popis + percentuálna istota.
4. **Zápis metadát** – `mutagen` zapíše popis do príslušného tagu.

> Rozpoznaný popis je **vždy jeden z kandidátov**, ktoré zadáte — aplikácia
> teda klasifikuje do vášho zoznamu, nevygeneruje ľubovoľný text.

### Overené výsledky (CPU referenčný test)

Numerická parita exportu: **cosine = 1.000000** medzi ONNX a pôvodným
PyTorch modelom. Rýchlosť ≈ **0,5 s / súbor** na CPU — s DirectML ešte
rýchlejšie. Test syntetických zvukov: sínusový tón → „a pure sine tone
beeping“ (99,9 %), biely šum → „static white noise“ (96 %).

## Inštalácia (Windows + AMD GPU)

```bat
:: 1) Python 3.10 – 3.13 (python.org) a potom v priečinku projektu:
python -m venv .venv
.venv\Scripts\activate

:: 2) závislosti
pip install -r requirements.txt

:: 3) spustenie
python main_gui.py
```

**Prvé spustenie** stiahne model z HuggingFace (~600 MB) a jednorazovo
vyexportuje ONNX grafy do `models/` (2–5 minút). Každé ďalšie spustenie
je rýchle — načítajú sa existujúce ONNX súbory.

> Prvý štart vyžaduje internet (download modelu). Ďalšie už nie.

## Použitie

1. Kliknite na **Pridať súbory…** / **Pridať priečinok…** (alebo presuňte
   súbory myšou do okna). Výber priečinka prehľadá **aj všetky podpriečinky**
   (rekurzívne, s prirodzeným triedením `file2 < file10`; skryté a systémové
   priečinky sa preskakujú).
2. Vpravo vyberte **predvoľbu popisov** (SFX pre film/reklamu ~170 popisov,
   pôvodný zoznam, rýchly štart) a pole podľa potreby upravte — jeden popis
   na riadok, po anglicky (CLAP je trénovaný na anglických textoch), riadky
   začínajúce `#` sú sekcie a do analýzy sa nezapočítavajú. Napr.:
   ```
   heavy rain falling on roof
   dog barking outdoors
   car engine revving
   birds singing in the forest
   ```
3. **Úseky na súbor (presnosť)** — koľko 10-sekundových okien sa v každom
   súbore analyzuje a spriemeruje (1 = len stred, rýchle; **4 = odporúčané**;
   8 = maximum). Zvyšuje presnosť pri dlhších súboroch s premenlivým obsahom.
4. (Voliteľné) zaškrtnite *„Zapísať aj istotu do popisu"*.
5. **▶ Spustiť AI analýzu** — priebeh vidno v tabuľke, progress bare
   (**percentá + odhad zostávajúceho času**) a logu; GUI počas behu
   nijako nezamrzá (práca beží vo vlákne na pozadí).

V stĺpci **Detail** u hotového súboru uvidíte okrem istoty aj **náskok**
pred druhým kandidátom — malý náskok (napr. +5 %) znamená nejednoznačný
výsledok a znamenie, že zoznam popisov treba spresniť.

Popisy sa dajú po dokončení prečítať napr. vo vlastnostiach súboru
(Windows Explorer → Podrobnosti → Komentár) alebo v ľubovoľnom tagery.

## Štruktúra projektu

```
├── main_gui.py          # PyQt6 rozhranie + QThread worker
├── core_analyzer.py     # CLAP engine: ONNX/DirectML, audio, skórovanie, tagy
├── requirements.txt     # závislosti
└── models/              # (vygenerované) ONNX grafy – vzniknú pri prvom spustení
```

### Test bez GUI

```bat
python core_analyzer.py zvuk.mp3 "heavy rain on roof" "dog barking" "car engine"
```

Vypíše poradie všetkých kandidátov so skóre a zapíše najlepší do metadát.

## Riešenie problémov

| Problém | Príčina / riešenie |
|---|---|
| V logu `⚠ DirectML zlyhalo → CPU` | Nainštalujte `onnxruntime-directml` a **odinštalujte** bežný `onnxruntime` (`pip uninstall onnxruntime`), sú konfliktné. Aplikácia pritom funguje ďalej na CPU. |
| Prvý štart trvá dlho | Sťahuje sa model (~600 MB) + jednorazový ONNX export (2–5 min). Platí len raz, výsledok je v `models/`. |
| V logu „textové embeddingy pôjdu cez torch (CPU)" | Export textového ONNX grafo zlyhal (málo RAM) – aplikácia automaticky použije torch fallback, výsledky sú identické. Pri ďalšom štarte sa export automaticky zopakuje cez alternatívny (dynamo) exporter. |
| MP3/OGG sa nenačítajú | `soundfile` (libsndfile ≥ 1.1) ich zvláda; prípadne doplňte `ffmpeg` do PATH. |
| Nízka istota | Pridajte viac rozličných popisov, formulujte konkrétnejšie (v angličtine), alebo skúste iný úsek — analyzuje sa stredných 10 s. |
| Chcem analyzovať celý súbor, nie 10 s | Upravte `CLIP_SECONDS` v `core_analyzer.py`. |

## Rust verzia (testovacia alternatíva)

Vedľa Python verzie existuje rýchlejšia **Rust verzia** (CLI, bez okna) –
dáva **rovnaké výsledky** (číselne overená parita) a je cca **1,7–1,9×
rýchlejšia**. Python verzia zostáva hlavná.

- Hotový Windows balík (`analyzator-gui.exe` – grafické okno,
  `analyzator-rs.exe` – príkazový riadok, knižnice; nič sa neinštaluje):
  [Releases – analyzator-rs-windows.zip](https://github.com/stemja1/Analizator-zvukovych-suborov/releases) –
  rozbaľte do hlavného priečinka programu; GUI spustíte dvojklikom na
  `analyzator-gui.exe`, CLI cez `TEST-RUST.bat`.
- Zdrojové kódy: `rust/` (`cargo build --release`; GPU build pozri `rust/README.md`).
- Viac detailov: sekcia 09 v `navod-analyzator.html`.

## Licencia

MIT (pozri `LICENSE`).
