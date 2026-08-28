# Changelog — Analyzátor zvukových súborov

Všetky významnejšie zmeny sú tu zdokumentované **čo sa zmenilo a prečo**.
Formát podľa [Keep a Changelog](https://keepachangelog.com), verzie SEMVER.

---

## [0.9.0] – 2026-08-28 – Rust GUI: grafické okno k Rust verzii

Prečo: Rust verzia bola doteraz len príkazový riadok; používateľ si
ju overil na svojom stroji (funguje) a prial si GUI ako má Python
verzia. Zdrojový kód CLI bol preto preštruktúrovaný na spoločnú
knižnicu (pipeline) + dva programy: CLI aj GUI zdieľajú úplne rovnakú
logiku, takže výsledky zostávajú identické.

### Pridané
- **`analyzator-gui.exe`** – grafické okno (egui/eframe, bez ďalších
  závislostí; väzí len na štandardných Windows knižniciach + OpenGL):
  zoznam súborov so stavmi (✔/⚠/✖/⚡/…), editovateľný zoznam popisov
  (predvolených 69, načítajú sa z popisy.txt pri programe), nastavenia
  (počet okien, prah istoty, vlákna, preskočenie AI podľa názvu,
  istota do popisu), tlačidlá Analyzovať/Zastaviť, progress bar a log.
  Priečinok so zvukmi možno **potiahnuť myskou priamo do okna** alebo
  vybrať dialógom. AI model hľadá pri programe, o úroveň vyššie
  (odporúčané umiestnenie v hlavnom priečinku programu) aj v CWD.
  Analýza beží vo vlastnom vlákne – okno nezamrzá a beh ide zastaviť.
- **Refactor: `src/lib.rs` + `src/pipeline.rs`** – celá analýza
  (model, príprava, inference, pravidlá, zápisy) presunutá do
  knižnice so správami (Event) a zastavením (AtomicBool). CLI
  (main.rs) aj GUI (gui.rs) sú nad ňou tenké obaly.
- Windows balík v0.9.0: pribudol analyzator-gui.exe (19,4 MB),
  README-RUST prepísané (GUI ako hlavný spôsob, CLI pre hromadné
  spúšťanie). Bez prehrávania zvukov (na kontrolu zvuku slúži Python
  verzia) – zamietnuté: zabudovávanie prehrávača (veľkosť a zložitosť
  balíka by výrazne vzrástli bez úžitku pre testovacie využitie).

### Overené
- CLI po refactori: build bez varovaní + funkčný test s modelom
  (rovnaký výsledok ako pred zmenou).
- GUI: kompilácia bez chýb (host aj cross-target), exe má subsystém
  „Windows GUI“ (nepreblikne mu konzola) a len štandardné DLL.
- GUI runtime vizuálne testovať v sandboxe nemožno (bez displeja) –
  overí používateľ; prípadné chyby opravím.

## [0.8.1] – 2026-08-28 – Oprava TEST-RUST.bat (okno len prebliklo)

Prečo: po dvojkliku na TEST-RUST.bat čierne okno okamžite zmizlo.
Príčina: v texte výpisu „echo (napríklad C:\…\Zvuky):“ bola zátvorka
„)“ VNÚTRI príkazového bloku if (…) – Windows ju vyhodnotil ako koniec
bloku → chybové ukončenie skriptu ešte pred akýmkoľvek výpisom.
Program (exe) bol v poriadku, chybný bol len spúšťač.

### Opravené
- TEST-RUST.bat prepísaný bez zátvoriek v textoch výpisov (skript
  ich nesmie obsahovať vnútri blokov if); pridaný záverečný výpis
  „Hotovo“. Nástroj kontroly: žiadny echo riadok so zátvorkou.
- Nový Windows balík v0.8.1 (exe a knižnice nezmenené – rovnaký
  program 0.8.0, len spúšťač); release v0.8.0 zmazaná, zostáva
  jediná aktuálna.

## [0.8.0] – 2026-08-28 – Odstránené učenie sa (slová aj zvukové vzory)

Prečo: používateľ hodnotil výsledky učenia ako nezmyselné – funkcia
spájala náhodné slová z názvov súborov s popismi a tým skresľovala
istotu. Rozhodol sa ju odstrániť celú (slová aj zvukové vzory).
Zamietnutá alternatíva: prísnejšie filtre učenia (napr. vyšší počet
opakovaní) – zamietnuté, používateľ chce jednoduché a predvídateľné
správanie bez skrytých asociácií.

### Odstránené (Python aj Rust – verzie zostávajú rovnocenné)
- **Učenie slov z názvov** (`naucene_spojenia.json`) – slovo → popis
  sa už nepamätá ani nepoužíva; tlačidlo 🧠 Naučené a jeho dialóg
  z GUI boli odstránené.
- **Zvukové vzory** (`naucene_vzory.npz`) – frekvenčné odtlačky sa
  už neukladajú ani neporovnávaťajú; zmizla poznámka
  „🧠 podobný naučenému zvuku“.
- Staré súbory `naucene_*` na disku zostávajú len ležať – program
  ich ignoruje (možno zmazať).
- Súvisiace konštanty z oboch verzií preč (AUDIO_SIM_MIN/BOOST,
  PATTERN_MAX_PER_LABEL).

### Zachované (neovplyvnené odstránením)
- **Posilnenie názvom súboru** (× 1,3, strop 99 %) – naďalej funguje,
  ale IBA pri priamej zhode slova z názvu v texte popisu
  („rain“ v „heavy rain on roof“); žiadne naučené asociácie.
- **Preskočenie AI pri jednoznačnom názve** – len priama zhoda
  (≥ 2 slová alebo 1 dlhé slovo ≥ 5 znakov na jediný popis).
- Overené testmi: priame zhody fungujú, všeobecné slová (final, mix)
  nič nespúšťajú. Syntax oboch súborov Pythonu OK; Rust build bez
  varovaní + funkčný test s modelom (žiadne naucene_* súbory).
- Windows balík prebalený (v0.8.0) – exe bez učenia, README-RUST
  aktualizované.

## [0.7.3] – 2026-08-28 – Oprava TEST-RUST.bat (model sa nenašiel)

Prečo: bat hľadal AI model v adresári spúšťača, ale keď bol priečinok
umiestnený v hlavnom priečinku programu (podľa návodu), model je
o úroveň vyššie (`..\models`) – program teda skončil chybou a Rust
verzia na používateľovom stroji neprešla. Nová release v0.7.3 nahrádza
pôvodnú v0.7.1 (zmazaná, aby zostal jediný funkčný odkaz).

### Opravené
- **TEST-RUST.bat**: hľadá model v poradí (1) `..\models\clap_…
  _onnx` – priečinok v hlavnom adresári programu (odporúčané;
  zároveň sa tak použijú aj spoločné naučené dáta `naucene_*`),
  (2) priečinok `model` pri programe, (3) aktuálny adresár.
  Pridané: kontrola, že exe existuje (ak ho zmazal antivírus,
  vypíše zrozumiteľné upozornenie) a `chcp 65001` pre správne
  zobrazenie slovenskej diakritiky vo výpise.
- Zdrojové podoby súborov balíka verzované v `rust/windows-bundle/`.

## [0.7.2] – 2026-08-28 – DIAGNOZA.bat: automatická diagnostika pre používateľa

Prečo: používateľovi sa po stiahnutí Rust verzie pokazil aj hlavný
(Python) analyzátor; na diaľku potrebujem presný stav jeho inštalácie
(screenshoty nevidím, text chyby je potrebný). 

### Pridané
- **`DIAGNOZA.bat`** – dvojklikom spustiteľná diagnostika (ASCII+CRLF):
  vypíše zoznam súborov programu, AI modelu, Python prostredia .venv,
  otestuje importy knižníc (onnxruntime vrátane zoznamu providerov,
  numpy, soundfile, transformers, PyQt6), bool DLL súborov v .venv
  (či ich antivírus nezmazal), naučené dáta, rust priečinok a najmä
  **históru blokácií Windows Defender** (Get-MpThreatDetection).
  Výsledok zapíše do `DIAGNOZA-VYSLEDOK.txt` a otvorí ho v poznámkovom
  bloku – používateľ ho skopíruje a pošle späť.

## [0.7.1] – 2026-08-28 – Windows balík Rust verzie na stiahnutie + návod v repo

Prečo: používateľ nemá kompilátor a chce Rust verziu testovať dvojklikom
ako ostatné časti programu; návod doteraz nebol súčasťou repa.

### Pridané
- **Windows balík v GitHub Releases** (príloha `analyzator-rs-windows.zip`
  pri tagu `v0.7.1`): `analyzator-rs.exe` (13,7 MB, zostavené pre
  64-bit Windows), `onnxruntime.dll` + `onnxruntime_providers_shared.dll`
  ( oficiálny NuGet balík Microsoft.ML.OnnxRuntime.DirectML 1.24.4 –
  jediná súčasná oficiálna distribúcia DirectML buildu; samostatné
  „directml-win-x64“ ZIPy Microsoft v releases už nezverejňuje),
  `TEST-RUST.bat` (spúšťač – dvojklik alebo potiahnutie priečinka
  myskou; ASCII + CRLF podľa konvencie), `popisy.txt` (rovnakých 69
  predvolených popisov ako GUI), `README-RUST.txt` (krátky návod SK).
  Exe závisí len od štandardných Windows knižníc (kontrola objdump) –
  nič netreba inštalovať.
- **`navod-analyzator.html` v repo** – kompletný návod (inštalácia,
  práca v GUI, riešenie problémov) + nová sekcia 09 o Rust verzii
  (kde stiahnuť, ako spustiť, spoločné naučené dáta).

### Zmenené
- `rust/Cargo.toml`: ONNX Runtime spôsob pripojenia je teraz výslovná
  voľba – `default = ["ort-static"]` (vkompilované binárky, Linux/
  vývoj) a `--no-default-features --features "ort-directml,
  ort-load-dynamic"` pre Windows build (exe načíta onnxruntime.dll
  za behu). Prečo: ort pre `x86_64-pc-windows-gnu` nemá predkompilované
  binárky, a DirectML/GPU takto funguje s oficiálnou DLL od Microsoftu.
- `rust/src/model.rs`: DirectML_executionProvider opravený názov
  (build s `ort-directml` predtým nešiel skompilovať) + pokus o DirectML
  s automatickým pádom na CPU, ak GPU/DirectML nie je k dispozícii
  (rovnaké správanie ako Python verzia; overené testom s CPU-only
  knižnicou).
- Overenie exe logiky: rovnaký zdrojový kód bol otestovaný na Linuxe
  s dynamickou knižnicou (načítanie DLL za behu, CPU fallback pri
  chýbajúcom DirectML, celý beh vrátane zápisu metadát) – Windows exe
  je z tohto kódu zostavené cross-kompilátorom mingw-w64.

## [0.7.0] – 2026-08-25 – Rust verzia: CLI alternatíva s preukázanou paritou

Prečo: používateľ si prial vyskúšať **alternatívnu testovaciu verziu v Ruste**
(rýchlejšie, jeden spustiteľný súbor bez Pythonu). Python verzia sa
**meníť nemá** – obe existujú vedľa seba a dávajú rovnaké výsledky.

### Pridané
- **`rust/` – kompletný port v Ruste** (CLI, zatiaľ bez GUI):
  `analyzator-rs PRIECINOK [--popisy SÚBOR] [--segments N] [--min-istota P]
  [--model-dir ADRESÁR] [--bez-preskocenia] [--istota-do-popisu] [--vlakien N]`.
  Rovnaký CLAP model (ONNX export z Pythonu, DirectML ak je k dispozícii),
  rovnaké konštanty, rovnaký formát metadát (ID3v2.4 COMM / DESCRIPTION / ICMT),
  rovnaké prirodzené triedenie súborov. Závislosti: ort 2.0-rc, symphonia
  (dekód wav/mp3/flac/ogg), rustfft, tokenizers, rayon.
- **Numerická parita preukázaná** proti Pythonu (transformers 5.15.1,
  onnxruntime 1.29, CPU): log-mel spektrogram **max rozdiel 0,0000 dB**
  (korelácia 1,00000000); embeddingy okien kosínus 1,0; pravdepodobnosti
  sa líšia max o **6,7e-07** (úroveň presnosti float32); istoty aj boosty
  (názov / naučený vzor / viac popisov) sa zhodujú na 0,00e+00.
  Prečo je to možné: mel filterbank nie je počítaný vzorcom, ale
  **zapracovaný bajtovo presne z Pythonu** (`src/mel_slaney.f32`, 513×64).
  Zistili sme pritom, že transformers 5.x pre tento model používa
  `truncation="rand_trunc"` + **Slaney** banku (nie „fusion“ + HTK ako
  v starších verziách) – Rust zrkadlí skutočné správanie 5.x.
- **Naučené dáta kompatibilné OBOJSTRANNE**: `naucene_spojenia.json` a
  `naucene_vzory.npz` píše/číta Rust aj Python vo formáte numpy `<U…`
  (UTF-32LE znaky doplnené nulami, bez dĺžkovej predpony) – overené
  zápismi v oboch smeroch a načítaním v druhej verzii. Pozor: Rust CLI
  hľadá naučené dáta v aktuálnom adresári (CWD), Python pri skripte.
- **Ladiace prepínače** (pre overenie parity, bežne sa nepoužívajú):
  `--dump-mel SÚBOR`, `--dump-emb SÚBOR`, `--json SÚBOR` (celé poradie
  vrátane pravdepodobností), `--debug-audio` (RMS vzoriek po dekóde).
- **Rýchlosť (sandbox, CPU, 4×60 s + 8 s súbory)**: Python 5,26 s →
  Rust 3,1 s (1 vlákno) / 2,7 s (4 vlákna) ≈ **1,7–1,9× rýchlejšie**.
  Nástroj hashuje/dekóduje paralelne (rayon); inference beží v ONNX
  Runtime rovnako ako v Pythone, takže s GPU (DirectML) sa zrýchli
  u oboch a náskok Rustu zostáva hlavne v dekóde a mel spektrograme.

### Zmenené
- `rust/` má vlastný `.gitignore` (target/ – 38 MB binárka sa do repa
  neposúva; zostavuje sa príkazom `cargo build --release`).

## [0.6.0] – 2026-08-24 – Názvy súborov, učenie sa, viac popisov, prahy istoty

Požiadavky používateľa: názov súboru má pomáhať pri určovaní istoty;
nízko istotné popisy nezapisovať a staré zmazať; AI preskočiť, keď názov
jasne určuje popis; súbory vyhodnocovať paralelne (so testom výkonu);
program sa má „učiť" nové kategórie z názvov aj z frekvenčných vzorov;
dlhšie nahrávky s viacerými zvukmi majú dostať viac popisov; učiť sa
iba anglické slová.

### Pridané
- **Názov súboru posilní istotu × 1,3** (60 % → 78 %, 70 % → 91 %; strop
  99 %). Násobenie namiesto „+30 %" zvolené zámerne – istota nikdy
  nepresiahne 100 % a silné výsledky posilní viac. Funguje aj cez
  naučené spojenia (napr. „whoas" → whoosh popis).
- **Zvukový vzor posilní istotu × 1,2**: embedding zvuku (= frekvenčný
  odtlačok) sa porovná s naučenými vzormi; podobnosť ≥ 80 % = dôkaz.
- **Preskočenie AI podľa názvu**: ak ≥ 2 slová z názvu (alebo 1 dlhé
  slovo ≥ 5 znakov, alebo naučené spojenie videné 2×) jednoznačne sedia
  na práve JEDEN kandidátny popis, popis sa zapíše bez AI. Pri remíze
  AI beží (radšej spoľahlivo). Ak sa všetky súbory vyriešia názvom,
  model sa vôbec nespúšťa (uveďte: „⚡ Podľa názvov netreba AI").
- **Učenie sa slov** (`naucene_spojenia.json`): slovo z názvu → popis,
  kam súbor skončil; len isté výsledky (nad prahom zápisu); slová už
  obsiahnuté v popisoch sa neučia (nič nové by to nedali); max 3 popisy
  na slovo. Iba ASCII/anglické slová – slová s diakritikou sa ignorujú.
- **Učenie sa zvukových vzorov** (`naucene_vzory.npz`): embedding +
  popis každého istého výsledku; max 30 vzorov na popis; nahrávky s
  viacerými popismi sa ako vzor neukladajú (nečistý vzor).
- **Viac popisov pre dlhšie nahrávky**: popis sa pridá, ak vyhrá aspoň
  v jednom 10 s okne a jeho priemerná pravdepodobnosť má ≥ 40 % víťaza
  (výrazne prítomný zvuk, nie náhoda). Max 3 popisy, zápis „a + b".
- **Dialóg 🧠 Naučené** – prehľad spojení a vzorov + mazanie
  (vybraných / všetkých). Počet vidno na tlačidle.
- Nový stav riadku **„Nízka istota"** (oranžová) + rozpis v záverečnej
  správe (✔ hotovo, ⚡ podľa názvu, ⚠ nízka istota, ✖ chyby).
- Widgety: „Nezapisovať popis pod istotu [50 %]" a „⚡ Preskočiť AI…".

### Zmenené
- **Prah istoty (východiskovo 50 %)**: pod ním sa popis NEzapíše
  („radšej nič ako nezmysel"). Starý popis sa pritom zmaže, ale LEN ak
  vyzerá, že ho napísala táto appka (obsahuje „(istota" alebo sedí na
  kandidátny popis) – ručne napísané/cudzie popisy ostanú nedotknuté.
- Worker: preload(N+1) pred analyzou N (skutočné prekrytie CPU/GPU),
  2 dekódovacie vlákna, hĺbka prefetchu 3 (paralelné dekódovanie).
- `AnalysisResult`: nové polia `name_boosted`, `pattern_boosted`,
  `additional`, `audio_embedding`.

### Testy (sandbox, CPU)
- Jednotkové: kľúčové slová (diakritika/stoplisty), skip pravidlá a
  remízy, učenie slov, vzory (podobný zvuk 0,90 ✓ / cudzí 0,03 ✓),
  zápis/mazanie popisu WAV, matematika boostov.
- analyze_file na syntetických embeddingoch: zmiešaná nahrávka → 2
  popisy ✓; čistá → 1 popis ✓; boosty ×1,3/×1,2 ✓.
- Worker e2e (so stub modelom): skip podľa názvu ✓, 30 % → nezapísané
  + starý 20 % popis zmazaný ✓, učenie slov aj vzorov ✓, druhý beh:
  nový súbor „sirenka3" preskočil AI podľa naučeného spojenia ✓.
- Výkon: MP3 dekód prvého súboru 1,7 s (jednorazový warmup), ďalšie
  ~60 ms; paralelné dekódovanie 1,0–1,1× na rýchlom stroji – hlavné
  zrýchlenia plynú z 1 dekódu/súbor (0.4.0), preskakovania AI a warmupu
  bežiaceho popri GPU.

### Zamietnuté nápady
- **Slovenské slová bez diakritiky** („zvonenie") – od anglických sa
  nerozoznávajú bez slovníka; filtruje sa diakritika + stoplist,
  používateľ potvrdil výhradne anglické názvy súborov.
- **Rozpoznávanie reči** (vyslovené slová v nahrávke) – samostatná
  veľká funkcia (iný model, ~1 GB); „viac slov v nahrávke" riešime
  viacpopisovým pravidlom vyššie.

---

## [0.5.1] – 2026-08-24 – Oprava: AKTUALIZUJ.bat bez gitu

Hlásenie používateľa: „'git' is not recognized as an internal or external
command". Príčina: `AKTUALIZUJ.bat` spoliehal na `git` — ale používateľ
nemá git nainštalovaný (appka bola stiahnutá ako ZIP z webu, priečinok
nie je git repozitár).

### Opravené
- **`AKTUALIZUJ.bat` má nový bezgitový rezim:** ak git chýba (alebo
  priečinok nie je git repozitár), stiahne sa ZIP najnovšej verzie
  z GitHubu a rozbalí cez PowerShell (súčasť každého Windows — nič
  sa nemusí inštalovať). Súbory sa prepíšu, okrem:
  `.venv` (nainštalované prostredie), `models` (stiahnutý AI model),
  `__pycache__`. Ak je git k dispozícii, použije sa rýchly `git pull`.
- **Bezpečné samoupdatovanie bat súboru:** bežiaci `.bat` sa nemôže
  korektne prepísať sám za sebou — nová verzia sa odloží ako
  `AKTUALIZUJ.bat.new` a nainštaluje pri najbližšom spustení.
- **Kontrola knižníc po update:** `AKTUALIZUJ.bat` teraz po stiahnutí
  spustí `pip install -r requirements.txt` — keby nová verzia appky
  potrebovala novú knižnicu, dorobí sa sama (inak rýchla no-op kontrola).

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
