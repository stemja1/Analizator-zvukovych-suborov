# Analyzátor – Rust verzia (testovacia alternatíva)

Rovnaká logika ako Python verzia, spustená ako jeden program bez Pythonu.
Dva spustiteľné súbory: `analyzator-gui` (grafické okno) a
`analyzator-rs` (príkazový riadok) – obe používajú spoločnú knižnicu
(`lib.rs` + `pipeline.rs`). Python verzia zostáva hlavná.

## Zostavenie
```
cd rust
cargo build --release
```
Výsledok: `target/release/analyzator-rs` (~38 MB, obsahuje ONNX Runtime).

DirectML (GPU na Windows): `cargo build --release --features ort-directml`.

## Použitie – GUI
Dvojklik na `analyzator-gui` (okno: súbory vľavo, popisy a nastavenia
vpravo, log dole; priečinok so zvukmi možno potiahnuť do okna).

## Použitie – CLI
```
analyzator-rs PRIECINOK [prepínače]
  --popisy SÚBOR         zoznam popisov (1 na riadok, predvolene popisy.txt)
  --segments N           počet 10 s okien na súbor (predvolene 4)
  --min-istota P         prah istoty 0–1 (predvolene 0.5)
  --model-dir ADRESÁR    ONNX modely (clap_audio.onnx, clap_text.onnx,
                         tokenizer.json, export_meta.json)
  --bez-preskocenia      nevypínať AI ani pri jednoznačnom názve súboru
  --istota-do-popisu     zápis „(istota NN %)" do metadát
  --vlakien N            počet vlákien dekódu (predvolene 4)
```
Ladenie parity: `--dump-mel`, `--dump-emb`, `--json`, `--debug-audio`.
