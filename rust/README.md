# Analyzátor – Rust verzia (testovacia alternatíva)

Rovnaká logika ako Python verzia (0.6.0+), spustená ako jeden program
bez Pythonu. CLI, zatiaľ bez GUI. Python verzia zostáva hlavná.

## Zostavenie
```
cd rust
cargo build --release
```
Výsledok: `target/release/analyzator-rs` (~38 MB, obsahuje ONNX Runtime).

DirectML (GPU na Windows): `cargo build --release --features ort-directml`.

## Použitie
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
