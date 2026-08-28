ANALYZÁTOR ZVUKOVÝCH SÚBOROV – RUST TESTOVACIA VERZIA (0.7.1)
================================================================

Toto je rýchlejšia alternatívna verzia programu. Dáva ROVNAKÉ
výsledky ako pôvodná (Python) verzia – to bolo overené číslami.

ČO JE V TOMTO PRIECINKU
  analyzator-rs.exe ................ samotný program
  onnxruntime.dll .................. knižnica pre AI (nutná, nechať tu)
  onnxruntime_providers_shared.dll . knižnica pre AI (nutná, nechať tu)
  TEST-RUST.bat .................... spúšťač (dvojklik)
  popisy.txt ....................... zoznam popisov (možno upravovať)
  README-RUST.txt .................. tento súbor

AKO POUŽÍVAŤ
  1. Celý tento priecinok nakopíruj do hlavného priecinka programu
     (tam, kde je SPUSTI.bat a priečinok „models“). Vďaka tomu
     program použije už raz stiahnutý AI model aj naučené dáta.
  2. Dvojklik na TEST-RUST.bat a zadaj cestu k priečinku so zvukmi
     (alebo potiahni priečinok myskou na TEST-RUST.bat).
  3. Program vypíše popis každého súboru a zapíše ho do metadát
     (rovnako ako Python verzia – vrátane učenia sa).

POZNÁMKY
  - Program podporuje aj GPU (DirectML) automaticky, ak je k dispozícii.
  - Naučené dáta (naucene_spojenia.json, naucene_vzory.npz) sú spoločné
    s Python verziou – čo naučí jedna, druhá vie.
  - Toto je testovacia verzia bez grafického okna (pracuje v čiernom
    okne príkazového riadku). Python verzia zostáva hlavná.
