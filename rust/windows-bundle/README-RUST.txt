ANALYZÁTOR ZVUKOVÝCH SÚBOROV – RUST TESTOVACIA VERZIA (0.8.0)
================================================================

Rýchlejšia alternatívna verzia programu. Dáva ROVNAKÉ výsledky
ako pôvodná (Python) verzia – overené číslami.

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
     program použije už raz stiahnutý AI model.
  2. Dvojklik na TEST-RUST.bat a zadaj cestu k priecinku so zvukmi
     (alebo potiahni priečinok myskou na TEST-RUST.bat).
  3. Program vypíše popis každého súboru a zapíše ho do metadát.

POZNÁMKY
  - Program podporuje aj GPU (DirectML) automaticky, ak je k dispozícii.
  - Od verzie 0.8.0 už program nemá funkciu učenia sa (naucene_*
    súbory sa netvoria ani nepoužívajú). Názov súboru posilní istotu
    vždy, keď slovo z názvu priamo sedí na popis.
  - Toto je testovacia verzia bez grafického okna. Python verzia
    zostáva hlavná.
