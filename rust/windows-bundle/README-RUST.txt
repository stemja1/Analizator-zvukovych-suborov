ANALYZÁTOR ZVUKOVÝCH SÚBOROV – RUST TESTOVACIA VERZIA (0.9.0)
================================================================

Rýchlejšia alternatívna verzia programu. Dáva ROVNAKÉ výsledky
ako pôvodná (Python) verzia – overené číslami.

ČO JE V TOMTO PRIECINKU
  analyzator-gui.exe ............ GRAFICKÁ VERZIA – dvojklik a máte okno
  analyzator-rs.exe ............. príkazová verzia (používa TEST-RUST.bat)
  onnxruntime.dll ............... knižnica pre AI (nutná, nechať tu)
  onnxruntime_providers_shared.dll  knižnica pre AI (nutná, nechať tu)
  TEST-RUST.bat ................. spúšťač príkazovej verzie
  popisy.txt .................... zoznam popisov (okno ho automaticky načíta)
  README-RUST.txt ............... tento súbor

AKO POUŽÍVAŤ (GRAFICKÁ VERZIA)
  1. Celý tento priecinok nakopíruj do hlavného priecinka programu
     (tam, kde je SPUSTI.bat a priečinok „models“). Vďaka tomu
     program použije už raz stiahnutý AI model.
  2. Dvojklik na analyzator-gui.exe
     (Ak Windows zobrazí modrú tabuľku, klikni „Viac informácií“
     a „Napriek tomu spustiť“.)
  3. Do poľa hore napíš cestu k priečinku so zvukmi a stlač Enter,
     alebo klikni „Vybrať priečinok…“, alebo jednoducho potiahni
     priečinok so zvukmi myskou do okna.
  4. Vpravo skontroluj zoznam popisov a nastavenia
     (prah istoty, počet okien…).
  5. Klikni „▶ Analyzovať“. Priebeh vidíš v tabuľke aj v logu dole.

POZNÁMKY
  - Program podporuje aj GPU (DirectML) automaticky, ak je k dispozícii.
  - Bez učenia sa (od verzie 0.8.0) – istotu posilní len priama zhoda
    slova z názvu súboru s textom popisu.
  - Okno nevie prehrávať zvuky (Python verzia to vie) – na kontrolu
    zvuku používajte Python verziu.
  - Príkazová verzia (TEST-RUST.bat) zostáva pre rychle hromadné
    spúšťanie a testovanie.
