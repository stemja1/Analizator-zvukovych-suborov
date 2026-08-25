//! Logika okolo názvov súborov – port z Pythonu (core_analyzer).
//! Kľúčové slová: len ASCII (anglické), dĺžka ≥ 4, bez všeobecných slov.

pub const NAME_MIN_WORD_LEN: usize = 4;
pub const NAME_SKIP_MIN_WORD_LEN: usize = 5;
pub const NAME_BOOST_FACTOR: f64 = 1.3;
pub const NAME_BOOST_CAP: f64 = 0.99;

pub const STOPWORDS: &[&str] = &[
    "sound", "sounds", "audio", "zvuk", "zvuky", "zvukova", "zvukove",
    "nahravka", "nahravky", "rec", "recording", "record", "file", "subor",
    "subory", "final", "finalna", "finalne", "mix", "demo", "test",
    "testovaci", "novy", "nova", "nove", "stary", "stara", "kopie",
    "kopija", "copy", "track", "sample", "samples", "edit", "uprava",
    "upraveny", "cut", "rez", "video", "wav", "mp3", "flac", "ogg",
    "the", "and", "with", "from", "this", "new", "old", "max", "min",
    "ver", "version", "hlas", "song", "klip", "full", "free", "best",
    "good", "diag", "diagnoza", "vyrok", "siec", "sit", "train",
];

/// Významové anglické slová z názvu súboru (malej písmená, bez čísel).
pub fn filename_keywords(path: &str) -> Vec<String> {
    let base = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in base.chars() {
        if ch.is_alphabetic() {
            cur.push(ch);
        } else if !cur.is_empty() {
            push_word(&mut cur, &mut out);
        }
    }
    if !cur.is_empty() {
        push_word(&mut cur, &mut out);
    }
    out
}

fn push_word(cur: &mut String, out: &mut Vec<String>) {
    let w = std::mem::take(cur);
    if w.len() >= NAME_MIN_WORD_LEN
        && w.is_ascii()
        && !STOPWORDS.contains(&w.as_str())
        && !out.contains(&w)
    {
        out.push(w);
    }
}

/// Sedí kľúčové slovo na popis? (podreťazec; „birds“ sedí aj na „bird“)
pub fn keyword_in_description(kw: &str, desc: &str) -> bool {
    let d = desc.to_lowercase();
    if d.contains(kw) {
        return true;
    }
    if kw.ends_with('s') && kw.len() > NAME_MIN_WORD_LEN {
        return d.contains(&kw[..kw.len() - 1]);
    }
    false
}

/// Slová z názvu vs. popis (+ naučené spojenia).
pub fn name_matches_description(
    path: &str,
    desc: &str,
    learned: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    for kw in filename_keywords(path) {
        if keyword_in_description(&kw, desc) {
            return true;
        }
        if let Some(counts) = learned.get(&kw).and_then(|v| v.as_object()) {
            if counts.get(desc).and_then(|c| c.as_i64()).unwrap_or(0) >= 1 {
                return true;
            }
        }
    }
    false
}

/// Popis, ak názov súboru JEDNOZNAČNE určuje práve jeden popis (inak None).
pub fn name_skip_description(
    path: &str,
    descriptions: &[String],
    learned: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    let kws = filename_keywords(path);
    if kws.is_empty() {
        return None;
    }
    let hits: Vec<usize> = descriptions
        .iter()
        .map(|d| kws.iter().filter(|k| keyword_in_description(k, d)).count())
        .collect();
    let best_i = hits
        .iter()
        .enumerate()
        .max_by_key(|(_, h)| **h)
        .map(|(i, _)| i)?;
    let best = hits[best_i];
    if best > 0 && hits.iter().filter(|h| **h == best).count() == 1 {
        let strong = best >= 2
            || kws
                .iter()
                .any(|k| k.len() >= NAME_SKIP_MIN_WORD_LEN && keyword_in_description(k, &descriptions[best_i]));
        if strong {
            return Some(descriptions[best_i].clone());
        }
    }
    // naučené spojenie: slovo viedlo 2× k rovnakému popisu (jednoznačne)
    for kw in &kws {
        if let Some(counts) = learned.get(kw).and_then(|v| v.as_object()) {
            let tops: Vec<(&String, i64)> = counts
                .iter()
                .filter(|(_, c)| c.as_i64().unwrap_or(0) >= 2)
                .map(|(d, c)| (d, c.as_i64().unwrap_or(0)))
                .collect();
            if tops.len() == 1 && descriptions.iter().any(|d| d == tops[0].0) {
                return Some(tops[0].0.clone());
            }
        }
    }
    None
}
