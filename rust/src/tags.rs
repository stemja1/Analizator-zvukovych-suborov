//! Čítanie/zápis/mazanie popisu vo vlastnostiach súborov.
//! Formáty a konvencie ROVNAKÉ ako Python verzia:
//!  - MP3: ID3v2.4 COMM, lang "eng", desc "Description"
//!  - FLAC/OGG: Vorbis komentár "DESCRIPTION"
//!  - WAV: RIFF LIST INFO ICMT (vlastný čitateľ/zapisovač)

use anyhow::{bail, Result};

pub fn read_description(path: &str) -> String {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "mp3" => read_mp3(path),
        "flac" | "ogg" => read_lofty(path, &ext),
        "wav" => read_wav(path),
        _ => String::new(),
    }
}

pub fn write_description(path: &str, text: &str) -> Result<String> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "mp3" => write_mp3(path, text),
        "flac" | "ogg" => write_lofty(path, text),
        "wav" => write_wav(path, text),
        _ => bail!("nepodporovaný typ súboru: {ext} ({path})"),
    }
}

/// Zmaže popis (ak existuje). Vráti správu pre log.
pub fn remove_description(path: &str) -> Result<String> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "mp3" => remove_mp3(path),
        "flac" | "ogg" => remove_lofty(path, &ext),
        "wav" => remove_wav(path),
        _ => bail!("nepodporovaný typ súboru: {ext} ({path})"),
    }
}

// ---------- MP3 (ID3 COMM) ----------------------------------------------------

fn read_mp3(path: &str) -> String {
    match id3::Tag::read_from_path(path) {
        Ok(tag) => tag
            .comments()
            .find(|c| c.description == "Description")
            .map(|c| c.text.clone())
            .unwrap_or_default(),
        Err(_) => String::new(),
    }
}

fn write_mp3(path: &str, text: &str) -> Result<String> {
    use id3::TagLike;
    let mut tag = id3::Tag::read_from_path(path).unwrap_or_default();
    tag.remove_comment(None, None);
    let comment = id3::frame::Comment {
        lang: "eng".to_string(),
        description: "Description".to_string(),
        text: text.to_string(),
    };
    tag.add_frame(id3::frame::Frame::with_content(
        "COMM",
        id3::frame::Content::Comment(comment),
    ));
    tag.write_to_path(path, id3::Version::Id3v24)?;
    Ok("ID3 COMM".into())
}

fn remove_mp3(path: &str) -> Result<String> {
    use id3::TagLike;
    let mut tag = match id3::Tag::read_from_path(path) {
        Ok(t) => t,
        Err(_) => return Ok("už bolo prázdne".into()),
    };
    if tag.comments().next().is_none() {
        return Ok("už bolo prázdne".into());
    }
    tag.remove_comment(None, None);
    tag.write_to_path(path, id3::Version::Id3v24)?;
    Ok("starý popis zmazaný (ID3)".into())
}

// ---------- FLAC / OGG (Vorbis DESCRIPTION cez lofty) --------------------------

fn read_lofty(path: &str, _ext: &str) -> String {
    use lofty::prelude::*;
    let f = match lofty::read_from_path(path) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };
    f.primary_tag()
        .or_else(|| f.first_tag())
        .and_then(|t| t.get_string(&lofty::tag::ItemKey::Unknown("DESCRIPTION".into())).map(|s| s.to_string()))
        .unwrap_or_default()
}

fn write_lofty(path: &str, text: &str) -> Result<String> {
    use lofty::prelude::*;
    let mut f = lofty::read_from_path(path)?;
    let key = lofty::tag::ItemKey::Unknown("DESCRIPTION".into());
    let tag = if let Some(t) = f.primary_tag_mut() {
        t
    } else if let Some(t) = f.first_tag_mut() {
        t
    } else {
        anyhow::bail!("súbor nemá tag (FLAC/OGG)")
    };
    {
        tag.insert_text(key, text.to_string());
    }
    f.save_to_path(path, lofty::config::WriteOptions::default())?;
    Ok("Vorbis DESCRIPTION".into())
}

fn remove_lofty(path: &str, _ext: &str) -> Result<String> {
    use lofty::prelude::*;
    let mut f = lofty::read_from_path(path)?;
    let key = lofty::tag::ItemKey::Unknown("DESCRIPTION".into());
    let had = f.primary_tag().or_else(|| f.first_tag()).is_some()
        && f
            .primary_tag()
            .or_else(|| f.first_tag())
            .and_then(|t| t.get_string(&key))
            .is_some();
    if !had {
        return Ok("už bolo prázdne".into());
    }
    if let Some(tag) = f.primary_tag_mut() {
        tag.remove_key(&key);
    } else if let Some(tag) = f.first_tag_mut() {
        tag.remove_key(&key);
    }

    f.save_to_path(path, lofty::config::WriteOptions::default())?;
    Ok("starý popis zmazaný (DESCRIPTION)".into())
}

// ---------- WAV (RIFF LIST INFO ICMT) – port z Pythonu --------------------------

fn parse_chunks(data: &[u8]) -> Result<(Vec<(Vec<u8>, Vec<u8>)>, bool)> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        bail!("nie je platný WAV (RIFF/WAVE) súbor");
    }
    let mut chunks = Vec::new();
    let mut pos = 12usize;
    while pos + 8 <= data.len() {
        let cid = data[pos..pos + 4].to_vec();
        let size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
            as usize;
        if pos + 8 + size > data.len() {
            bail!("poškorený WAV chunk {:?}", cid);
        }
        chunks.push((cid, data[pos + 8..pos + 8 + size].to_vec()));
        pos += 8 + size + (size & 1);
    }
    Ok((chunks, true))
}

fn rebuild(chunks: &[(Vec<u8>, Vec<u8>)]) -> Vec<u8> {
    let mut body: Vec<u8> = b"WAVE".to_vec();
    for (cid, payload) in chunks {
        body.extend_from_slice(cid);
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(payload);
        if payload.len() & 1 == 1 {
            body.push(0);
        }
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

fn read_wav(path: &str) -> String {
    let Ok(data) = std::fs::read(path) else {
        return String::new();
    };
    let Ok((chunks, _)) = parse_chunks(&data) else {
        return String::new();
    };
    for (cid, payload) in &chunks {
        if cid == b"LIST" && payload.len() >= 4 && &payload[0..4] == b"INFO" {
            let mut p = 4usize;
            while p + 8 <= payload.len() {
                let iid = &payload[p..p + 4];
                let isz = u32::from_le_bytes([
                    payload[p + 4], payload[p + 5], payload[p + 6], payload[p + 7],
                ]) as usize;
                if p + 8 + isz > payload.len() {
                    break;
                }
                if iid == b"ICMT" {
                    return String::from_utf8_lossy(&payload[p + 8..p + 8 + isz])
                        .trim_end_matches('\0')
                        .to_string();
                }
                p += 8 + isz + (isz & 1);
            }
        }
    }
    String::new()
}

fn write_wav(path: &str, text: &str) -> Result<String> {
    let data = std::fs::read(path)?;
    let (mut chunks, _) = parse_chunks(&data)?;
    chunks.retain(|(cid, payload)| !(cid == b"LIST" && payload.starts_with(b"INFO")));
    let mut info: Vec<u8> = b"INFO".to_vec();
    let mut val = text.as_bytes().to_vec();
    val.push(0);
    info.extend_from_slice(b"ICMT");
    info.extend_from_slice(&(val.len() as u32).to_le_bytes());
    info.extend_from_slice(&val);
    if info.len() & 1 == 1 {
        info.push(0);
    }
    chunks.push((b"LIST".to_vec(), info));
    std::fs::write(path, rebuild(&chunks))?;
    Ok("RIFF INFO ICMT".into())
}

fn remove_wav(path: &str) -> Result<String> {
    let data = std::fs::read(path)?;
    let (mut chunks, _) = parse_chunks(&data)?;
    let before = chunks.len();
    chunks.retain(|(cid, payload)| !(cid == b"LIST" && payload.starts_with(b"INFO")));
    if chunks.len() == before {
        return Ok("už bolo prázdne".into());
    }
    std::fs::write(path, rebuild(&chunks))?;
    Ok("starý popis zmazaný (RIFF INFO)".into())
}
