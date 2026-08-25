//! Dekódovanie zvuku (wav/mp3/ogg/flac), premix na mono, resample na 48 kHz
//! a krájanie na 10-sekundové okná – rovnaké pravidlá ako Python verzia.

use anyhow::{bail, Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const TARGET_SR: u32 = 48_000;
pub const CLIP_SECONDS: f64 = 10.0;

/// Dekóduje súbor cez symphonia → (mono vzorky f32, pôvodná vzorkovacia frekvencia).
pub fn decode_mono(path: &str) -> Result<(Vec<f32>, u32)> {
    let file = std::fs::File::open(path).with_context(|| format!("otváram {path}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("nepodarilo sa prečítať formát: {path}"))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("žiadna audio stopa"))?;
    let sr = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("neznáma vzorkovacia frekvencia"))?;
    let channels = track
        .codec_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut mono: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => {
                bail!("stream reset – nepodporované")
            }
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track.id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let spec = *decoded.spec();
        let capacity = decoded.capacity() as u64;
        let buf = sample_buf.get_or_insert_with(|| SampleBuffer::<f32>::new(capacity, spec));
        buf.copy_interleaved_ref(decoded);
        // premix na mono = priemer kanálov (rovnako ako librosa mono=True)
        mono.extend(
            buf.samples()
                .chunks(channels)
                .map(|ch| ch.iter().sum::<f32>() / ch.len() as f32),
        );
    }
    Ok((mono, sr))
}

/// Resample windowed-sinc (16-tap, Blackman) – kvalita blízka librosa,
/// deterministický a bez externých závislostí.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let taps = 16usize;
    let mut out = Vec::with_capacity(out_len + taps);
    let scale = if to > from { 1.0 } else { from as f64 / to as f64 }; // anti-alias
    for i in 0..out_len {
        let pos = i as f64 / ratio;
        let base = pos.floor() as i64;
        let frac = pos - base as f64;
        let mut acc = 0.0f64;
        for t in -(taps as i64 / 2)..=(taps as i64 / 2 - 1) {
            let idx = base + t;
            if idx < 0 || idx >= input.len() as i64 {
                continue;
            }
            let x = (t as f64 - frac) * scale;
            let s = if x.abs() < 1e-9 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let w = {
                let n = taps as f64 - 1.0;
                let k = (t as f64 - frac + taps as f64 / 2.0).clamp(0.0, n);
                0.42 - 0.5 * (2.0 * std::f64::consts::PI * k / n).cos()
                    + 0.08 * (4.0 * std::f64::consts::PI * k / n).cos()
            };
            acc += input[idx as usize] as f64 * s * w;
        }
        out.push((acc / scale).clamp(-1.0, 1.0) as f32);
    }
    out
}

/// Načíta súbor ako mono 48 kHz (dekód + resample).

/// Časy štartov 10 s okien – kópia logiky `_prepare_windows` z Pythonu.
pub fn window_starts(total_seconds: f64, segments: usize) -> Vec<f64> {
    let segments = segments.max(1);
    let need = CLIP_SECONDS;
    if total_seconds <= need + 0.5 {
        return vec![0.0];
    }
    let mut starts: Vec<f64> = Vec::new();
    for i in 0..segments {
        let c = (i as f64 + 0.5) / segments as f64 * total_seconds;
        let s = (c - need / 2.0).clamp(0.0, total_seconds - need);
        let s = (s * 10.0).round() / 10.0;
        if !starts.contains(&s) {
            starts.push(s);
        }
    }
    starts
}

/// Vykrája okná (doplnené nulami na presne 10 s) z 48 kHz mono.
pub fn cut_windows(y: &[f32], starts: &[f64]) -> Vec<Vec<f32>> {
    let clip_n = (CLIP_SECONDS * TARGET_SR as f64) as usize;
    starts
        .iter()
        .map(|&s| {
            let i0 = (s * TARGET_SR as f64).round() as usize;
            let end = (i0 + clip_n).min(y.len());
            let mut w = vec![0.0f32; clip_n];
            if i0 < y.len() {
                w[..end - i0].copy_from_slice(&y[i0..end]);
            }
            w
        })
        .collect()
}
