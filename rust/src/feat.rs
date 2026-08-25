//! Log-mel spektrogram – EXAKTNý port ClapFeatureExtractor (transformers).
//! Parametre: 48 kHz, FFT 1024, hop 480, hann (periodické), mocnina 2,
//! Slaney mel filterbank 64 (0–14 000 Hz), dB = 10·log10(max(·,1e-10)).
//! Výstup na okno: 1001 × 64 (f32) – rovnaký tvar ako Python verzia.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

pub const N_FFT: usize = 1024;
pub const HOP: usize = 480;
pub const N_MELS: usize = 64;

/// Slaney mel filterbank: (N_MELS, N_FFT/2+1), s area normalizáciou.
/// Presná Slaney banka (513×64) exportovaná z transformers – garantuje
/// numerickú paritu s Python verziou (žiadne odchýlky vzorca).
pub fn mel_filter_bank_slaney() -> Vec<Vec<f64>> {
    const RAW: &[u8] = include_bytes!("mel_slaney.f32"); // 513×64×4 bajtov
    let n_bins = N_FFT / 2 + 1;
    let mut bank = vec![vec![0.0f64; n_bins]; N_MELS];
    for k in 0..n_bins {
        for m in 0..N_MELS {
            let off = (k * N_MELS + m) * 4;
            let b = [RAW[off], RAW[off + 1], RAW[off + 2], RAW[off + 3]];
            bank[m][k] = f32::from_le_bytes(b) as f64;
        }
    }
    bank
}

/// Periodické Hannovo okno (np.hanning(N+1)[:-1]).
fn hann_periodic() -> Vec<f64> {
    (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())
        .collect()
}

pub fn log_mel(window: &[f32], bank: &[Vec<f64>]) -> Vec<f32> {
    assert_eq!(window.len(), 480_000);
    let frame_len = N_FFT;
    let half = frame_len / 2; // 512

    // 1) reflect padding 512 obojstranne (np.pad mode="reflect")
    let n = window.len();
    let mut y = vec![0.0f64; n + 2 * half];
    for i in 0..half {
        y[i] = window[half - i] as f64; // zrkadlenie bez okrajovej vzorky
    }
    for i in 0..n {
        y[half + i] = window[i] as f64;
    }
    for i in 0..half {
        y[half + n + i] = window[n - 2 - i] as f64;
    }

    // 2) FFT rámcov
    let win = hann_periodic();
    let num_frames = 1 + (y.len() - frame_len) / HOP; // 1001
    let n_bins = N_FFT / 2 + 1;
    let mut spec = vec![vec![0.0f64; n_bins]; num_frames];

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(frame_len);

    let mut buf = vec![Complex::new(0.0, 0.0); frame_len];
    for (t, spec_frame) in spec.iter_mut().enumerate() {
        let start = t * HOP;
        for (j, b) in buf.iter_mut().enumerate() {
            b.re = y[start + j] * win[j];
            b.im = 0.0;
        }
        fft.process(&mut buf);
        // onesided + power 2.0
        for (k, s) in spec_frame.iter_mut().enumerate() {
            *s = buf[k].norm_sqr();
        }
    }

    // 3) mel filterbank + dB (10·log10(max(·,1e-10))) → f32
    let mut out = vec![0.0f32; num_frames * N_MELS];
    for (t, spec_frame) in spec.iter().enumerate() {
        for (m, row) in bank.iter().enumerate() {
            let mut acc = 0.0f64;
            for (k, w) in row.iter().enumerate() {
                if *w != 0.0 {
                    acc += w * spec_frame[k];
                }
            }
            let v = acc.max(1e-10);
            out[t * N_MELS + m] = (10.0 * v.log10()) as f32;
        }
    }
    out
}
