//! Aktualizácie – zistenie novej verzie na GitHube, stiahnutie a rozbalenie.
//! Používa to GUI (tlačidlo 🔄) aj CLI (--aktualizacia). Žiadne tajné
//! kľúče sa do programu nevkladajú – repozitár je verejný; voliteľný
//! token sa prečíta z premennej prostredia GITHUB_TOKEN (len na vývoj).

use anyhow::{Context, Result};
use std::io::{Read, Write as _};
use std::path::{Path, PathBuf};

pub const REPO_API: &str = "https://api.github.com/repos/stemja1/Analizator-zvukovych-suborov/releases/latest";
pub const ASSET_NAME: &str = "analyzator-rs-windows.zip";
/// Názov priečinka vo vnútri balíka.
pub const BUNDLE_DIR: &str = "analyzator-rs-windows";

#[derive(Debug, Clone)]
pub struct ReleaseInfo {
    pub tag: String,   // napr. "v0.11.0"
    pub name: String,  // titulok release
    pub url: String,   // priamy odkaz na zip
    pub size: u64,     // bajtov
}

fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(20))
        .build()
}

/// Najnovšie vydanie na GitHube (verejný repozitár – token netreba).
pub fn latest_release() -> Result<ReleaseInfo> {
    let mut req = http_agent().get(REPO_API).set("User-Agent", "analyzator-rs");
    if let Ok(tok) = std::env::var("GITHUB_TOKEN") {
        req = req.set("Authorization", &format!("Bearer {tok}"));
    }
    let resp = req.call().context("GitHub neodpovedá (kontrola internetu)")?;
    let v: serde_json::Value = resp
        .into_json()
        .context("neplatná odpoveď z GitHubu")?;
    let tag = v
        .get("tag_name")
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string();
    let name = v
        .get("name")
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string();
    let asset = v
        .get("assets")
        .and_then(|a| a.as_array())
        .and_then(|a| {
            a.iter()
                .find(|x| x.get("name").and_then(|n| n.as_str()) == Some(ASSET_NAME))
        })
        .ok_or_else(|| anyhow::anyhow!("v release {tag} chýba súbor {ASSET_NAME}"))?;
    Ok(ReleaseInfo {
        tag,
        name,
        url: asset
            .get("browser_download_url")
            .and_then(|u| u.as_str())
            .unwrap_or_default()
            .to_string(),
        size: asset.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
    })
}

/// Je `latest` (napr. „v0.11.0") novšia ako `current` (napr. „0.10.1")?
pub fn is_newer(latest: &str, current: &str) -> bool {
    let parse = |s: &str| -> Vec<u64> {
        s.trim_start_matches('v')
            .split(|c: char| !c.is_ascii_digit())
            .filter(|p| !p.is_empty())
            .filter_map(|p| p.parse::<u64>().ok())
            .collect()
    };
    let (a, b) = (parse(latest), parse(current));
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        if x != y {
            return x > y;
        }
    }
    false
}

/// Stiahne `url` do `path` a hlási priebeh (prečítané / celkom).
pub fn download_to(url: &str, path: &Path, progress: &dyn Fn(u64, u64)) -> Result<()> {
    let resp = http_agent()
        .get(url)
        .set("User-Agent", "analyzator-rs")
        .call()
        .context("sťahovanie sa nepodarilo")?;
    let total: u64 = resp
        .header("Content-Length")
        .and_then(|h| h.parse().ok())
        .unwrap_or(0);
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut done: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    file.flush()?;
    Ok(())
}

/// Rozbalí balík do `dest`; vráti cestu k priečinku s novými súbormi.
pub fn extract_bundle(zip_path: &Path, dest: &Path) -> Result<PathBuf> {
    let f = std::fs::File::open(zip_path)?;
    let mut ar = zip::ZipArchive::new(f)?;
    ar.extract(dest)?;
    let inner = dest.join(BUNDLE_DIR);
    anyhow::ensure!(
        inner.join("analyzator-gui.exe").exists(),
        "v balíku chýba analyzator-gui.exe"
    );
    Ok(inner)
}

/// Zapíše a spustí aktualizačný .bat (Windows): počka na ukončenie
/// procesu `pid`, prekopíruje nové súbory, spustí GUI a uprace.
#[cfg(windows)]
pub fn install_and_restart(src: &Path, pid: u32) -> Result<()> {
    use std::io::Write;
    let dir = src
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let bat = dir.join("_AKTUALIZUJ-RUST.bat");
    let mut b = std::fs::File::create(&bat)?;
    let content = format!(
        "@echo off\r\n\
         rem Automaticka aktualizacia - vytvoril analyzator-gui\r\n\
         set \"DIR={dir}\"\r\n\
         powershell -NoProfile -Command \"Wait-Process -Id {pid}\" >nul 2>&1\r\n\
         robocopy \"{src}\" \"%DIR%\" /E /XF _AKTUALIZUJ-RUST.bat /NFL /NDL /NJH /NJS >nul\r\n\
         rmdir /s /q \"%DIR%_update_tmp\" >nul 2>&1\r\n\
         del \"%DIR%_update.zip\" >nul 2>&1\r\n\
         start \"\" \"%DIR%analyzator-gui.exe\"\r\n\
         del \"%~f0\" >nul 2>&1\r\n",
        dir = dir.display(),
        src = src.display(),
        pid = pid,
    );
    b.write_all(content.as_bytes())?;
    use std::os::windows::process::CommandExt;
    std::process::Command::new("cmd")
        .args(["/c", &bat.display().to_string()])
        .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
        .spawn()?;
    Ok(())
}
