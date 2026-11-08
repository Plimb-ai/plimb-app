//! Téléchargements HTTP (modèles et mises à jour).

use anyhow::{anyhow, bail, Result};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const USER_AGENT: &str = concat!("PlimbStudio/", env!("CARGO_PKG_VERSION"));

pub fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .user_agent(USER_AGENT)
        .timeout_connect(Some(Duration::from_secs(20)))
        .build()
        .into()
}

pub fn get_json(url: &str) -> Result<serde_json::Value> {
    let mut resp = agent()
        .get(url)
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| anyhow!("{e}"))?;
    let body = resp.body_mut().read_to_string()?;
    Ok(serde_json::from_str(&body)?)
}

/// Télécharge `url` vers `dest` (via un fichier `.part`, avec reprise).
/// `progress(reçu, total)` est appelé régulièrement.
pub fn download(
    url: &str,
    dest: &Path,
    expected: Option<u64>,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = dest.with_extension(format!(
        "{}part",
        dest.extension().map(|e| format!("{}.", e.to_string_lossy())).unwrap_or_default()
    ));
    let mut start = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if let Some(exp) = expected {
        if start > exp {
            start = 0;
        }
    }
    let mut req = agent().get(url);
    if start > 0 {
        req = req.header("Range", format!("bytes={start}-"));
    }
    let mut resp = req.call().map_err(|e| anyhow!("{e}"))?;
    let resumed = resp.status() == 206;
    if !resumed {
        start = 0;
    }
    let len = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let total = expected.or(len.map(|l| l + start)).unwrap_or(0);
    let mut file = if resumed {
        std::fs::OpenOptions::new().append(true).open(&part)?
    } else {
        std::fs::File::create(&part)?
    };
    let mut reader = resp.body_mut().with_config().limit(u64::MAX).reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut got = start;
    let mut last = Instant::now();
    progress(got, total);
    loop {
        if cancel.load(Ordering::Relaxed) {
            bail!("annulé");
        }
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        got += n as u64;
        if last.elapsed() > Duration::from_millis(150) {
            progress(got, total);
            last = Instant::now();
        }
    }
    file.flush()?;
    drop(file);
    if let Some(exp) = expected {
        if got != exp {
            bail!("taille inattendue ({got} / {exp} octets)");
        }
    }
    progress(got, total.max(got));
    std::fs::rename(&part, dest)?;
    Ok(())
}
