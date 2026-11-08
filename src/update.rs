//! Vérification et installation des nouvelles versions publiées sur GitHub Releases.

use anyhow::{anyhow, Result};

pub const REPO: &str = "plimb-ai/pilmb-app";
pub const CURRENT: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Debug)]
pub struct Release {
    pub version: String,
    pub notes: String,
    pub page_url: String,
    pub asset_url: Option<String>,
    pub asset_size: u64,
}

fn parse_version(tag: &str) -> Option<semver::Version> {
    semver::Version::parse(tag.trim().trim_start_matches(['v', 'V'])).ok()
}

/// Renvoie la dernière release si elle est plus récente que la version installée.
pub fn check() -> Result<Option<Release>> {
    let json = crate::net::get_json(&format!("https://api.github.com/repos/{REPO}/releases/latest"))?;
    let tag = json["tag_name"].as_str().ok_or_else(|| anyhow!("réponse GitHub inattendue"))?;
    let latest = parse_version(tag).ok_or_else(|| anyhow!("version invalide : {tag}"))?;
    let current = parse_version(CURRENT).expect("version du paquet");
    if latest <= current {
        return Ok(None);
    }
    let wanted = crate::platform::release_asset_name();
    let asset = json["assets"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|a| Some(a["name"].as_str().unwrap_or("")) == wanted);
    Ok(Some(Release {
        version: latest.to_string(),
        notes: json["body"].as_str().unwrap_or("").trim().to_string(),
        page_url: json["html_url"].as_str().unwrap_or("").to_string(),
        asset_url: asset.and_then(|a| a["browser_download_url"].as_str().map(String::from)),
        asset_size: asset.and_then(|a| a["size"].as_u64()).unwrap_or(0),
    }))
}
