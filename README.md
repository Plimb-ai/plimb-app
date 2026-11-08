# Plimb Studio

Application d'IA **100 % locale** pour les modèles publiés par Plimb sur
[huggingface.co/plimb](https://huggingface.co/plimb). Écrite entièrement en **Rust**
(interface [Slint](https://slint.dev), inférence [Candle](https://github.com/huggingface/candle)),
elle fonctionne sur **Windows**, **Linux** et **Android**.

Les modèles sont téléchargés une seule fois depuis Hugging Face, puis tout s'exécute hors ligne
sur le processeur de l'appareil : aucun texte ni aucune image ne quitte la machine.

## Cas d'utilisation et modèles

| Catégorie | Modèle | Ce qu'il fait |
|---|---|---|
| **Discussion** | [`gladios-0.3b-eng-prototype`](https://huggingface.co/plimb/gladios-0.3b-eng-prototype) | Chat en anglais (≈300 M paramètres, checkpoint PyTorch lu directement en Rust) |
| **Atelier d'histoires** | [`gladios-tiny.story-0.1B`](https://huggingface.co/plimb/gladios-tiny.story-0.1B) | Écrit de courtes histoires à partir d'un début de phrase (GPT-2, safetensors) |
| **Détourage couleur** | [`codelor-color-1.2`](https://huggingface.co/plimb/codelor-color-1.2) | Retire un fond vert (ou d'une autre couleur) d'une image (PNG transparent) ou d'une vidéo (MP4) |

Fonctions principales :

- accueil par catégories, page de chaque catégorie avec ses modèles ;
- gestionnaire de modèles : téléchargement avec progression et reprise, suppression, espace utilisé ;
- discussion avec réponses en streaming, historique, paramètres (longueur, Top-K, Top-P, température, pénalité de répétition) ;
- détourage d'images : choix de la couleur, seuil, bords adoucis, fond transparent/blanc/noir, vue du masque, export PNG ;
- détourage de vidéos MP4 (H.264) : traitement image par image avec progression, fond noir ou blanc, export MP4 (le son n'est pas conservé, le HEVC/H.265 n'est pas pris en charge) ;
- thème auto/clair/sombre, interface adaptée au bureau comme au téléphone ;
- **mises à jour intégrées** : l'app vérifie la dernière release GitHub et se met à jour
  (remplacement de l'exécutable sur Windows/Linux, téléchargement de l'APK sur Android).

## Télécharger

Les binaires sont publiés dans les [Releases](https://github.com/Plimb-ai/plimb-app/releases) :

- `plimb-studio-windows-x86_64.exe` — Windows 10/11 ;
- `plimb-studio-linux-x86_64` (ou l'archive `.tar.gz`) — Linux x86_64 ;
- `plimb-studio-android-arm64.apk` — Android 8.0+ (arm64).

## Releases automatiques

Le workflow [`.github/workflows/release.yml`](.github/workflows/release.yml) compile les trois
plateformes à chaque pull request, et **publie automatiquement une release à chaque push sur `main`** :

- si la version de `Cargo.toml` est plus récente que le dernier tag, elle est utilisée
  (modifiez-la pour passer en 0.2.0, 1.0.0…) ;
- sinon le numéro de correctif est incrémenté automatiquement (0.1.0 → 0.1.1 → 0.1.2…).

Les applications installées détectent la nouvelle release au démarrage et proposent la mise à jour.

### Signature Android (à faire une fois)

Pour qu'Android accepte d'installer une mise à jour par-dessus l'ancienne version, tous les APK
doivent être signés avec **la même clé**. Créez-la une fois :

```bash
keytool -genkeypair -keystore plimb-release.keystore -alias plimb -keyalg RSA -keysize 2048 \
  -validity 10000 -storepass MOT_DE_PASSE -keypass MOT_DE_PASSE -dname "CN=Plimb Studio,O=Plimb"
base64 -w0 plimb-release.keystore   # à copier dans le secret ci-dessous
```

puis ajoutez dans *Settings → Secrets and variables → Actions* du dépôt :

- `ANDROID_KEYSTORE_BASE64` : le contenu base64 du fichier ;
- `ANDROID_KEYSTORE_PASSWORD` : le mot de passe (identique pour le keystore et la clé).

Sans ces secrets, le workflow signe avec une clé temporaire (un avertissement s'affiche) et
chaque nouvelle version demandera de désinstaller la précédente. Gardez le keystore en lieu sûr.

## Compiler soi-même

```bash
# Bureau (Linux : paquets libfontconfig1-dev libxkbcommon-dev libwayland-dev libx11-dev requis)
cargo run --release

# Android (Android SDK + NDK, build-tools ≥ 35 ou JDK 17)
rustup target add aarch64-linux-android
cargo install cargo-apk
cargo apk build --release --lib --target aarch64-linux-android
```

Test rapide des moteurs d'inférence sans interface, sur des modèles déjà téléchargés :

```bash
cargo run --release --example smoke -- <dossier contenant story/ chat/ codelor/>
```

## Organisation du code

```
src/ai/        moteur d'inférence (tokenizer BPE, GPT, Codelor) en Rust pur
src/app.rs     logique de l'application (navigation, téléchargements, génération, détourage)
src/catalog.rs catalogue des modèles Plimb et catégories
src/update.rs  vérification des releases GitHub
src/video.rs   lecture/écriture MP4 H.264 (mp4 + OpenH264)
src/platform/  spécificités bureau / Android (fichiers, liens, permissions, auto-update)
ui/            interface Slint
```

Pour ajouter un modèle : déclarez-le dans `src/catalog.rs` (et son moteur dans `src/ai/` si
son architecture est nouvelle).

## Licence

MIT. Chaque modèle a sa propre fiche et licence sur Hugging Face.
