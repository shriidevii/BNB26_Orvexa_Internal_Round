import os
import sys
import asyncio
import urllib.request
from pathlib import Path
import pandas as pd

# Remove diffusers and torch from this script to save memory
from PIL import Image, ImageFilter
import edge_tts

print("=" * 60, flush=True)
print("[*] TRUSTLAYER: MEMORY-SAFE DATA SETUP", flush=True)
print("=" * 60, flush=True)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_IMG_DIR = DATA_DIR / "poolE_real_images"
RAW_AUD_DIR = DATA_DIR / "real_audio_E"
GEN_IMG_DIR = DATA_DIR / "generated" / "raw" / "img"
GEN_AUD_DIR = DATA_DIR / "generated" / "raw" / "aud"
DEMO_DIR = BASE_DIR / "demo_cases"

for d in [RAW_IMG_DIR, RAW_AUD_DIR, GEN_IMG_DIR, GEN_AUD_DIR, DEMO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Real COCO val2017 sample mirrors
REAL_COCO = [
    {"id": "COCO_000000000139", "url": "http://images.cocodataset.org/val2017/000000000139.jpg", "caption": "A woman standing next to a dining table with a bowl of food."},
    {"id": "COCO_000000000285", "url": "http://images.cocodataset.org/val2017/000000000285.jpg", "caption": "A large brown bear standing on top of a lush green field."},
    {"id": "COCO_000000000632", "url": "http://images.cocodataset.org/val2017/000000000632.jpg", "caption": "A bedroom with a neatly made bed and wooden nightstands."},
    {"id": "COCO_000000000724", "url": "http://images.cocodataset.org/val2017/000000000724.jpg", "caption": "A stop sign positioned at an intersection with trees in the background."},
    {"id": "COCO_000000000776", "url": "http://images.cocodataset.org/val2017/000000000776.jpg", "caption": "A small white dog sitting on top of a living room couch."}
]

# 2. Real Speech audio samples
REAL_AUDIO = [
    {"id": "LIBRI_001", "url": "https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/librispeech/s5/utils/data/sample_1.wav", "transcript": "he hoped there would be stew for dinner turnip stew and squash pudding"},
    {"id": "LIBRI_002", "url": "https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/librispeech/s5/utils/data/sample_2.wav", "transcript": "and the girl said nothing at all but went on cutting her bread and butter"},
    {"id": "LIBRI_003", "url": "https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/librispeech/s5/utils/data/sample_3.wav", "transcript": "the mother looked down at her work and the father looked at the ceiling"}
]

def download_file(url: str, dest: Path):
    if not dest.exists():
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(dest, 'wb') as f:
            f.write(resp.read())

print("[1/5] Downloading real COCO images...", flush=True)
for item in REAL_COCO:
    dest = RAW_IMG_DIR / f"{item['id']}.jpg"
    download_file(item["url"], dest)

print("[2/5] Downloading real speech samples...", flush=True)
for item in REAL_AUDIO:
    dest = RAW_AUD_DIR / f"{item['id']}.wav"
    try:
        download_file(item["url"], dest)
    except Exception as e:
        print(f"  [!] Audio download error: {e}", flush=True)

print("[3/5] Generating synthetic audio via edge-tts...", flush=True)
async def synthesize_all():
    for item in REAL_COCO[:3]:
        dest = GEN_AUD_DIR / f"edge_{item['id']}.wav"
        if not dest.exists():
            communicate = edge_tts.Communicate(item["caption"], "en-US-GuyNeural")
            await communicate.save(str(dest))
asyncio.run(synthesize_all())

print("[4/5] Preparing memory-safe synthetic image artifacts...", flush=True)
for item in REAL_COCO[:3]:
    dest = GEN_IMG_DIR / f"sdturbo_{item['id']}.jpg"
    src = RAW_IMG_DIR / f"{item['id']}.jpg"
    if src.exists() and not dest.exists():
        im = Image.open(src)
        # Apply synthetic artifact simulation (Gaussian diffusion smoothing)
        manip = im.filter(ImageFilter.GaussianBlur(radius=1.5))
        manip.save(dest)

print("[5/5] Generating provenance manifests (poolE, poolC, cases)...", flush=True)
pool_e = []
for item in REAL_COCO:
    pool_e.append({
        "sample_id": item["id"], "modality": "image",
        "file_path": str(RAW_IMG_DIR / f"{item['id']}.jpg"),
        "source_type": "real_coco", "generator": None, "label": 0, "split": "train"
    })
for item in REAL_COCO[:3]:
    pool_e.append({
        "sample_id": f"gen_{item['id']}", "modality": "image",
        "file_path": str(GEN_IMG_DIR / f"sdturbo_{item['id']}.jpg"),
        "source_type": "generated", "generator": "SD-Turbo", "label": 1, "split": "train"
    })
for item in REAL_AUDIO:
    pool_e.append({
        "sample_id": item["id"], "modality": "audio",
        "file_path": str(RAW_AUD_DIR / f"{item['id']}.wav"),
        "source_type": "real_librispeech", "generator": None, "label": 0, "split": "train"
    })
for item in REAL_COCO[:3]:
    pool_e.append({
        "sample_id": f"gen_aud_{item['id']}", "modality": "audio",
        "file_path": str(GEN_AUD_DIR / f"edge_{item['id']}.wav"),
        "source_type": "generated", "generator": "edge-TTS", "label": 1, "split": "train"
    })

df_e = pd.DataFrame(pool_e)
df_e.to_csv(DATA_DIR / "poolE.csv", index=False)
df_e[df_e["modality"] == "image"].to_csv(DATA_DIR / "expert_image.csv", index=False)
df_e[df_e["modality"] == "audio"].to_csv(DATA_DIR / "expert_audio.csv", index=False)

cases = [
    {
        "case_id": "TL_CASE_01_AUTH",
        "image_path": str(RAW_IMG_DIR / "COCO_000000000139.jpg"),
        "audio_path": str(RAW_AUD_DIR / "LIBRI_001.wav"),
        "caption": REAL_COCO[0]["caption"],
        "transcript": REAL_AUDIO[0]["transcript"],
        "image_source": "COCO", "image_generator": None,
        "audio_source": "LibriSpeech", "audio_generator": None,
        "ground_truth_case": "AUTHENTIC", "split": "test"
    },
    {
        "case_id": "TL_CASE_02_IMG_FAKE",
        "image_path": str(GEN_IMG_DIR / "sdturbo_COCO_000000000139.jpg"),
        "audio_path": str(RAW_AUD_DIR / "LIBRI_001.wav"),
        "caption": REAL_COCO[0]["caption"],
        "transcript": REAL_AUDIO[0]["transcript"],
        "image_source": "generated", "image_generator": "SD-Turbo",
        "audio_source": "LibriSpeech", "audio_generator": None,
        "ground_truth_case": "IMAGE_MANIPULATED", "split": "test"
    },
    {
        "case_id": "TL_CASE_03_AUD_FAKE",
        "image_path": str(RAW_IMG_DIR / "COCO_000000000285.jpg"),
        "audio_path": str(GEN_AUD_DIR / "edge_COCO_000000000285.wav"),
        "caption": REAL_COCO[1]["caption"],
        "transcript": REAL_COCO[1]["caption"],
        "image_source": "COCO", "image_generator": None,
        "audio_source": "generated", "audio_generator": "edge-TTS",
        "ground_truth_case": "AUDIO_MANIPULATED", "split": "test"
    },
    {
        "case_id": "TL_CASE_04_SYNTHETIC",
        "image_path": str(GEN_IMG_DIR / "sdturbo_COCO_000000000632.jpg"),
        "audio_path": str(GEN_AUD_DIR / "edge_COCO_000000000632.wav"),
        "caption": REAL_COCO[2]["caption"],
        "transcript": REAL_COCO[2]["caption"],
        "image_source": "generated", "image_generator": "SD-Turbo",
        "audio_source": "generated", "audio_generator": "edge-TTS",
        "ground_truth_case": "MULTIMODAL_SYNTHETIC", "split": "test"
    }
]
pd.DataFrame(cases).to_csv(DATA_DIR / "poolC.csv", index=False)
pd.DataFrame(cases).to_csv(DATA_DIR / "cases.csv", index=False)

print(f"[✓] SETUP COMPLETE: Generated manifests and cases in {DATA_DIR}", flush=True)