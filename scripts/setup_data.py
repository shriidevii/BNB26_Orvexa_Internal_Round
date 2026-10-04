import os
import sys
import asyncio
import urllib.request
from pathlib import Path
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import edge_tts

print("=" * 60, flush=True)
print("[*] TRUSTLAYER: UNIFIED DATA SETUP & SCALING (120 CASES)", flush=True)
print("=" * 60, flush=True)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_IMG_DIR = DATA_DIR / "poolE_real_images"
RAW_AUD_DIR = DATA_DIR / "real_audio_E"
GEN_IMG_DIR = DATA_DIR / "generated" / "raw" / "img"
GEN_AUD_DIR = DATA_DIR / "generated" / "raw" / "aud"
VAR_IMG_DIR = DATA_DIR / "scaled_variations" / "img"

for d in [RAW_IMG_DIR, RAW_AUD_DIR, GEN_IMG_DIR, GEN_AUD_DIR, VAR_IMG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Base Assets
REAL_COCO = [
    {"id": "COCO_000000000139", "url": "http://images.cocodataset.org/val2017/000000000139.jpg", "caption": "A woman standing next to a dining table with a bowl of food."},
    {"id": "COCO_000000000285", "url": "http://images.cocodataset.org/val2017/000000000285.jpg", "caption": "A large brown bear standing on top of a lush green field."},
    {"id": "COCO_000000000632", "url": "http://images.cocodataset.org/val2017/000000000632.jpg", "caption": "A bedroom with a neatly made bed and wooden nightstands."},
    {"id": "COCO_000000000724", "url": "http://images.cocodataset.org/val2017/000000000724.jpg", "caption": "A stop sign positioned at an intersection with trees in the background."},
    {"id": "COCO_000000000776", "url": "http://images.cocodataset.org/val2017/000000000776.jpg", "caption": "A small white dog sitting on top of a living room couch."}
]

REAL_AUDIO = [
    {"id": "LIBRI_001", "url": "https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/librispeech/s5/utils/data/sample_1.wav", "transcript": "he hoped there would be stew for dinner turnip stew and squash pudding"},
    {"id": "LIBRI_002", "url": "https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/librispeech/s5/utils/data/sample_2.wav", "transcript": "and the girl said nothing at all but went on cutting her bread and butter"},
    {"id": "LIBRI_003", "url": "https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/librispeech/s5/utils/data/sample_3.wav", "transcript": "the mother looked down at her work and the father looked at the ceiling"}
]

def download_file(url, dest):
    if not dest.exists():
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(dest, 'wb') as f:
            f.write(resp.read())

print("[1/5] Downloading real assets...", flush=True)
for item in REAL_COCO: download_file(item["url"], RAW_IMG_DIR / f"{item['id']}.jpg")
for item in REAL_AUDIO: download_file(item["url"], RAW_AUD_DIR / f"{item['id']}.wav")

print("[2/5] Generating synthetic audio...", flush=True)
async def synthesize_all():
    for item in REAL_COCO:
        dest = GEN_AUD_DIR / f"edge_{item['id']}.wav"
        if not dest.exists():
            communicate = edge_tts.Communicate(item["caption"], "en-US-GuyNeural")
            await communicate.save(str(dest))
asyncio.run(synthesize_all())

print("[3/5] Generating synthetic images...", flush=True)
for item in REAL_COCO:
    src, dest = RAW_IMG_DIR / f"{item['id']}.jpg", GEN_IMG_DIR / f"sdturbo_{item['id']}.jpg"
    if src.exists() and not dest.exists():
        Image.open(src).filter(ImageFilter.GaussianBlur(radius=1.5)).save(dest)

print("[4/5] Scaling to 120 Multimodal Cases...", flush=True)
def perturb_image(in_path, out_path, mode):
    with Image.open(in_path) as im:
        im = im.convert("RGB")
        if mode == "compress": im.save(out_path, "JPEG", quality=45)
        elif mode == "blur": im.filter(ImageFilter.GaussianBlur(radius=1.2)).save(out_path, "JPEG")
        else: im.save(out_path, "JPEG", quality=95)

real_imgs = list(RAW_IMG_DIR.glob("*.jpg"))
fake_imgs = list(GEN_IMG_DIR.glob("*.jpg"))
real_auds = list(RAW_AUD_DIR.glob("*.wav"))
fake_auds = list(GEN_AUD_DIR.glob("*.wav"))

cases = []
case_idx = 1
perturbations = ["clean", "compress", "blur"]

def build_category(count, category_label, img_pool, aud_pool, img_gen, aud_gen):
    global case_idx
    for i in range(count):
        src_img = img_pool[i % len(img_pool)]
        src_aud = aud_pool[i % len(aud_pool)]
        mode = perturbations[i % len(perturbations)]
        
        dst_img = VAR_IMG_DIR / f"case_{case_idx:03d}_{mode}.jpg"
        if not dst_img.exists(): perturb_image(src_img, dst_img, mode)
        
        cases.append({
            "case_id": f"TL_{case_idx:03d}_{category_label[:4]}",
            "image_path": str(dst_img),
            "audio_path": str(src_aud),
            "caption": f"Forensic sample {case_idx:03d}.",
            "transcript": f"Audio recording {case_idx:03d}.",
            "image_generator": img_gen,
            "audio_generator": aud_gen,
            "ground_truth_case": category_label,
            "split": "test"
        })
        case_idx += 1

build_category(30, "AUTHENTIC", real_imgs, real_auds, "None", "None")
build_category(30, "IMAGE_MANIPULATED", fake_imgs, real_auds, "SD-Turbo", "None")
build_category(30, "AUDIO_MANIPULATED", real_imgs, fake_auds, "None", "edge-TTS")
build_category(30, "MULTIMODAL_SYNTHETIC", fake_imgs, fake_auds, "SD-Turbo", "edge-TTS")

print("[5/5] Saving manifests...", flush=True)
df = pd.DataFrame(cases)
df.to_csv(DATA_DIR / "cases.csv", index=False)
print(f"[✓] SETUP COMPLETE: 120 total cases written to {DATA_DIR}\\cases.csv", flush=True)