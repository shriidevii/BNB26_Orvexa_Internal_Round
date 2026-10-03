import os
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CASES_CSV = DATA_DIR / "cases.csv"
RAW_IMG = DATA_DIR / "poolE_real_images"
GEN_IMG = DATA_DIR / "generated" / "raw" / "img"
RAW_AUD = DATA_DIR / "real_audio_E"
GEN_AUD = DATA_DIR / "generated" / "raw" / "aud"

# Get available files
real_imgs = list(RAW_IMG.glob("*.jpg"))
fake_imgs = list(GEN_IMG.glob("*.jpg"))
real_auds = list(RAW_AUD.glob("*.wav"))
fake_auds = list(GEN_AUD.glob("*.wav"))

print(f"[*] Scaling cases using {len(real_imgs)} real imgs, {len(fake_imgs)} synth imgs, {len(real_auds)} real auds, {len(fake_auds)} synth auds...")

cases = []
case_idx = 1

# 1. 10 Authentic Cases (Real Image + Real Audio)
for i in range(10):
    img = real_imgs[i % len(real_imgs)]
    aud = real_auds[i % len(real_auds)]
    cases.append({
        "case_id": f"TL_AUTH_{case_idx:03d}",
        "image_path": str(img),
        "audio_path": str(aud),
        "caption": "Authentic verified human context capture.",
        "transcript": "Natural spoken human recording.",
        "image_source": "COCO", "image_generator": None,
        "audio_source": "LibriSpeech", "audio_generator": None,
        "ground_truth_case": "AUTHENTIC", "split": "test"
    })
    case_idx += 1

# 2. 10 Image Manipulated Cases (Synthetic Image + Real Audio)
for i in range(10):
    img = fake_imgs[i % len(fake_imgs)]
    aud = real_auds[i % len(real_auds)]
    cases.append({
        "case_id": f"TL_IMG_FAKE_{case_idx:03d}",
        "image_path": str(img),
        "audio_path": str(aud),
        "caption": "Synthetic diffusion image with natural speech.",
        "transcript": "Natural spoken human recording.",
        "image_source": "generated", "image_generator": "SD-Turbo",
        "audio_source": "LibriSpeech", "audio_generator": None,
        "ground_truth_case": "IMAGE_MANIPULATED", "split": "test"
    })
    case_idx += 1

# 3. 10 Audio Manipulated Cases (Real Image + Synthetic Audio)
for i in range(10):
    img = real_imgs[i % len(real_imgs)]
    aud = fake_auds[i % len(fake_auds)]
    cases.append({
        "case_id": f"TL_AUD_FAKE_{case_idx:03d}",
        "image_path": str(img),
        "audio_path": str(aud),
        "caption": "Authentic visual context paired with synthetic neural TTS.",
        "transcript": "Synthetic neural voice output.",
        "image_source": "COCO", "image_generator": None,
        "audio_source": "generated", "audio_generator": "edge-TTS",
        "ground_truth_case": "AUDIO_MANIPULATED", "split": "test"
    })
    case_idx += 1

# 4. 10 Fully Multimodal Synthetic Cases (Synthetic Image + Synthetic Audio)
for i in range(10):
    img = fake_imgs[i % len(fake_imgs)]
    aud = fake_auds[i % len(fake_auds)]
    cases.append({
        "case_id": f"TL_FULL_FAKE_{case_idx:03d}",
        "image_path": str(img),
        "audio_path": str(aud),
        "caption": "Complete synthetic multimodal artifact.",
        "transcript": "Synthetic neural voice output.",
        "image_source": "generated", "image_generator": "SD-Turbo",
        "audio_source": "generated", "audio_generator": "edge-TTS",
        "ground_truth_case": "MULTIMODAL_SYNTHETIC", "split": "test"
    })
    case_idx += 1

df = pd.DataFrame(cases)
df.to_csv(CASES_CSV, index=False)
df.to_csv(DATA_DIR / "poolC.csv", index=False)
print(f"[✓] Successfully generated 40 balanced test cases in {CASES_CSV}")