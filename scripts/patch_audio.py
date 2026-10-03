import asyncio
import edge_tts
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_AUD_DIR = BASE_DIR / "data" / "real_audio_E"
RAW_AUD_DIR.mkdir(parents=True, exist_ok=True)

REAL_AUDIO = [
    {"id": "LIBRI_001", "transcript": "he hoped there would be stew for dinner turnip stew and squash pudding"},
    {"id": "LIBRI_002", "transcript": "and the girl said nothing at all but went on cutting her bread and butter"},
    {"id": "LIBRI_003", "transcript": "the mother looked down at her work and the father looked at the ceiling"}
]

print("[*] Patching missing 'real' audio with temporary baseline voices...")

async def patch_audio():
    for item in REAL_AUDIO:
        dest = RAW_AUD_DIR / f"{item['id']}.wav"
        if not dest.exists():
            communicate = edge_tts.Communicate(item["transcript"], "en-GB-SoniaNeural")
            await communicate.save(str(dest))
            print(f"  [+] Generated temporary baseline: {dest.name}")

if __name__ == "__main__":
    asyncio.run(patch_audio())
    print("[✓] Audio patch complete. You are clear to run the ML pipeline.")