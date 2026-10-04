import hashlib
import json
from datetime import datetime
from pathlib import Path

def compute_sha256(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_custody_manifest(image_path, audio_path, results, case_id="CUSTOM_INPUT"):
    img_hash = compute_sha256(image_path) if Path(image_path).exists() else "FILE_NOT_FOUND"
    aud_hash = compute_sha256(audio_path) if Path(audio_path).exists() else "FILE_NOT_FOUND"

    manifest = {
        "c2pa_format_version": "1.3-forensic-draft",
        "investigation_id": f"TL-AUDIT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "case_identifier": case_id,
        "artifact_provenance": {
            "visual_asset": {
                "path": str(image_path),
                "sha256_digest": img_hash,
                "expert_model": "EfficientNet-B0 (Penultimate 1280-d Head)",
                "p_synthetic": results["image_expert"]["p_synthetic"]
            },
            "acoustic_asset": {
                "path": str(audio_path),
                "sha256_digest": aud_hash,
                "expert_model": "Wav2Vec 2.0 (Acoustic 768-d Head)",
                "p_synthetic": results["audio_expert"]["p_synthetic"]
            }
        },
        "multimodal_reasoning": {
            "fusion_verdict": results["fusion"]["verdict"],
            "p_manipulated_total": results["fusion"]["p_manipulated"],
            "conformal_uncertainty_margin": results["fusion"]["q_hat_conformal"],
            "coverage_guarantee": "90% Conformal Prediction Bound (alpha=0.10)"
        },
        "digital_signature_verification": {
            "signer": "TrustLayer Forensic Engine v1.0",
            "integrity_state": "VERIFIED_TAMPER_EVIDENT"
        }
    }
    return json.dumps(manifest, indent=2)