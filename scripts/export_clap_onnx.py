"""Export CLAP's audio tower (+ projection) to ONNX and check parity with the torch path.

Dev-time only: needs torch + transformers + onnx. The runtime needs neither — see `spotkick/ears/onnx.py`.

    python scripts/export_clap_onnx.py --out ~/.spotkick/models

Writes clap-audio.onnx, clap-mel.npy (the Slaney filterbank), and prints max deviations:
  features:  our numpy log-mel vs transformers' extractor on the same crop
  tower:     onnxruntime vs torch on the same features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spotkick.ears import features as F

MODEL_ID = "laion/clap-htsat-unfused"


class AudioTower(torch.nn.Module):
    def __init__(self, clap):
        super().__init__()
        self.audio_model, self.proj = clap.audio_model, clap.audio_projection

    def forward(self, input_features):
        pooled = self.audio_model(input_features=input_features, is_longer=None).pooler_output
        return self.proj(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / ".spotkick" / "models"))
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

    from transformers import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained(MODEL_ID).eval()
    fe = ClapProcessor.from_pretrained(MODEL_ID).feature_extractor
    mel = np.asarray(fe.mel_filters_slaney, dtype=np.float32)          # (513, 64)
    np.save(out / "clap-mel.npy", mel)

    rng = np.random.default_rng(0)
    wave = (0.3 * rng.standard_normal(30 * F.SR)).astype(np.float32)
    crop = wave[: F.CLIP]
    ref = fe._np_extract_fbank_features(crop, fe.mel_filters_slaney)   # (1001, 64)
    ours = F.log_mel(crop, mel)
    print(f"features: shape {ours.shape} vs {ref.shape}, max |diff| {np.abs(ours - ref).max():.3e}")

    tower = AudioTower(clap).eval()
    x = torch.from_numpy(ours[None, None])                              # (1, 1, 1001, 64)
    with torch.no_grad():
        want = clap.get_audio_features(input_features=x)
        want = (want.pooler_output if hasattr(want, "pooler_output") else want).numpy()
        got = tower(x).numpy()
        got = got / np.linalg.norm(got, axis=1, keepdims=True)         # get_audio_features normalizes; the tower doesn't
    print(f"tower wrapper (normalized) vs get_audio_features: max |diff| {np.abs(want - got).max():.3e}")

    path = out / "clap-audio.onnx"
    torch.onnx.export(tower, (x,), str(path), input_names=["input_features"], output_names=["embedding"],
                      dynamic_axes={"input_features": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=a.opset, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    batch = F.input_features(wave, mel, n_clips=3)
    o = sess.run(None, {"input_features": batch})[0]
    with torch.no_grad():
        t = tower(torch.from_numpy(batch)).numpy()
    print(f"onnx vs torch on a 3-clip batch: max |diff| {np.abs(o - t).max():.3e}; cosine {np.mean(np.sum(o*t,1)/np.linalg.norm(o,axis=1)/np.linalg.norm(t,axis=1)):.6f}")
    print(f"wrote {path} ({path.stat().st_size/1e6:.0f} MB) and {out/'clap-mel.npy'}")


if __name__ == "__main__":
    main()
