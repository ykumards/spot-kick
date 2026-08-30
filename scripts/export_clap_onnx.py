# pyright: reportMissingImports=false
# torch and transformers come from the optional `export` extra and are not in the runtime venv.
"""Export CLAP's audio tower (+ projection) to ONNX and check parity with the torch path.

Dev-time only: needs torch + transformers + onnx. The runtime needs neither — see `spotkick/ears/clap.py`.

    python scripts/export_clap_onnx.py --out ~/.spotkick/models

Writes clap-audio.onnx (fp16 weights, float32 in and out: half the size of the fp32 export, ~2e-6 cosine
drift), clap-mel.npy (the Slaney filterbank), and prints max deviations:
  features:  our numpy log-mel vs transformers' extractor on the same crop
  tower:     onnxruntime vs torch on the same features
  fp16:      the shipped half-precision model vs the fp32 export
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spotkick.ears import features  # after the sys.path insert above, so the script runs from any cwd

MODEL_ID = "laion/clap-htsat-unfused"
DEFAULT_OUT_DIR = Path.home() / ".spotkick" / "models"
DEFAULT_OPSET = 17
ONNX_FILE = "clap-audio.onnx"
MEL_FILE = "clap-mel.npy"
TEST_SECONDS = 30
TEST_AMPLITUDE = 0.3
TEST_SEED = 0
PARITY_CLIPS = 3
BYTES_PER_MB = 1e6


class AudioTower(torch.nn.Module):
    """The audio half of CLAP as one module: HTSAT encoder, pooled output, projection. Output is not normalised."""

    def __init__(self, clap_model):
        super().__init__()
        self.audio_model = clap_model.audio_model
        self.projection = clap_model.audio_projection

    def forward(self, input_features):
        pooled = self.audio_model(input_features=input_features, is_longer=None).pooler_output
        return self.projection(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    return parser.parse_args()


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def check_feature_parity(extractor, mel_filters: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """Compare the numpy log-mel with transformers' extractor on one 10 s crop; return the numpy features."""
    reference = extractor._np_extract_fbank_features(crop, extractor.mel_filters_slaney)
    ours = features.log_mel(crop, mel_filters)
    max_diff = np.abs(ours - reference).max()
    print(f"features: shape {ours.shape} vs {reference.shape}, max |diff| {max_diff:.3e}")
    return ours


def check_tower_parity(clap_model, tower: AudioTower, model_input: torch.Tensor) -> None:
    """Compare the normalised wrapper output with ``get_audio_features``, which normalises internally."""
    with torch.no_grad():
        wanted = clap_model.get_audio_features(input_features=model_input)
        if hasattr(wanted, "pooler_output"):
            wanted = wanted.pooler_output
        wanted = wanted.numpy()
        got = unit_rows(tower(model_input).numpy())
    max_diff = np.abs(wanted - got).max()
    print(f"tower wrapper (normalized) vs get_audio_features: max |diff| {max_diff:.3e}")


def export_onnx(tower: AudioTower, example_input: torch.Tensor, path: Path, opset: int) -> None:
    torch.onnx.export(
        tower,
        (example_input,),
        str(path),
        input_names=["input_features"],
        output_names=["embedding"],
        dynamic_axes={"input_features": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )


def check_onnx_parity(path: Path, tower: AudioTower, wave: np.ndarray, mel_filters: np.ndarray) -> None:
    """Compare onnxruntime with torch on a 3-clip batch, the shape the app runs."""
    import onnxruntime  # deferred: only the parity check needs it

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    batch = features.input_features(wave, mel_filters, n_clips=PARITY_CLIPS)
    from_onnx = np.asarray(session.run(None, {"input_features": batch})[0], dtype=np.float32)
    with torch.no_grad():
        from_torch = tower(torch.from_numpy(batch)).numpy()
    max_diff = np.abs(from_onnx - from_torch).max()
    dot_products = np.sum(from_onnx * from_torch, axis=1)
    onnx_lengths = np.linalg.norm(from_onnx, axis=1)
    torch_lengths = np.linalg.norm(from_torch, axis=1)
    cosine = np.mean(dot_products / onnx_lengths / torch_lengths)
    print(f"onnx vs torch on a 3-clip batch: max |diff| {max_diff:.3e}; cosine {cosine:.6f}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import ClapModel, ClapProcessor  # deferred: slow import, dev-only dependency

    clap_model = ClapModel.from_pretrained(MODEL_ID).eval()
    extractor = ClapProcessor.from_pretrained(MODEL_ID).feature_extractor
    mel_filters = np.asarray(extractor.mel_filters_slaney, dtype=np.float32)  # (513, 64)
    np.save(out_dir / MEL_FILE, mel_filters)

    rng = np.random.default_rng(TEST_SEED)
    wave = (TEST_AMPLITUDE * rng.standard_normal(TEST_SECONDS * features.SR)).astype(np.float32)
    crop = wave[: features.CLIP]
    our_log_mel = check_feature_parity(extractor, mel_filters, crop)

    tower = AudioTower(clap_model).eval()
    example_input = torch.from_numpy(our_log_mel[None, None])  # (1, 1, 1001, 64)
    check_tower_parity(clap_model, tower, example_input)

    onnx_path = out_dir / ONNX_FILE
    export_onnx(tower, example_input, onnx_path, args.opset)
    check_onnx_parity(onnx_path, tower, wave, mel_filters)
    halve_weights(onnx_path, wave, mel_filters)
    size_mb = onnx_path.stat().st_size / BYTES_PER_MB
    print(f"wrote {onnx_path} ({size_mb:.0f} MB) and {out_dir / MEL_FILE}")


def halve_weights(onnx_path: Path, wave: np.ndarray, mel_filters: np.ndarray) -> None:
    """Rewrite the export with fp16 weights, keeping float32 inputs and outputs.

    onnxruntime's own converter is used; onnxconverter-common mistypes the Cast nodes inside HTSAT's attention
    blocks and the result does not load.
    """
    import onnx
    import onnxruntime
    from onnxruntime.transformers.float16 import convert_float_to_float16

    fp32_path = onnx_path.with_suffix(".fp32.onnx")
    onnx_path.rename(fp32_path)
    half = convert_float_to_float16(onnx.load(str(fp32_path)), keep_io_types=True)
    onnx.save(half, str(onnx_path))

    model_input = features.input_features(wave, mel_filters)
    def embed(path: Path) -> np.ndarray:
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        vector = np.asarray(session.run(None, {"input_features": model_input})[0])[0]
        return vector / np.linalg.norm(vector)
    drift = 1.0 - float(embed(onnx_path) @ embed(fp32_path))
    print(f"  fp16 vs fp32 cosine distance: {drift:.2e}")
    fp32_path.unlink()


if __name__ == "__main__":
    main()
