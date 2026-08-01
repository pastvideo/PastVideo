#!/usr/bin/env python3
"""Persistent JSON-lines worker for Qwen3-VL multimodal embeddings.

The Rust process keeps this worker alive so the model is loaded exactly once
for indexing and all subsequent queries. Protocol messages are written only to
stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from PIL import Image


RETRIEVAL_INSTRUCTION = "Retrieve video clips relevant to the user's query."


def emit(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def video_frames(path: str, max_frames: int) -> list[Image.Image]:
    """Decode evenly spaced frames and return PIL objects.

    Passing frame objects avoids the upstream file:// handling mismatch between
    qwen-vl-utils and decord on Windows.
    """

    reader = VideoReader(path)
    if len(reader) == 0:
        raise ValueError(f"video contains no frames: {path}")
    count = min(max_frames, len(reader))
    indices = np.linspace(0, len(reader) - 1, count, dtype=int)
    return [Image.fromarray(frame) for frame in reader.get_batch(indices).asnumpy()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=230_400)
    parser.add_argument("--total-pixels", type=int, default=1_843_200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).resolve()
    module_dir = model_path / "scripts"
    if not module_dir.is_dir():
        raise FileNotFoundError(
            f"Qwen model scripts were not found at {module_dir}; "
            "download the official Qwen3-VL-Embedding checkpoint first"
        )
    sys.path.insert(0, str(module_dir))
    from qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    started = time.perf_counter()
    model = Qwen3VLEmbedder(
        str(model_path),
        torch_dtype=dtype,
        attn_implementation="sdpa",
        max_pixels=args.max_pixels,
        total_pixels=args.total_pixels,
        fps=1.0,
        max_frames=args.max_frames,
    )
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    emit(
        {
            "ok": True,
            "ready": True,
            "device": device,
            "gpu": gpu,
            "dimensions": 2048,
            "load_ms": round((time.perf_counter() - started) * 1000),
        }
    )

    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            operation = request.get("op")
            started = time.perf_counter()
            if operation == "text":
                payload = {
                    "text": request["text"],
                    "instruction": RETRIEVAL_INSTRUCTION,
                }
            elif operation == "image":
                with Image.open(request["path"]) as image:
                    payload = {"image": image.convert("RGB").copy()}
            elif operation == "video":
                payload = {
                    "video": video_frames(request["path"], args.max_frames)
                }
            elif operation == "ping":
                emit({"ok": True, "pong": True})
                continue
            else:
                raise ValueError(f"unknown operation: {operation!r}")

            embedding = model.process([payload])[0].float().cpu().tolist()
            emit(
                {
                    "ok": True,
                    "embedding": embedding,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                }
            )
        except Exception as error:  # keep the warm worker alive after a bad input
            traceback.print_exc(file=sys.stderr)
            emit({"ok": False, "error": str(error)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
