#!/usr/bin/env python3
"""Persistent JSON-lines worker for Qwen3-VL multimodal embeddings.

The Rust process keeps this worker alive so the model is loaded exactly once
for indexing and all subsequent queries. Protocol messages are written only to
stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from PIL import Image


RETRIEVAL_INSTRUCTION = "Retrieve video clips relevant to the user's query."
_ALIAS_DIR = tempfile.TemporaryDirectory(prefix="pastvideo-qwen-")
_ALIAS_CACHE: dict[str, str] = {}
_VIDEO_READER_CACHE: OrderedDict[str, VideoReader] = OrderedDict()
_MAX_CACHED_VIDEO_READERS = 2


def close_video_readers() -> None:
    """Release Windows file handles before the temporary alias directory."""

    _VIDEO_READER_CACHE.clear()
    gc.collect()


atexit.register(close_video_readers)


def emit(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def decoder_path(path: str) -> str:
    """Return a Decord-safe path without copying on normal NTFS volumes.

    Decord's Windows filename bridge can turn non-ASCII paths into surrogate
    characters. An ASCII hard-link keeps direct range sampling zero-copy; the
    copy fallback covers filesystems that do not support hard links.
    """

    if os.name != "nt" or path.isascii():
        return path
    cached = _ALIAS_CACHE.get(path)
    if cached and Path(cached).is_file():
        return cached
    suffix = Path(path).suffix.lower()
    if not suffix.isascii() or len(suffix) > 10:
        suffix = ".video"
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    alias = Path(_ALIAS_DIR.name) / f"{digest}{suffix}"
    if not alias.is_file():
        try:
            os.link(path, alias)
        except OSError:
            shutil.copy2(path, alias)
    _ALIAS_CACHE[path] = str(alias)
    return str(alias)


def video_reader(path: str) -> VideoReader:
    """Reuse open decoders while indexing consecutive spans of one source."""

    safe_path = decoder_path(path)
    cached = _VIDEO_READER_CACHE.get(safe_path)
    if cached is not None:
        _VIDEO_READER_CACHE.move_to_end(safe_path)
        return cached
    reader = VideoReader(safe_path)
    _VIDEO_READER_CACHE[safe_path] = reader
    if len(_VIDEO_READER_CACHE) > _MAX_CACHED_VIDEO_READERS:
        _VIDEO_READER_CACHE.popitem(last=False)
    return reader


def video_frame_indices(
    reader: VideoReader,
    path: str,
    max_frames: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> np.ndarray:
    """Return the evenly spaced source-frame indexes for one video span."""

    if len(reader) == 0:
        raise ValueError(f"video contains no frames: {path}")
    fps = max(float(reader.get_avg_fps()), 0.001)
    first = 0 if start_time is None else max(0, int(start_time * fps))
    last = len(reader) - 1
    if end_time is not None:
        last = min(last, max(first, int(end_time * fps) - 1))
    if first >= len(reader):
        raise ValueError(f"video span starts beyond the file: {path} @ {start_time}")
    count = min(max_frames, last - first + 1)
    return np.linspace(first, last, count, dtype=int)


def video_frames(
    path: str,
    max_frames: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[Image.Image]:
    """Decode evenly spaced frames and return PIL objects.

    Passing frame objects avoids the upstream file:// handling mismatch between
    qwen-vl-utils and decord on Windows.
    """

    reader = video_reader(path)
    indices = video_frame_indices(
        reader, path, max_frames, start_time, end_time
    )
    return [Image.fromarray(frame) for frame in reader.get_batch(indices).asnumpy()]


def video_span_payloads(spans: list[dict], max_frames: int) -> list[dict]:
    """Decode all spans from each source with one batched decoder request."""

    payloads: list[dict] = [{} for _ in spans]
    spans_by_path: dict[str, list[tuple[int, dict]]] = {}
    for position, item in enumerate(spans):
        spans_by_path.setdefault(item["path"], []).append((position, item))

    for path, source_spans in spans_by_path.items():
        reader = video_reader(path)
        all_indices: list[int] = []
        slices: list[tuple[int, int, int]] = []
        for position, item in source_spans:
            indices = video_frame_indices(
                reader,
                path,
                max_frames,
                float(item["start_time"]),
                float(item["end_time"]),
            )
            start = len(all_indices)
            all_indices.extend(indices.tolist())
            slices.append((position, start, len(indices)))

        frames = reader.get_batch(np.asarray(all_indices, dtype=int)).asnumpy()
        for position, start, count in slices:
            payloads[position] = {
                "video": [
                    Image.fromarray(frame)
                    for frame in frames[start : start + count]
                ]
            }
    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=230_400)
    parser.add_argument("--total-pixels", type=int, default=1_843_200)
    return parser.parse_args()


def main() -> int:
    # Rust speaks UTF-8 JSON-lines. Python otherwise inherits the Windows ANSI
    # pipe encoding, which corrupts CJK paths into surrogate characters.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="strict")
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
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
            elif operation == "video_batch":
                payloads = [
                    {"video": video_frames(path, args.max_frames)}
                    for path in request["paths"]
                ]
                embeddings = [
                    value.float().cpu().tolist()
                    for value in model.process(payloads)
                ]
                emit(
                    {
                        "ok": True,
                        "embeddings": embeddings,
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000
                        ),
                    }
                )
                continue
            elif operation == "video_span_batch":
                payloads = video_span_payloads(
                    request["spans"], args.max_frames
                )
                embeddings = [
                    value.float().cpu().tolist()
                    for value in model.process(payloads)
                ]
                emit(
                    {
                        "ok": True,
                        "embeddings": embeddings,
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000
                        ),
                    }
                )
                continue
            elif operation == "text_batch":
                payloads = [
                    {"text": text, "instruction": RETRIEVAL_INSTRUCTION}
                    for text in request["texts"]
                ]
                embeddings = [
                    value.float().cpu().tolist()
                    for value in model.process(payloads)
                ]
                emit(
                    {
                        "ok": True,
                        "embeddings": embeddings,
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000
                        ),
                    }
                )
                continue
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
