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
import subprocess
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from PIL import Image


RETRIEVAL_INSTRUCTION = "Retrieve video clips relevant to the user's query."
_ALIAS_DIR = tempfile.TemporaryDirectory(prefix="pastvideo-qwen-")
_ALIAS_CACHE: dict[str, str] = {}
_VIDEO_READER_CACHE: OrderedDict[str, VideoReader] = OrderedDict()
_VIDEO_METADATA_CACHE: dict[str, tuple[int, int, str]] = {}
_MAX_CACHED_VIDEO_READERS = 2
_VIDEO_PATCH_FACTOR = 64
_VIDEO_MIN_PIXELS = 128 * 32 * 32
_VIDEO_MAX_PIXELS = 768 * 32 * 32
_RESIZE_WORKERS = max(
    1,
    min(
        8,
        int(os.environ.get("PASTVIDEO_QWEN_RESIZE_THREADS", os.cpu_count() or 4)),
    ),
)
_DECODE_WORKERS = max(
    1,
    min(2, int(os.environ.get("PASTVIDEO_QWEN_DECODE_WORKERS", "2"))),
)
_HW_DECODE_SETTING = os.environ.get(
    "PASTVIDEO_QWEN_HW_DECODE", "auto"
).strip().lower()
_HW_DECODE_MIN_PIXELS = int(
    os.environ.get("PASTVIDEO_QWEN_HW_DECODE_MIN_PIXELS", "3000000")
)
_CUDA_DECODE_ENABLED = False
_SUBPROCESS_FLAGS = (
    getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
)


def find_ffmpeg() -> Path | None:
    executable = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates = [
        os.environ.get("PASTVIDEO_FFMPEG"),
        Path(__file__).resolve().parent.parent / ".tools" / "ffmpeg" / "bin" / executable,
        shutil.which(executable),
        shutil.which("ffmpeg"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    return None


_FFMPEG = find_ffmpeg()


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


def run_subprocess(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    if os.name == "nt":
        kwargs["creationflags"] = _SUBPROCESS_FLAGS
    return subprocess.run(command, **kwargs)


def video_metadata(path: str) -> tuple[int, int, str]:
    """Read the primary video shape/codec without decoding full frames."""

    cached = _VIDEO_METADATA_CACHE.get(path)
    if cached is not None:
        return cached
    if _FFMPEG is None:
        raise FileNotFoundError("ffmpeg was not found for hardware decoding")
    ffprobe = _FFMPEG.with_name(
        "ffprobe.exe" if os.name == "nt" else "ffprobe"
    )
    if not ffprobe.is_file():
        raise FileNotFoundError(f"ffprobe was not found beside {_FFMPEG}")
    result = run_subprocess(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "json",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if not streams:
        raise ValueError(f"video stream metadata was not found: {path}")
    stream = streams[0]
    metadata = (
        int(stream["width"]),
        int(stream["height"]),
        str(stream.get("codec_name", "")).lower(),
    )
    _VIDEO_METADATA_CACHE[path] = metadata
    return metadata


def should_use_cuda_decode(path: str) -> bool:
    if not _CUDA_DECODE_ENABLED or _FFMPEG is None:
        return False
    if _HW_DECODE_SETTING in {"0", "false", "off", "no"}:
        return False
    width, height, codec = video_metadata(path)
    if _HW_DECODE_SETTING in {"1", "true", "on", "yes", "force"}:
        return True
    return (
        width * height >= _HW_DECODE_MIN_PIXELS
        and codec in {"av1", "h264", "hevc", "mpeg2video", "vp9"}
    )


def cuda_video_frames(
    path: str,
    max_frames: int,
    total_pixels: int,
    start_time: float,
    end_time: float,
) -> list[Image.Image]:
    """Decode and shrink one high-resolution span through NVIDIA NVDEC."""

    if _FFMPEG is None:
        raise FileNotFoundError("ffmpeg was not found for hardware decoding")
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(f"invalid video span: {start_time}..{end_time}")
    width, height, _ = video_metadata(path)
    target_height, target_width = video_frame_dimensions(
        height, width, max_frames, total_pixels
    )
    filter_graph = (
        f"fps={max_frames}/{duration:.9f},"
        f"scale_cuda={target_width}:{target_height}:format=nv12,"
        "hwdownload,format=nv12,format=rgb24"
    )
    result = run_subprocess(
        [
            str(_FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{start_time:.6f}",
            "-t",
            f"{duration:.6f}",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            path,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            filter_graph,
            "-frames:v",
            str(max_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=max(60.0, duration * 3),
    )
    frame_bytes = target_width * target_height * 3
    if len(result.stdout) % frame_bytes != 0:
        raise ValueError(
            f"hardware decoder returned {len(result.stdout)} malformed bytes"
        )
    count = len(result.stdout) // frame_bytes
    if count == 0:
        raise ValueError(f"hardware decoder returned no frames: {path}")
    return [
        Image.frombytes(
            "RGB",
            (target_width, target_height),
            result.stdout[offset : offset + frame_bytes],
        )
        for offset in range(0, len(result.stdout), frame_bytes)
    ]


def cuda_span_payloads(
    source_spans: list[tuple[int, dict]],
    max_frames: int,
    total_pixels: int,
) -> list[tuple[int, dict]]:
    """Keep two NVDEC sessions fed without overwhelming decoder memory."""

    def decode(item: tuple[int, dict]) -> tuple[int, dict]:
        position, span = item
        frames = cuda_video_frames(
            span["path"],
            max_frames,
            total_pixels,
            float(span["start_time"]),
            float(span["end_time"]),
        )
        return position, {"video": frames}

    with ThreadPoolExecutor(
        max_workers=min(_DECODE_WORKERS, len(source_spans))
    ) as executor:
        return list(executor.map(decode, source_spans))


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


def video_frame_dimensions(
    height: int,
    width: int,
    frame_count: int,
    total_pixels: int,
) -> tuple[int, int]:
    """Choose a Qwen-native frame size before expensive Torch preprocessing.

    qwen-vl-utils first aligns list-backed video frames to 64 pixels and later
    aligns video tensors to 32. Returning a 64-aligned size inside the model's
    video pixel budget makes both downstream resize passes no-ops while keeping
    the source aspect ratio close.
    """

    if height <= 0 or width <= 0 or frame_count <= 0:
        raise ValueError("video frame dimensions and count must be positive")
    max_pixels = max(
        min(_VIDEO_MAX_PIXELS, total_pixels / frame_count * 2),
        int(_VIDEO_MIN_PIXELS * 1.05),
    )
    source_ratio = width / height
    max_units = max(1, int(max_pixels // (_VIDEO_PATCH_FACTOR**2)))
    min_units = max(1, int(np.ceil(_VIDEO_MIN_PIXELS / (_VIDEO_PATCH_FACTOR**2))))
    best: tuple[float, int, int, int] | None = None
    for height_units in range(1, max_units + 1):
        for width_units in range(1, max_units // height_units + 1):
            area_units = height_units * width_units
            if area_units < min_units:
                continue
            ratio = width_units / height_units
            aspect_error = abs(np.log(ratio / source_ratio))
            area_penalty = 0.05 * (1 - area_units / max_units)
            candidate = (
                float(aspect_error + area_penalty),
                -area_units,
                height_units,
                width_units,
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        scale = (max_pixels / (height * width)) ** 0.5
        return max(2, int(height * scale)), max(2, int(width * scale))
    return best[2] * _VIDEO_PATCH_FACTOR, best[3] * _VIDEO_PATCH_FACTOR


def resize_video_frames(
    frames: np.ndarray,
    frame_count: int,
    total_pixels: int,
) -> list[Image.Image]:
    """Resize decoded frames in Pillow's native threads and detach the batch."""

    if len(frames) == 0:
        return []
    target_height, target_width = video_frame_dimensions(
        int(frames.shape[1]),
        int(frames.shape[2]),
        frame_count,
        total_pixels,
    )

    return resize_frame_tasks(
        [(frame, target_height, target_width) for frame in frames]
    )


def resize_frame_tasks(
    tasks: list[tuple[np.ndarray, int, int]],
) -> list[Image.Image]:
    """Resize differently sized span groups through one shared native pool."""

    def resize(task: tuple[np.ndarray, int, int]) -> Image.Image:
        frame, target_height, target_width = task
        image = Image.fromarray(frame)
        if image.size == (target_width, target_height):
            return image.copy()
        return image.resize(
            (target_width, target_height), Image.Resampling.BILINEAR
        )

    with ThreadPoolExecutor(
        max_workers=min(_RESIZE_WORKERS, len(tasks))
    ) as executor:
        return list(executor.map(resize, tasks))


def video_frames(
    path: str,
    max_frames: int,
    total_pixels: int,
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
    frames = reader.get_batch(indices).asnumpy()
    return resize_video_frames(frames, len(indices), total_pixels)


def video_span_payloads(
    spans: list[dict], max_frames: int, total_pixels: int
) -> list[dict]:
    """Decode all spans from each source with one batched decoder request."""

    payloads: list[dict] = [{} for _ in spans]
    spans_by_path: dict[str, list[tuple[int, dict]]] = {}
    for position, item in enumerate(spans):
        spans_by_path.setdefault(item["path"], []).append((position, item))

    for path, source_spans in spans_by_path.items():
        try:
            if should_use_cuda_decode(path):
                for position, payload in cuda_span_payloads(
                    source_spans, max_frames, total_pixels
                ):
                    payloads[position] = payload
                continue
        except Exception as error:
            print(
                f"PastVideo NVDEC fallback for {path!r}: {error}",
                file=sys.stderr,
                flush=True,
            )

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
        resize_tasks: list[tuple[np.ndarray, int, int]] = []
        for _, start, count in slices:
            target_height, target_width = video_frame_dimensions(
                int(frames.shape[1]),
                int(frames.shape[2]),
                count,
                total_pixels,
            )
            resize_tasks.extend(
                (frame, target_height, target_width)
                for frame in frames[start : start + count]
            )
        resized = resize_frame_tasks(resize_tasks)
        del frames
        resized_start = 0
        for position, _, count in slices:
            payloads[position] = {
                "video": resized[resized_start : resized_start + count]
            }
            resized_start += count
    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=230_400)
    parser.add_argument("--total-pixels", type=int, default=1_843_200)
    return parser.parse_args()


def embed_payloads(model, payloads: list[dict]) -> tuple[list[list[float]], int]:
    """Run one model batch and promptly release its temporary tensors."""

    inference_started = time.perf_counter()
    values = model.process(payloads)
    embeddings = [value.float().cpu().tolist() for value in values]
    inference_ms = round((time.perf_counter() - inference_started) * 1000)
    del values
    return embeddings, inference_ms


def main() -> int:
    global _CUDA_DECODE_ENABLED

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
    _CUDA_DECODE_ENABLED = (
        device == "cuda"
        and _FFMPEG is not None
        and _HW_DECODE_SETTING not in {"0", "false", "off", "no"}
    )
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
            "video_decoder": (
                "adaptive-nvdec" if _CUDA_DECODE_ENABLED else "decord"
            ),
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
                    "video": video_frames(
                        request["path"], args.max_frames, args.total_pixels
                    )
                }
            elif operation == "video_batch":
                decode_started = time.perf_counter()
                payloads = [
                    {
                        "video": video_frames(
                            path, args.max_frames, args.total_pixels
                        )
                    }
                    for path in request["paths"]
                ]
                decode_ms = round(
                    (time.perf_counter() - decode_started) * 1000
                )
                embeddings, inference_ms = embed_payloads(model, payloads)
                del payloads
                gc.collect()
                emit(
                    {
                        "ok": True,
                        "embeddings": embeddings,
                        "decode_ms": decode_ms,
                        "inference_ms": inference_ms,
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000
                        ),
                    }
                )
                continue
            elif operation == "video_span_batch":
                decode_started = time.perf_counter()
                payloads = video_span_payloads(
                    request["spans"], args.max_frames, args.total_pixels
                )
                decode_ms = round(
                    (time.perf_counter() - decode_started) * 1000
                )
                embeddings, inference_ms = embed_payloads(model, payloads)
                del payloads
                gc.collect()
                emit(
                    {
                        "ok": True,
                        "embeddings": embeddings,
                        "decode_ms": decode_ms,
                        "inference_ms": inference_ms,
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
                embeddings, inference_ms = embed_payloads(model, payloads)
                del payloads
                gc.collect()
                emit(
                    {
                        "ok": True,
                        "embeddings": embeddings,
                        "inference_ms": inference_ms,
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
