#!/usr/bin/env python3
"""Run PastVideo's local Caption, OCR, and Whisper analyzers for one video.

The worker prints diagnostics to stderr and exactly one machine-readable result
line to stdout. Source media never leaves the machine. Models are loaded lazily
and released between GPU-heavy stages so the default pipeline fits a 24 GB GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

RESULT_PREFIX = "PASTVIDEO_RESULT\t"
CAPTION_MODEL = os.environ.get(
    "PASTVIDEO_CAPTION_MODEL", "Qwen/Qwen3-VL-4B-Instruct"
)
WHISPER_MODEL = os.environ.get("PASTVIDEO_WHISPER_MODEL", "small")

CAPTION_PROMPT = (
    "Return only one minified JSON object using this exact schema: "
    '{"description":"one factual sentence under 20 words",'
    '"setting":"under 6 words","activities":["up to 3 short items"],'
    '"salient_objects":["up to 5 visible nouns"],'
    '"camera_motion":"static|panning|tracking|handheld|unknown"}. '
    "Use at most 55 English words total. State only visible facts; do not guess."
)


def diagnostic(message: str) -> None:
    print(f"PastVideo understanding: {message}", file=sys.stderr, flush=True)


def emit(payload: dict[str, Any]) -> None:
    # Keep the machine protocol ASCII-safe. Windows console pipes commonly use
    # a legacy code page (for example GBK), while OCR/transcripts can contain
    # arbitrary Unicode that cannot be encoded by that code page.
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--chunk-duration", type=float, default=30.0)
    parser.add_argument("--overlap", type=float, default=5.0)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--caption-model", default=CAPTION_MODEL)
    parser.add_argument("--whisper-model", default=WHISPER_MODEL)
    parser.add_argument("--caption-frames", type=int, default=4)
    parser.add_argument("--ocr-frames", type=int, default=3)
    parser.add_argument("--skip-caption", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-transcript", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def video_duration(path: str) -> float:
    import cv2

    capture = cv2.VideoCapture(path)
    try:
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frames > 0 and fps > 0:
            return frames / fps
    finally:
        capture.release()
    raise ValueError(f"could not determine video duration: {path}")


def segment_spans(
    duration: float, chunk_duration: float, overlap: float, maximum: int | None
) -> list[tuple[float, float]]:
    if chunk_duration <= 0 or overlap < 0 or overlap >= chunk_duration:
        raise ValueError("chunk duration must be positive and overlap must be smaller")
    if duration <= chunk_duration:
        spans = [(0.0, duration)]
    else:
        spans = []
        step = chunk_duration - overlap
        start = 0.0
        while start < duration:
            end = min(duration, start + chunk_duration)
            spans.append((start, end))
            start += step
            if start + overlap >= duration:
                break
    if maximum is not None:
        spans = spans[:maximum]
    return [(start, end) for start, end in spans if end > start]


def read_frames(path: str, start: float, end: float, count: int) -> list[Any]:
    import cv2

    count = max(1, count)
    capture = cv2.VideoCapture(path)
    frames: list[Any] = []
    try:
        for position in range(count):
            fraction = (position + 1) / (count + 1)
            timestamp = start + (end - start) * fraction
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            longest = max(height, width)
            if longest > 768:
                scale = 768.0 / longest
                frame = cv2.resize(
                    frame,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append((timestamp, frame))
    finally:
        capture.release()
    return frames


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {
        "description": cleaned[:800],
        "setting": "unknown",
        "activities": [],
        "salient_objects": [],
        "camera_motion": "unknown",
        "parse_fallback": True,
    }


def caption_output(
    path: str,
    spans: list[tuple[float, float]],
    model_name: str,
    frame_count: int,
    offline: bool,
) -> dict[str, Any]:
    import cv2
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    diagnostic(f"loading Caption model {model_name}")
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_name, local_files_only=offline
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa",
        local_files_only=offline,
    ).eval()
    load_seconds = time.perf_counter() - load_started
    records: list[dict[str, Any]] = []
    inference_seconds = 0.0

    for index, (start, end) in enumerate(spans):
        sampled = read_frames(path, start, end, frame_count)
        if not sampled:
            raise ValueError(f"Caption could not decode segment {index}")
        images = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for _, frame in sampled
        ]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": images},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, _ = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            fps=2.0,
        ).to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=128, do_sample=False
            )
        inference_seconds += time.perf_counter() - started
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        data = extract_json_object(text)
        records.append(
            {
                "segment_id": f"caption_{index:06}",
                "start_ms": round(start * 1000),
                "end_ms": round(end * 1000),
                "data": data,
                "metadata": {
                    "sampled_frames": len(images),
                    "raw_output": text if data.get("parse_fallback") else None,
                },
            }
        )
        diagnostic(f"Caption {index + 1}/{len(spans)} completed")
        del inputs, generated, trimmed, images

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    diagnostic(
        f"Caption model loaded in {load_seconds:.2f}s; inference used {inference_seconds:.2f}s"
    )
    return {
        "name": "scene_caption",
        "analyzer_type": "vlm_caption",
        "model_provider": "local",
        "model_name": str(model_name),
        "model_revision": "caption-schema-v1",
        "config": {
            "chunk_duration": spans[0][1] - spans[0][0] if spans else 0,
            "frame_count": frame_count,
            "prompt_revision": "compact-factual-v1",
        },
        "artifact_type": "scene_caption",
        "schema_version": 1,
        "schema_definition": {
            "description": "string",
            "setting": "string",
            "activities": "array<string>",
            "salient_objects": "array<string>",
            "camera_motion": "string",
        },
        "records": records,
    }


def normalize_ocr_text(text: str) -> str:
    return " ".join(text.casefold().split())


def ocr_output(
    path: str, spans: list[tuple[float, float]], frame_count: int
) -> dict[str, Any]:
    import cv2
    import numpy as np
    from rapidocr import RapidOCR

    diagnostic("loading RapidOCR")
    engine = RapidOCR()
    records: list[dict[str, Any]] = []
    inference_seconds = 0.0
    for index, (start, end) in enumerate(spans):
        sampled = read_frames(path, start, end, frame_count)
        observations: dict[str, dict[str, Any]] = {}
        previous_thumb: Any | None = None
        analyzed_frames = 0
        for timestamp, frame in sampled:
            thumb = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)
            )
            if previous_thumb is not None:
                difference = float(
                    np.mean(np.abs(thumb.astype(np.int16) - previous_thumb.astype(np.int16)))
                )
                if difference < 2.5:
                    continue
            previous_thumb = thumb
            analyzed_frames += 1
            started = time.perf_counter()
            result = engine(frame, text_score=0.5)
            inference_seconds += time.perf_counter() - started
            boxes = getattr(result, "boxes", None)
            texts = getattr(result, "txts", ()) or ()
            scores = getattr(result, "scores", ()) or ()
            if boxes is None:
                boxes = []
            for box, text, score in zip(boxes, texts, scores):
                normalized = normalize_ocr_text(str(text))
                if not normalized:
                    continue
                observation = {
                    "text": str(text),
                    "normalized_text": normalized,
                    "confidence": round(float(score), 5),
                    "frame_ms": round(timestamp * 1000),
                    "box": [[round(float(x)), round(float(y))] for x, y in box],
                }
                current = observations.get(normalized)
                if current is None or observation["confidence"] > current["confidence"]:
                    observations[normalized] = observation
        items = sorted(observations.values(), key=lambda item: item["frame_ms"])
        records.append(
            {
                "segment_id": f"ocr_{index:06}",
                "start_ms": round(start * 1000),
                "end_ms": round(end * 1000),
                "data": {
                    "text": " ".join(item["text"] for item in items),
                    "items": items,
                },
                "metadata": {
                    "sampled_frames": len(sampled),
                    "analyzed_frames": analyzed_frames,
                },
            }
        )
        diagnostic(f"OCR {index + 1}/{len(spans)} completed")
    diagnostic(f"OCR inference used {inference_seconds:.2f}s")
    return {
        "name": "ocr",
        "analyzer_type": "optical_character_recognition",
        "model_provider": "local",
        "model_name": "RapidOCR PP-OCRv6",
        "model_revision": "rapidocr-3.9.2",
        "config": {
            "frame_count": frame_count,
            "text_score": 0.5,
            "frame_difference_threshold": 2.5,
        },
        "artifact_type": "ocr",
        "schema_version": 1,
        "schema_definition": {"text": "string", "items": "array<object>"},
        "records": records,
    }


def transcribe_once(path: str, model_name: str, device: str, offline: bool):
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        local_files_only=offline,
    )
    segments, info = model.transcribe(
        path,
        vad_filter=True,
        word_timestamps=True,
        beam_size=5,
    )
    rows = list(segments)
    return model, rows, info


def transcript_output(
    path: str, duration: float, model_name: str, offline: bool
) -> dict[str, Any]:
    diagnostic(f"loading Whisper model {model_name}")
    started = time.perf_counter()
    device = "cuda"
    try:
        model, segments, info = transcribe_once(path, model_name, device, offline)
    except Exception as error:
        diagnostic(f"Whisper CUDA fallback to CPU: {error}")
        device = "cpu"
        model, segments, info = transcribe_once(path, model_name, device, offline)
    elapsed = time.perf_counter() - started
    diagnostic(f"Whisper inference used {elapsed:.2f}s on {device}")
    records: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if not text:
            continue
        words = [
            {
                "word": word.word,
                "start_ms": round(float(word.start or segment.start) * 1000),
                "end_ms": round(float(word.end or segment.end) * 1000),
                "probability": round(float(word.probability), 5),
            }
            for word in (segment.words or [])
        ]
        records.append(
            {
                "segment_id": f"transcript_{index:06}",
                "start_ms": max(0, round(float(segment.start) * 1000)),
                "end_ms": max(1, round(float(segment.end) * 1000)),
                "data": {
                    "text": text,
                    "language": info.language,
                    "words": words,
                },
                "metadata": {
                    "avg_logprob": float(segment.avg_logprob),
                    "no_speech_probability": float(segment.no_speech_prob),
                },
            }
        )
    if not records:
        records.append(
            {
                "segment_id": "transcript_000000",
                "start_ms": 0,
                "end_ms": max(1, round(duration * 1000)),
                "data": {"text": "", "language": None, "words": []},
                "metadata": {"no_speech": True},
            }
        )
    del model
    gc.collect()
    return {
        "name": "transcript",
        "analyzer_type": "speech_to_text",
        "model_provider": "local",
        "model_name": f"faster-whisper-{model_name}",
        "model_revision": "faster-whisper-1.2.1",
        "config": {
            "language": "auto",
            "vad": True,
            "word_timestamps": True,
            "device": device,
        },
        "artifact_type": "transcript",
        "schema_version": 1,
        "schema_definition": {
            "text": "string",
            "language": "string|null",
            "words": "array<object>",
        },
        "records": records,
    }


def mock_outputs(duration: float) -> list[dict[str, Any]]:
    end_ms = max(1, round(duration * 1000))
    common = {
        "model_provider": "local-test",
        "model_revision": "mock-v1",
        "schema_version": 1,
    }
    return [
        {
            **common,
            "name": "scene_caption",
            "analyzer_type": "vlm_caption",
            "model_name": "mock-caption",
            "config": {"mock": True},
            "artifact_type": "scene_caption",
            "schema_definition": {"description": "string"},
            "records": [
                {
                    "segment_id": "caption_000000",
                    "start_ms": 0,
                    "end_ms": end_ms,
                    "data": {
                        "description": "A presenter demonstrates PastVideo on a computer screen.",
                        "setting": "office",
                        "activities": ["software demonstration"],
                        "salient_objects": ["person", "computer"],
                        "camera_motion": "static",
                    },
                    "metadata": {},
                }
            ],
        },
        {
            **common,
            "name": "ocr",
            "analyzer_type": "optical_character_recognition",
            "model_name": "mock-ocr",
            "config": {"mock": True},
            "artifact_type": "ocr",
            "schema_definition": {"text": "string"},
            "records": [
                {
                    "segment_id": "ocr_000000",
                    "start_ms": 0,
                    "end_ms": end_ms,
                    "data": {"text": "PastVideo GPU indexing", "items": []},
                    "metadata": {},
                }
            ],
        },
        {
            **common,
            "name": "transcript",
            "analyzer_type": "speech_to_text",
            "model_name": "mock-whisper",
            "config": {"mock": True},
            "artifact_type": "transcript",
            "schema_definition": {"text": "string"},
            "records": [
                {
                    "segment_id": "transcript_000000",
                    "start_ms": 0,
                    "end_ms": end_ms,
                    "data": {
                        "text": "Today we are testing PastVideo search.",
                        "language": "en",
                        "words": [],
                    },
                    "metadata": {},
                }
            ],
        },
    ]


def main() -> int:
    args = parse_args()
    path = str(Path(args.path).resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    duration = video_duration(path)
    spans = segment_spans(
        duration, args.chunk_duration, args.overlap, args.max_segments
    )
    started = time.perf_counter()
    if args.mock:
        analyzers = [
            analyzer
            for analyzer in mock_outputs(duration)
            if not (
                (analyzer["artifact_type"] == "scene_caption" and args.skip_caption)
                or (analyzer["artifact_type"] == "ocr" and args.skip_ocr)
                or (
                    analyzer["artifact_type"] == "transcript"
                    and args.skip_transcript
                )
            )
        ]
    else:
        analyzers = []
        if not args.skip_caption:
            analyzers.append(
                caption_output(
                    path,
                    spans,
                    args.caption_model,
                    args.caption_frames,
                    args.offline,
                )
            )
        if not args.skip_ocr:
            analyzers.append(ocr_output(path, spans, args.ocr_frames))
        if not args.skip_transcript:
            analyzers.append(
                transcript_output(
                    path, duration, args.whisper_model, args.offline
                )
            )
    emit(
        {
            "ok": True,
            "source": path,
            "duration_seconds": duration,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "analyzers": analyzers,
        }
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit({"ok": False, "error": str(error)})
        raise SystemExit(1)
