# Windows distribution

PastVideo's Windows release is designed for an end user who has no development
tools installed. After extracting the main ZIP, `PastVideo.exe` starts directly.
FFmpeg is bundled with the application, and the optional local-GPU stack is
installed by PastVideo itself only when it is needed.

## Release assets

The `v0.2.0` release contains these assets:

| Asset | Purpose | Approximate compressed size | SHA-256 |
| --- | --- | ---: | --- |
| `PastVideo-v0.2.0-win-x64.zip` | Native app, FFmpeg, worker, model metadata, docs | generated during release | recorded in the release notes |
| `PastVideo-AI-Runtime-Core-win-x64.zip` | Portable CPython and Python packages | 0.24 GiB | `b9b10c80e85c878c21d33c2953da60e8fa2589a462d19fb27d0d1402f09f5ac8` |
| `PastVideo-AI-Runtime-CUDA-1-win-x64.zip` | First half of CUDA/PyTorch native libraries | 1.19 GiB | `cc966215d73fc7d3191a0595990d5d9a69f1fd51b9f74e610db820e7d4370881` |
| `PastVideo-AI-Runtime-CUDA-2-win-x64.zip` | Second half of CUDA/PyTorch native libraries | 1.26 GiB | `14571fe630e92bbd98ca8d9a53ca9eb49507dcd6e55fa09c7091b9152f978697` |

The CUDA runtime is split because GitHub limits an individual Release asset to
2 GiB. PastVideo downloads all three runtime pieces, verifies each archive, and
extracts them into one atomic installation.

The separate Qwen3-VL-Embedding-2B model weight is 4,255,140,312 bytes. Its
expected SHA-256 is
`c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`.
PastVideo downloads it from the model publisher, or the user can select an
already downloaded `model.safetensors` file. Small configuration and tokenizer
files ship in the main PastVideo ZIP.

## Why the Python runtime is separate

The current Qwen implementation depends on CUDA-enabled PyTorch and the model's
reference Python code. A normal virtual environment cannot be moved because it
contains absolute paths, while bundling the complete environment in the main ZIP
would make every application update several gigabytes.

The release builder therefore creates a relocatable CPython distribution with
exact package versions. This preserves the retrieval behavior already tested in
PastVideo, avoids requiring Python or a CUDA Toolkit, and lets application
updates remain small. Replacing the inference engine is not necessary for this
release and would introduce model-conversion and result-quality risk.

## Installation behavior

1. Clicking **Index new videos** checks the selected provider.
2. If local GPU indexing is selected and required files are missing, the native
   **Prepare local AI** dialog opens instead of failing.
3. The user may download inside PastVideo, copy direct URLs into another download
   manager, or select an existing model weight file.
4. Downloads use `.download` partial files, resume with HTTP Range requests, and
   verify SHA-256 before extraction or activation.
5. Runtime replacement is staged and renamed atomically. A failed update leaves
   the previous complete runtime available.
6. Once both components are ready, **Continue indexing** resumes the original
   action.

The UI chooses English, Simplified Chinese, or Traditional Chinese from the
Windows locale on first launch. The language can be changed live in Settings and
is persisted independently of model/index configuration.

## Build and verify

```powershell
# Build and smoke-test portable CPython/PyTorch on the local NVIDIA GPU.
.\scripts\build_portable_runtime.ps1

# Recreate only the three runtime ZIPs after code/signature review.
.\scripts\build_portable_runtime.ps1 -ArchiveOnly

# Build the native application package.
.\scripts\package_windows.ps1 -CreateArchive

# Repository verification.
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

Before publishing, test the packaged executable with legacy development paths
disabled (`PASTVIDEO_DISABLE_LEGACY_AI=1`). Confirm that the first Index action
opens the AI preparation dialog, all three languages render correctly, the
runtime Python can import CUDA PyTorch, and the worker reports the detected GPU.

## Updating a release

When runtime dependencies change, rebuild all runtime archives, copy their new
hashes into `src/distribution.rs`, and publish them under the same version used
by the hard-coded asset URLs. Never reuse an old asset name with different
bytes. Application-only updates can reuse an unchanged versioned runtime.
