# PastVideo

> An open-source video database that makes the moments inside your videos
> searchable.

PastVideo turns ordinary video folders into a library you can explore with
natural language. Add your folders, let PastVideo index them, then search for
things such as *"delivery van in traffic"* or *"a dog running on the beach."*

Your original files stay where they are. With the default local AI backend,
video analysis and search also stay on your computer.

**[Download PastVideo for Windows](../../releases/latest)**

![PastVideo library](docs/screenshots/pastvideo-library.png)

## Find the moment you remember

PastVideo searches inside videos instead of relying on filenames. Results point
to the matching moment, show the matched frame, and open the video at the right
timestamp.

![PastVideo search results](docs/screenshots/pastvideo-search.png)

From there you can play the full video, seek backward or forward, enlarge the
player, save the matched segment, or reveal the original file in Explorer.

## What PastVideo can do

- Combine several folders into one video library without moving the source
  files.
- Organize indexed videos into useful categories automatically.
- Search with everyday language while the rest of the library is still being
  indexed.
- Show thumbnails only when they are needed, keeping large libraries
  responsive.
- Play matched moments, open the full video, and export useful segments.
- Use a local NVIDIA GPU automatically when available, with CPU and external
  embedding options for other environments.
- Switch between English, Simplified Chinese, and Traditional Chinese.

## Get started on Windows

1. Download the latest Windows ZIP from
   **[Releases](../../releases/latest)**.
2. Extract it and double-click `PastVideo.exe`. There is no installer.
3. Choose **Add folder** and select one or more folders containing videos.
4. Choose **Index new videos**, then start searching as soon as the first
   videos finish.

On first use, PastVideo offers to download the local AI runtime and model if
they are not already available. You can download them inside the app, copy the
links into another download manager, or select a model file you downloaded
earlier. PastVideo remembers the setup for future launches.

The Windows package includes the native app and its media tools. You do not
need to install Python, Rust, FFmpeg, or the CUDA Toolkit.

## Local-first by default

PastVideo is designed for personal archives and private footage:

- Videos are indexed in place and are never copied into the database.
- The default model runs locally on your GPU or CPU.
- Indexes, thumbnails, and preferences live on your machine.
- Remote services are optional and only used when you select one.

## Desktop and server

PastVideo is available in two forms:

| Edition | Intended for |
| --- | --- |
| **PastVideo Desktop** | Personal libraries, browsing, playback, and clip export through a native Windows interface |
| **PastVideo Server** | Headless indexing, automation, and integrations through the CLI and HTTP API |

The desktop app is Windows-first. macOS support is planned. Both editions share
the same Rust indexing and search engine.

## Supported video files

PastVideo supports common formats including MP4, MOV, M4V, MKV, AVI, WebM, WMV,
MPG/MPEG, 3GP/3G2, FLV/F4V, OGV, and VOB.

Files without a recognized video extension are ignored. MTS and M2TS files are
intentionally not imported.

## Build from source

PastVideo is written primarily in Rust. To build the desktop app on Windows:

```powershell
.\scripts\run_desktop.ps1
```

To build the headless server and see its commands:

```powershell
cargo build --release --bin pastvideo
.\target\release\pastvideo.exe --help
```

Run the test suite with:

```powershell
cargo test --all-targets
```

Release maintainers can find packaging and runtime details in
[`docs/DISTRIBUTION.md`](docs/DISTRIBUTION.md).

## Project direction

PastVideo aims to become an open, dependable video database rather than a
closed media catalog. Current priorities are:

- faster and more accurate local indexing;
- a smoother Windows desktop experience;
- macOS support;
- easier server deployment;
- more local models and embedding providers;
- richer organization and duplicate detection.

## Contributing

Issues and pull requests are welcome. Please keep changes focused, include tests
for behavior changes, and do not commit private footage, downloaded models,
generated indexes, or API keys.

## License

PastVideo is licensed under [Apache-2.0](LICENSE).
