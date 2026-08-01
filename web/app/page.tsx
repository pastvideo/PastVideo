"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

const PUBLIC_BASE = (process.env.NEXT_PUBLIC_PASTVIDEO_BASE ?? "")
  .trim()
  .replace(/\/$/, "");
const API_ORIGIN = (process.env.NEXT_PUBLIC_PASTVIDEO_API ?? PUBLIC_BASE)
  .trim()
  .replace(/\/$/, "");

const EXAMPLE_QUERIES = [
  "an archer shooting an arrow",
  "a person bowling",
  "people flying a kite",
  "an athlete doing the high jump",
  "a marching band",
];

type ApiStatus = {
  ready: boolean;
  total_chunks: number;
  source_files: number;
};

type VideoItem = {
  media_id: string;
  filename: string;
  media_url: string;
};

type SearchResult = VideoItem & {
  rank: number;
  score: number;
  start_time: number;
  end_time: number;
};

type SearchResponse = {
  query: string;
  elapsed_ms: number;
  results: SearchResult[];
};

const formatTime = (seconds: number) => {
  const rounded = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const rest = rounded % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `${minutes}:${String(rest).padStart(2, "0")}`;
};

const apiUrl = (path: string) => `${API_ORIGIN}${path}`;

export default function Home() {
  const [status, setStatus] = useState<ApiStatus | null>(null);
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogSelection, setCatalogSelection] = useState<VideoItem | null>(null);
  const [query, setQuery] = useState("an archer shooting an arrow");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedClip, setSavedClip] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const catalogPlayerRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const load = async () => {
      const [statusResult, videosResult] = await Promise.allSettled([
        fetch(apiUrl("/api/status")).then(async (res) => {
          if (!res.ok) throw new Error("The local search API is not ready.");
          return (await res.json()) as ApiStatus;
        }),
        fetch(apiUrl("/api/videos")).then(async (res) => {
          if (!res.ok) throw new Error("Could not load the video library.");
          return (await res.json()) as VideoItem[];
        }),
      ]);

      if (statusResult.status === "fulfilled") setStatus(statusResult.value);
      if (videosResult.status === "fulfilled") {
        setVideos(videosResult.value);
        setCatalogSelection(videosResult.value[0] ?? null);
      } else {
        setCatalogError(videosResult.reason instanceof Error ? videosResult.reason.message : "Could not load the video library.");
      }
      setCatalogLoading(false);
    };
    void load();
  }, []);

  const selectedVideoUrl = useMemo(() => {
    if (!selected) return null;
    return `${apiUrl(selected.media_url)}#t=${selected.start_time},${selected.end_time}`;
  }, [selected]);

  const runSearch = async (nextQuery?: string) => {
    const value = (nextQuery ?? query).trim();
    if (!value || isSearching) return;
    setQuery(value);
    setIsSearching(true);
    setError(null);
    setSavedClip(null);
    try {
      const res = await fetch(apiUrl("/api/search"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: value, results: 6, dedupe: 0.98 }),
      });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error ?? "Search failed.");
      const data = payload as SearchResponse;
      setResponse(data);
      setSelected(data.results[0] ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Search failed.");
    } finally {
      setIsSearching(false);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void runSearch();
  };

  const playMoment = () => {
    if (!selected || !videoRef.current) return;
    videoRef.current.currentTime = selected.start_time;
    void videoRef.current.play();
  };

  const saveClip = async () => {
    if (!selected || isSaving) return;
    setIsSaving(true);
    setError(null);
    try {
      const res = await fetch(apiUrl("/api/clip"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          media_id: selected.media_id,
          start_time: selected.start_time,
          end_time: selected.end_time,
        }),
      });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error ?? "Could not save the clip.");
      setSavedClip(apiUrl(payload.clip_url));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not save the clip.");
    } finally {
      setIsSaving(false);
    }
  };

  const selectCatalogVideo = (video: VideoItem) => {
    setCatalogSelection(video);
    requestAnimationFrame(() => {
      const player = catalogPlayerRef.current;
      if (!player) return;
      player.currentTime = 0;
      void player.play().catch(() => undefined);
    });
  };

  return (
    <main>
      <section className="search-zone" aria-labelledby="search-heading">
        <div className="search-label-row">
          <label id="search-heading" htmlFor="video-query">SEARCH THE FOOTAGE</label>
          <span className={status?.ready ? "status-ready" : ""}>
            {status?.ready ? `${status.total_chunks} MOMENTS / ${status.source_files} VIDEOS` : "CONNECTING"}
          </span>
        </div>
        <form className="search-box" onSubmit={submit}>
          <span className="search-glyph" aria-hidden="true" />
          <input
            id="video-query"
            data-testid="search-input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="e.g. an athlete clearing the high-jump bar"
            autoComplete="off"
          />
          <button data-testid="search-button" type="submit" disabled={isSearching}>
            {isSearching ? "SEARCHING…" : "SEARCH"}<span aria-hidden="true">↗</span>
          </button>
        </form>
        <div className="suggestions" aria-label="Example searches">
          <span>TRY</span>
          {EXAMPLE_QUERIES.map((example) => (
            <button key={example} type="button" onClick={() => void runSearch(example)}>{example}</button>
          ))}
        </div>
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      <section className="workspace" aria-live="polite">
        <div className="viewer">
          {selected && selectedVideoUrl ? (
            <>
              <div className="video-shell">
                <video
                  key={selectedVideoUrl}
                  ref={videoRef}
                  src={selectedVideoUrl}
                  controls
                  preload="metadata"
                  onLoadedMetadata={(event) => { event.currentTarget.currentTime = selected.start_time; }}
                />
                <div className="video-badge">MATCH {String(selected.rank).padStart(2, "0")}</div>
              </div>
              <div className="viewer-meta">
                <div><p>{selected.filename}</p><strong>{formatTime(selected.start_time)} — {formatTime(selected.end_time)}</strong></div>
                <div className="viewer-actions">
                  <button className="secondary-button" type="button" onClick={playMoment}>▶ Play moment</button>
                  <button className="secondary-button" type="button" onClick={() => void saveClip()} disabled={isSaving}>{isSaving ? "Saving…" : "↓ Save clip"}</button>
                </div>
              </div>
              {savedClip && <a className="saved-clip" href={savedClip} download>Clip ready — download MP4 ↘</a>}
            </>
          ) : (
            <div className="empty-viewer"><div className="empty-symbol" aria-hidden="true">⌁</div><p>Your best match will play here.</p></div>
          )}
        </div>

        <aside className="results-panel" aria-label="Search results">
          <div className="results-header"><div><span>RANKED MOMENTS</span><b>{response ? `${response.results.length} FOUND` : "WAITING"}</b></div>{response && <span>{response.elapsed_ms} MS</span>}</div>
          {response ? (
            <ol className="results-list" data-testid="results-list">
              {response.results.map((result) => (
                <li key={`${result.media_id}-${result.start_time}`}>
                  <button type="button" className={selected?.media_id === result.media_id && selected.start_time === result.start_time ? "selected" : ""} onClick={() => { setSelected(result); setSavedClip(null); }}>
                    <span className="rank">{String(result.rank).padStart(2, "0")}</span>
                    <span className="result-main"><strong>{formatTime(result.start_time)} — {formatTime(result.end_time)}</strong><small>{result.filename}</small></span>
                    <span className="score"><small>MATCH</small>{(result.score * 100).toFixed(1)}</span>
                  </button>
                </li>
              ))}
            </ol>
          ) : (
            <div className="results-empty"><p>Run a search to rank the indexed timeline.</p></div>
          )}
        </aside>
      </section>

      <section className="library" aria-labelledby="library-heading">
        <div className="section-heading">
          <div><span>VIDEO LIBRARY</span><h2 id="library-heading">All videos</h2></div>
          <b>{catalogLoading ? "LOADING" : `${videos.length} VIDEOS`}</b>
        </div>
        <div className="library-layout">
          <div className="video-list" data-testid="video-list" aria-label="All indexed videos">
            {catalogLoading && <p className="library-message">Loading videos…</p>}
            {catalogError && <p className="library-message error" role="alert">{catalogError}</p>}
            {!catalogLoading && !catalogError && videos.length === 0 && <p className="library-message">No videos indexed.</p>}
            {videos.map((video) => (
              <button
                key={video.media_id}
                type="button"
                className={`video-list-item ${catalogSelection?.media_id === video.media_id ? "selected" : ""}`}
                onClick={() => selectCatalogVideo(video)}
                aria-label={`Play ${video.filename}`}
              >
                <span className="thumbnail-shell">
                  <video
                    src={`${apiUrl(video.media_url)}#t=0.1`}
                    muted
                    playsInline
                    preload="metadata"
                    tabIndex={-1}
                    onLoadedMetadata={(event) => { event.currentTarget.currentTime = Math.min(0.1, event.currentTarget.duration || 0); }}
                  />
                  <span className="thumbnail-play" aria-hidden="true">▶</span>
                </span>
                <span className="video-list-name">{video.filename}</span>
              </button>
            ))}
          </div>
          <div className="library-player">
            {catalogSelection ? (
              <>
                <video key={catalogSelection.media_id} ref={catalogPlayerRef} src={apiUrl(catalogSelection.media_url)} controls preload="metadata" />
                <div className="library-player-meta"><span>NOW PLAYING</span><strong>{catalogSelection.filename}</strong></div>
              </>
            ) : (
              <div className="library-player-empty">Select a video to play it.</div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
