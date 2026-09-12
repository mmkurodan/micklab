// VOSK 文字起こしアプリのオフライン用サービスワーカー。
// アプリシェル（HTML/JS/アイコン/マニフェスト）と vosk-browser の WASM バンドルを
// キャッシュする。VOSK モデル本体（数十MB）は vosk-browser が IndexedDB(IDBFS) に
// 独自キャッシュするため、ここでは扱わない。
const CACHE = "vosk-transcribe-v1";
const VOSK_SRC = "https://cdn.jsdelivr.net/npm/vosk-browser@0.0.8/dist/vosk.js";
const SHELL = [
  "./",
  "./index.html",
  "./index-en.html",
  "./main.js",
  "./manifest.webmanifest",
  "./manifest-en.webmanifest",
  "./icon-192.png",
  "./icon-512.png",
  "./icon-maskable-512.png",
  "./apple-touch-icon.png",
  "./favicon-32.png",
  VOSK_SRC,
];

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE).then((cache) =>
      Promise.allSettled(SHELL.map((url) => cache.add(url)))
    )
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => key.startsWith("vosk-transcribe") && key !== CACHE)
          .map((key) => caches.delete(key))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // モデル本体（HuggingFace / CDN / alphacephei）はキャッシュせずネットワークに任せる。
  const host = url.hostname;
  if (
    host.includes("huggingface.co") ||
    host.includes("hf.co") ||
    host.includes("alphacephei.com")
  ) {
    return;
  }

  const isVoskBundle = request.url === VOSK_SRC;
  const isSameOrigin = url.origin === self.location.origin;
  if (!isSameOrigin && !isVoskBundle) return;

  event.respondWith(
    (async () => {
      const cached = await caches.match(request);
      if (cached) return cached;
      try {
        const response = await fetch(request);
        if (response && response.ok) {
          const copy = response.clone();
          caches.open(CACHE).then((cache) => cache.put(request, copy)).catch(() => {});
        }
        return response;
      } catch (error) {
        if (request.mode === "navigate") {
          const fallback = await caches.match("./index.html");
          if (fallback) return fallback;
        }
        throw error;
      }
    })()
  );
});
