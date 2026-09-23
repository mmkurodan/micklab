// VOSK 文字起こしアプリのオフライン用サービスワーカー。
// アプリシェル（HTML/JS/アイコン/マニフェスト）と vosk-browser の WASM バンドルを
// キャッシュする。VOSK モデル本体（数十MB）は vosk-browser が IndexedDB(IDBFS) に
// 独自キャッシュするため、ここでは扱わない。
const CACHE = "vosk-transcribe-v2";
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

// モデル ZIP をストリーミングしながらバイト数を計測し、BroadcastChannel で通知する。
// Web Worker 内の fetch も SW が横取りできるため、vosk-browser のワーカー経由の
// ダウンロードにも有効。
async function trackZipDownload(request) {
  const res = await fetch(request);
  if (!res.ok || !res.body) return res;

  const contentLength = res.headers.get("Content-Length");
  const total = contentLength ? parseInt(contentLength, 10) : 0;
  let received = 0;
  let lastNotify = 0;

  const channel = new BroadcastChannel("vosk-dl-progress");
  const reader = res.body.getReader();

  const stream = new ReadableStream({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) {
        channel.postMessage({ type: "done", received, total });
        channel.close();
        controller.close();
        return;
      }
      received += value.byteLength;
      // 100ms ごとに通知してメッセージ数を抑える。
      const now = Date.now();
      if (now - lastNotify > 100) {
        channel.postMessage({ type: "progress", received, total });
        lastNotify = now;
      }
      controller.enqueue(value);
    },
  });

  return new Response(stream, {
    headers: res.headers,
    status: res.status,
    statusText: res.statusText,
  });
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  const host = url.hostname;

  // モデル ZIP（HuggingFace / CDN）は進捗追跡しながら透過する。
  const isModelZip =
    request.url.includes(".zip") &&
    (host.includes("huggingface.co") ||
      host.includes("hf.co") ||
      host.includes("alphacephei.com"));

  if (isModelZip) {
    event.respondWith(trackZipDownload(request));
    return;
  }

  // その他の HuggingFace / CDN リクエストはキャッシュせずネットワークに任せる。
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
