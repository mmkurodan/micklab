// コードジェネレータ（/codegen.html）をオフラインでも開けるようにするサービスワーカー。
// 登録スコープは "/codegen.html" のみで、サイトの他のページには影響しない。
// キャッシュするのはページ本体・マニフェスト・アイコンだけで、入力値や生成コードには一切触れない。
"use strict";

const CACHE = "codegen-v1";
const PAGE = "/codegen.html";
const ASSETS = [
  PAGE,
  "/codegen/manifest.webmanifest",
  "/codegen/icon-192.png",
  "/codegen/icon-512.png",
  "/codegen/icon-maskable-512.png",
  "/codegen/apple-touch-icon.png",
  "/codegen/favicon-32.png",
];

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      // ページ本体が取得できなければインストール失敗とする（アイコン類は失敗しても続行）
      await cache.add(new Request(PAGE, { cache: "reload" }));
      await Promise.allSettled(
        ASSETS.filter((url) => url !== PAGE).map((url) => cache.add(new Request(url, { cache: "reload" })))
      );
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys.filter((key) => key.startsWith("codegen-") && key !== CACHE).map((key) => caches.delete(key))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || !ASSETS.includes(url.pathname)) return;

  // ネットワーク優先: オンラインなら常にサイト上の最新版を使い、キャッシュも更新する。
  // 取得できないとき（オフライン等）だけ、前回保存した版を返す。
  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);
      try {
        const response = await fetch(url.pathname, { cache: "no-cache" });
        if (!response.ok) throw new Error("HTTP " + response.status);
        await cache.put(url.pathname, response.clone());
        return response;
      } catch (error) {
        const cached = await cache.match(url.pathname);
        if (cached) return cached;
        throw error;
      }
    })()
  );
});
