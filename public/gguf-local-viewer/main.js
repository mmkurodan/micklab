import { LoggerWithoutDebug, Wllama } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/index.js";

const gpuStatusEl = document.getElementById("gpuStatus");
const modelStatusEl = document.getElementById("modelStatus");
const loadModelBtn = document.getElementById("loadModelBtn");
const promptInputEl = document.getElementById("promptInput");
const runBtn = document.getElementById("runBtn");
const responseEl = document.getElementById("response");

const MODEL_URL =
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true";
const MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf";
const WLLAMA_ASSETS = {
  "single-thread/wllama.wasm":
    "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/single-thread/wllama.wasm",
  "multi-thread/wllama.wasm":
    "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/multi-thread/wllama.wasm",
};

const DB_NAME = "gguf-viewer-cache";
const STORE_NAME = "models";
const DB_VERSION = 1;
const MODEL_CACHE_KEY = MODEL_URL;

const state = {
  wllama: null,
  model: null,
};

function setStatus(el, message, isError = false) {
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function getErrorMessage(error) {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** power;
  return `${value.toFixed(power === 0 ? 0 : 1)} ${units[power]}`;
}

function openCacheDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("IndexedDB open failed"));
  });
}

async function getCachedModel(id) {
  const db = await openCacheDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).get(id);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error || new Error("IndexedDB read failed"));
  });
}

async function putCachedModel(record) {
  const db = await openCacheDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(record);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error("IndexedDB write failed"));
  });
}

async function downloadModelFromUrl(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`モデルのダウンロードに失敗しました (${response.status})`);
  }
  return response.arrayBuffer();
}

async function resolveModelBytes() {
  const cached = await getCachedModel(MODEL_CACHE_KEY);
  if (cached && cached.bytes) {
    setStatus(
      modelStatusEl,
      `IndexedDBキャッシュを使用: ${MODEL_NAME} (${formatBytes(cached.size || cached.bytes.byteLength)})`
    );
    return { bytes: cached.bytes, source: "cache" };
  }

  setStatus(modelStatusEl, `モデルをダウンロード中: ${MODEL_NAME}`);
  const bytes = await downloadModelFromUrl(MODEL_URL);
  await putCachedModel({
    id: MODEL_CACHE_KEY,
    name: MODEL_NAME,
    url: MODEL_URL,
    size: bytes.byteLength,
    updatedAt: Date.now(),
    bytes,
  });
  setStatus(modelStatusEl, `ダウンロード完了。IndexedDBに保存: ${MODEL_NAME} (${formatBytes(bytes.byteLength)})`);
  return { bytes, source: "download" };
}

async function loadModel() {
  loadModelBtn.disabled = true;
  runBtn.disabled = true;
  setStatus(modelStatusEl, "モデル準備中...");
  try {
    const { bytes, source } = await resolveModelBytes();
    if (state.wllama) {
      await state.wllama.exit();
      state.wllama = null;
    }

    setStatus(modelStatusEl, "推論エンジンを初期化中...");
    const blob = new Blob([bytes], { type: "application/octet-stream" });
    state.wllama = new Wllama(WLLAMA_ASSETS, { logger: LoggerWithoutDebug });
    await state.wllama.loadModel([blob], { n_ctx: 2048 });

    state.model = {
      key: MODEL_CACHE_KEY,
      name: MODEL_NAME,
      size: bytes.byteLength,
      source,
    };
    setStatus(
      modelStatusEl,
      `モデル準備完了 (${source === "cache" ? "IndexedDB" : "ダウンロード"}) : ${MODEL_NAME}`
    );
    runBtn.disabled = false;
  } catch (error) {
    setStatus(modelStatusEl, `モデル読み込み失敗: ${getErrorMessage(error)}`, true);
  } finally {
    loadModelBtn.disabled = false;
  }
}

async function runInference() {
  const prompt = promptInputEl.value.trim();
  if (!state.wllama || !state.model) {
    setStatus(responseEl, "先にモデルを読み込んでください。", true);
    return;
  }
  if (!prompt) {
    setStatus(responseEl, "プロンプトを入力してください。", true);
    return;
  }

  runBtn.disabled = true;
  setStatus(responseEl, "推論中...");
  try {
    let streamedText = "";
    const output = await state.wllama.createChatCompletion(
      [{ role: "user", content: prompt }],
      {
        nPredict: 256,
        useCache: false,
        sampling: {
          temp: 0.7,
          top_k: 40,
          top_p: 0.9,
        },
        onNewToken: (_token, _piece, currentText) => {
          streamedText = currentText;
          setStatus(responseEl, currentText || "推論中...");
        },
      }
    );
    setStatus(responseEl, output || streamedText || "(空の応答)");
  } catch (error) {
    setStatus(responseEl, `推論エラー: ${getErrorMessage(error)}`, true);
  } finally {
    runBtn.disabled = false;
  }
}

function bootstrap() {
  setStatus(gpuStatusEl, "WASM/llama.cpp 推論モードで起動しました。");
  setStatus(modelStatusEl, "モデル未ロード。ボタン押下でURLから取得し、IndexedDBキャッシュを再利用します。");
  loadModelBtn.disabled = false;
  runBtn.disabled = true;

  loadModelBtn.addEventListener("click", () => {
    loadModel();
  });
  runBtn.addEventListener("click", () => {
    runInference();
  });
}

bootstrap();
