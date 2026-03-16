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
const DEFAULT_MODEL_ALIAS = "default";
const OLLAMA_ENDPOINT_RE = /\/api\/(tags|generate|chat)$/;
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
  inferenceInFlight: false,
  apiFetchInstalled: false,
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

function jsonResponse(payload, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...extraHeaders,
    },
  });
}

function ollamaErrorResponse(status, message) {
  return jsonResponse({ error: message }, status);
}

function methodNotAllowedResponse(allowedMethods) {
  return ollamaErrorResponse(405, `method not allowed (allowed: ${allowedMethods.join(", ")})`);
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

function getModelNameFromRequest(body) {
  const requestedModel =
    body && typeof body.model === "string" && body.model.trim() ? body.model.trim() : DEFAULT_MODEL_ALIAS;
  return requestedModel;
}

function isSupportedModelName(modelName) {
  if (!modelName) {
    return true;
  }
  if (modelName === DEFAULT_MODEL_ALIAS || modelName === MODEL_NAME) {
    return true;
  }
  return Boolean(state.model && modelName === state.model.name);
}

function getTagsResponsePayload() {
  const modifiedAt = new Date(state.model?.updatedAt || Date.now()).toISOString();
  const modelSize = state.model?.size || 0;
  return {
    models: [
      {
        name: DEFAULT_MODEL_ALIAS,
        model: DEFAULT_MODEL_ALIAS,
        modified_at: modifiedAt,
        size: modelSize,
        details: {
          format: "gguf",
          family: "llama",
          parameter_size: "unknown",
          quantization_level: "unknown",
        },
      },
    ],
  };
}

function getRequestMethod(input, init) {
  if (init && init.method) {
    return String(init.method).toUpperCase();
  }
  if (input instanceof Request) {
    return input.method.toUpperCase();
  }
  return "GET";
}

function getOllamaEndpoint(pathname) {
  const match = pathname.match(OLLAMA_ENDPOINT_RE);
  return match ? match[1] : null;
}

async function getRequestBodyText(input, init) {
  if (init && Object.prototype.hasOwnProperty.call(init, "body")) {
    const body = init.body;
    if (body == null) {
      return "";
    }
    if (typeof body === "string") {
      return body;
    }
    if (body instanceof Blob) {
      return body.text();
    }
    if (body instanceof URLSearchParams) {
      return body.toString();
    }
    if (body instanceof ArrayBuffer) {
      return new TextDecoder().decode(body);
    }
    if (ArrayBuffer.isView(body)) {
      return new TextDecoder().decode(body);
    }
    return String(body);
  }

  if (input instanceof Request) {
    return input.clone().text();
  }

  return "";
}

async function getRequestJsonBody(input, init) {
  const bodyText = await getRequestBodyText(input, init);
  if (!bodyText) {
    return {};
  }
  return JSON.parse(bodyText);
}

function normalizeChatMessages(rawMessages) {
  if (!Array.isArray(rawMessages) || rawMessages.length === 0) {
    return null;
  }

  const normalized = [];
  for (const message of rawMessages) {
    if (!message || typeof message.role !== "string" || typeof message.content !== "string") {
      return null;
    }
    normalized.push({
      role: message.role,
      content: message.content,
    });
  }
  return normalized;
}

function createNdjsonResponse(producer) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (payload) => {
        controller.enqueue(encoder.encode(`${JSON.stringify(payload)}\n`));
      };

      try {
        await producer(send);
      } catch (error) {
        send({ error: getErrorMessage(error), done: true });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "content-type": "application/x-ndjson; charset=utf-8",
      "cache-control": "no-store",
      "x-accel-buffering": "no",
    },
  });
}

function inferenceErrorToResponse(error) {
  if (error && typeof error === "object" && "code" in error) {
    if (error.code === "busy") {
      return ollamaErrorResponse(503, "inference is busy");
    }
    if (error.code === "model_not_loaded") {
      return ollamaErrorResponse(503, "model is not loaded");
    }
  }
  return ollamaErrorResponse(500, getErrorMessage(error));
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
      updatedAt: Date.now(),
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

async function withInferenceLock(task) {
  if (state.inferenceInFlight) {
    const busyError = new Error("inference is busy");
    busyError.code = "busy";
    throw busyError;
  }

  state.inferenceInFlight = true;
  try {
    return await task();
  } finally {
    state.inferenceInFlight = false;
  }
}

async function runModelCompletion(messages, onToken) {
  if (!state.wllama || !state.model) {
    const modelError = new Error("model is not loaded");
    modelError.code = "model_not_loaded";
    throw modelError;
  }

  return state.wllama.createChatCompletion(messages, {
    nPredict: 256,
    useCache: false,
    sampling: {
      temp: 0.7,
      top_k: 40,
      top_p: 0.9,
    },
    onNewToken: (_token, piece, currentText) => {
      if (typeof onToken === "function") {
        onToken(piece || "", currentText || "");
      }
    },
  });
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
    const output = await withInferenceLock(() =>
      runModelCompletion([{ role: "user", content: prompt }], (_piece, currentText) => {
        streamedText = currentText;
        setStatus(responseEl, currentText || "推論中...");
      })
    );
    setStatus(responseEl, output || streamedText || "(空の応答)");
  } catch (error) {
    setStatus(responseEl, `推論エラー: ${getErrorMessage(error)}`, true);
  } finally {
    runBtn.disabled = false;
  }
}

async function handleTagsRequest(method) {
  if (method !== "GET" && method !== "POST") {
    return methodNotAllowedResponse(["GET", "POST"]);
  }
  return jsonResponse(getTagsResponsePayload());
}

function buildGenerateMessages(body) {
  const prompt = typeof body.prompt === "string" ? body.prompt.trim() : "";
  if (!prompt) {
    return { error: "prompt is required", messages: null };
  }

  const messages = [];
  const systemPrompt = typeof body.system === "string" ? body.system.trim() : "";
  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }
  messages.push({ role: "user", content: prompt });
  return { error: null, messages };
}

async function handleGenerateRequest(method, input, init) {
  if (method !== "POST") {
    return methodNotAllowedResponse(["POST"]);
  }

  let body;
  try {
    body = await getRequestJsonBody(input, init);
  } catch (_error) {
    return ollamaErrorResponse(400, "invalid json body");
  }

  const modelName = getModelNameFromRequest(body);
  if (!isSupportedModelName(modelName)) {
    return ollamaErrorResponse(404, `model '${modelName}' not found`);
  }

  const generatePayload = buildGenerateMessages(body);
  if (generatePayload.error) {
    return ollamaErrorResponse(400, generatePayload.error);
  }

  const stream = body.stream !== false;
  const createdAt = new Date().toISOString();

  if (!stream) {
    try {
      const text = await withInferenceLock(() => runModelCompletion(generatePayload.messages));
      return jsonResponse({
        model: modelName,
        created_at: createdAt,
        response: text || "",
        done: true,
      });
    } catch (error) {
      return inferenceErrorToResponse(error);
    }
  }

  if (!state.wllama || !state.model) {
    return ollamaErrorResponse(503, "model is not loaded");
  }
  if (state.inferenceInFlight) {
    return ollamaErrorResponse(503, "inference is busy");
  }

  return createNdjsonResponse(async (send) => {
    let sentChunk = false;
    const text = await withInferenceLock(() =>
      runModelCompletion(generatePayload.messages, (piece) => {
        if (!piece) {
          return;
        }
        sentChunk = true;
        send({
          model: modelName,
          created_at: createdAt,
          response: piece,
          done: false,
        });
      })
    );

    if (!sentChunk && text) {
      send({
        model: modelName,
        created_at: createdAt,
        response: text,
        done: false,
      });
    }

    send({
      model: modelName,
      created_at: createdAt,
      response: "",
      done: true,
    });
  });
}

async function handleChatRequest(method, input, init) {
  if (method !== "POST") {
    return methodNotAllowedResponse(["POST"]);
  }

  let body;
  try {
    body = await getRequestJsonBody(input, init);
  } catch (_error) {
    return ollamaErrorResponse(400, "invalid json body");
  }

  const modelName = getModelNameFromRequest(body);
  if (!isSupportedModelName(modelName)) {
    return ollamaErrorResponse(404, `model '${modelName}' not found`);
  }

  const messages = normalizeChatMessages(body.messages);
  if (!messages) {
    return ollamaErrorResponse(400, "messages must be a non-empty array of { role, content }");
  }

  const stream = body.stream !== false;
  const createdAt = new Date().toISOString();

  if (!stream) {
    try {
      const text = await withInferenceLock(() => runModelCompletion(messages));
      return jsonResponse({
        model: modelName,
        created_at: createdAt,
        message: {
          role: "assistant",
          content: text || "",
        },
        done: true,
      });
    } catch (error) {
      return inferenceErrorToResponse(error);
    }
  }

  if (!state.wllama || !state.model) {
    return ollamaErrorResponse(503, "model is not loaded");
  }
  if (state.inferenceInFlight) {
    return ollamaErrorResponse(503, "inference is busy");
  }

  return createNdjsonResponse(async (send) => {
    let sentChunk = false;
    const text = await withInferenceLock(() =>
      runModelCompletion(messages, (piece) => {
        if (!piece) {
          return;
        }
        sentChunk = true;
        send({
          model: modelName,
          created_at: createdAt,
          message: {
            role: "assistant",
            content: piece,
          },
          done: false,
        });
      })
    );

    if (!sentChunk && text) {
      send({
        model: modelName,
        created_at: createdAt,
        message: {
          role: "assistant",
          content: text,
        },
        done: false,
      });
    }

    send({
      model: modelName,
      created_at: createdAt,
      message: {
        role: "assistant",
        content: "",
      },
      done: true,
    });
  });
}

async function handleOllamaRequest(endpoint, input, init) {
  const method = getRequestMethod(input, init);

  if (method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        allow: "GET,POST,OPTIONS",
      },
    });
  }

  if (endpoint === "tags") {
    return handleTagsRequest(method);
  }
  if (endpoint === "generate") {
    return handleGenerateRequest(method, input, init);
  }
  if (endpoint === "chat") {
    return handleChatRequest(method, input, init);
  }

  return ollamaErrorResponse(404, "not found");
}

function installOllamaApiFetch() {
  if (state.apiFetchInstalled) {
    return;
  }

  const nativeFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const rawUrl = input instanceof Request ? input.url : String(input);
    let url;
    try {
      url = new URL(rawUrl, window.location.href);
    } catch (_error) {
      return nativeFetch(input, init);
    }

    if (url.origin !== window.location.origin) {
      return nativeFetch(input, init);
    }

    const endpoint = getOllamaEndpoint(url.pathname);
    if (!endpoint) {
      return nativeFetch(input, init);
    }

    return handleOllamaRequest(endpoint, input, init);
  };

  state.apiFetchInstalled = true;
}

function bootstrap() {
  installOllamaApiFetch();
  setStatus(gpuStatusEl, "ブラウザ内推論モードで起動しました。Ollama互換API: /api/tags /api/generate /api/chat");
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
