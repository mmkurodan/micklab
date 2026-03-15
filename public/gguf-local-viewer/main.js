const gpuStatusEl = document.getElementById("gpuStatus");
const modelStatusEl = document.getElementById("modelStatus");
const modelFileEl = document.getElementById("modelFile");
const loadModelBtn = document.getElementById("loadModelBtn");
const promptInputEl = document.getElementById("promptInput");
const runBtn = document.getElementById("runBtn");
const responseEl = document.getElementById("response");

const DB_NAME = "gguf-viewer-cache";
const STORE_NAME = "models";
const DB_VERSION = 1;

const state = {
  device: null,
  pipeline: null,
  engine: null,
  model: null,
};

function setStatus(el, message, isError = false) {
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function readFileAsArrayBuffer(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("FileReader error"));
    reader.readAsArrayBuffer(file);
  });
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

function createDummyPipeline(device) {
  const module = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read_write> values: array<u32>;
      @compute @workgroup_size(1)
      fn main() {
        values[0] = values[0] + 1u;
      }
    `,
  });
  return device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

async function initWebGPU() {
  if (!window.isSecureContext && location.hostname !== "localhost") {
    throw new Error("WebGPUはHTTPSまたはlocalhostでのみ利用できます。");
  }
  if (!("gpu" in navigator)) {
    throw new Error("このブラウザはWebGPUに対応していません。");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPUアダプタの取得に失敗しました。");
  }
  state.device = await adapter.requestDevice();
  state.pipeline = createDummyPipeline(state.device);
}

async function initWasmEngine() {
  // ローカルファイルはfetchせず、静的配置された engine.wasm のみ取得する。
  const response = await fetch("./engine.wasm", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`engine.wasmの読み込みに失敗しました (${response.status})`);
  }
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  state.engine = { instance, modelBytes: null };
}

function attachModelToEngine(modelBuffer) {
  // ダミーの受け渡し: 実際の実装ではここでWASMメモリへコピーして推論エンジンに渡す。
  state.engine.modelBytes = new Uint8Array(modelBuffer);
  if (typeof state.engine.instance.exports.infer === "function") {
    state.engine.instance.exports.infer();
  }
}

async function runWebGPUDummyPass(seedValue) {
  const inOutBuffer = state.device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const readBuffer = state.device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  state.device.queue.writeBuffer(inOutBuffer, 0, new Uint32Array([seedValue]));
  const bindGroup = state.device.createBindGroup({
    layout: state.pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: inOutBuffer } }],
  });

  const encoder = state.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(inOutBuffer, 0, readBuffer, 0, 4);
  state.device.queue.submit([encoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const mapped = readBuffer.getMappedRange();
  const value = new Uint32Array(mapped)[0];
  readBuffer.unmap();
  inOutBuffer.destroy();
  readBuffer.destroy();
  return value;
}

async function loadSelectedModel() {
  const file = modelFileEl.files && modelFileEl.files[0];
  if (!file) {
    setStatus(modelStatusEl, "モデルファイルを選択してください。", true);
    return;
  }

  loadModelBtn.disabled = true;
  runBtn.disabled = true;
  setStatus(modelStatusEl, `モデルを準備中: ${file.name}`);

  const modelId = `${file.name}:${file.size}:${file.lastModified}`;
  let modelBuffer = null;
  let cacheUsable = true;

  try {
    const cached = await getCachedModel(modelId);
    if (cached && cached.bytes) {
      modelBuffer = cached.bytes;
      setStatus(modelStatusEl, `IndexedDBキャッシュからロード: ${file.name}`);
    }
  } catch (error) {
    cacheUsable = false;
    setStatus(modelStatusEl, `IndexedDBエラー: ${error.message}\nキャッシュなしで続行します。`, true);
  }

  if (!modelBuffer) {
    modelBuffer = await readFileAsArrayBuffer(file);
    if (cacheUsable) {
      await putCachedModel({
        id: modelId,
        name: file.name,
        size: file.size,
        lastModified: file.lastModified,
        updatedAt: Date.now(),
        bytes: modelBuffer,
      });
      setStatus(modelStatusEl, `FileReaderで読み込み、IndexedDBに保存: ${file.name}`);
    } else {
      setStatus(modelStatusEl, `FileReaderで読み込み（キャッシュ無効）: ${file.name}`, true);
    }
  }

  state.model = { id: modelId, name: file.name, bytes: modelBuffer };
  attachModelToEngine(modelBuffer);
  runBtn.disabled = false;
  loadModelBtn.disabled = false;
}

async function runInference() {
  const prompt = promptInputEl.value.trim();
  if (!state.model) {
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
    const gpuValue = await runWebGPUDummyPass(prompt.length);
    const preview = prompt.length > 120 ? `${prompt.slice(0, 120)}...` : prompt;
    setStatus(
      responseEl,
      [
        "[ダミー推論結果]",
        `モデル: ${state.model.name}`,
        `入力文字数: ${prompt.length}`,
        `WebGPU計算値: ${gpuValue}`,
        "WASM: engine.wasm を初期化し、モデルバイト列を渡しました。",
        `応答: 受け取ったプロンプト「${preview}」`,
      ].join("\n")
    );
  } catch (error) {
    setStatus(responseEl, `推論エラー: ${error.message}`, true);
  } finally {
    runBtn.disabled = false;
  }
}

async function bootstrap() {
  modelFileEl.addEventListener("change", () => {
    setStatus(modelStatusEl, "ファイル選択済み。読み込みボタンを押してください。");
  });
  loadModelBtn.addEventListener("click", () => {
    loadSelectedModel().catch((error) => {
      setStatus(modelStatusEl, `モデル読み込み失敗: ${error.message}`, true);
      loadModelBtn.disabled = false;
    });
  });
  runBtn.addEventListener("click", () => {
    runInference();
  });

  try {
    await initWebGPU();
    await initWasmEngine();
    setStatus(gpuStatusEl, "WebGPU利用可能: 推論を実行できます。");
    setStatus(modelStatusEl, "モデル未ロード。GGUFファイルを選択して読み込んでください。");
    loadModelBtn.disabled = false;
  } catch (error) {
    setStatus(gpuStatusEl, `初期化失敗: ${error.message}`, true);
    setStatus(modelStatusEl, "WebGPU非対応または初期化失敗のため利用できません。", true);
    loadModelBtn.disabled = true;
    runBtn.disabled = true;
  }
}

bootstrap();
