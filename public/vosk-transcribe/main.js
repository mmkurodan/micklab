// VOSK ローカル音声文字起こし — ブラウザ内でモデルを保持し連続認識するデモ。
// モデルは vosk-browser (WASM) が IndexedDB(IDBFS) に自動キャッシュするため、
// 2 回目以降はダウンロード不要でオフライン動作します。

const VOSK_SRC = "https://cdn.jsdelivr.net/npm/vosk-browser@0.0.8/dist/vosk.js";

const MODELS = {
  "ja-JP": {
    label: "日本語 (ja-JP)",
    url: "https://huggingface.co/rhasspy/vosk-models/resolve/main/ja/vosk-model-small-ja-0.22.zip",
    approxMB: 48,
    joiner: "", // 日本語はトークン間の空白を詰める
  },
  "en-US": {
    label: "English (en-US)",
    url: "https://huggingface.co/rhasspy/vosk-models/resolve/main/en/vosk-model-small-en-us-0.15.zip",
    approxMB: 40,
    joiner: " ",
  },
};

const DEFAULT_LANG = "ja-JP";

// この時間(ms)以上無音が続いたあとの発話は、直前に改行を入れて段落を分ける。
const SILENCE_GAP_MS = 1500;

const locale = document.documentElement.lang.toLowerCase().startsWith("en") ? "en" : "ja";
const uiText = {
  ja: {
    initModelStatus: "モデル未ロード。言語を選び「モデルを取得」を押してください。",
    loadingVosk: "認識エンジン(WASM)を読み込み中...",
    voskLoadFailed: (e) => `認識エンジンの読み込みに失敗: ${e}`,
    downloading: (label, mb) =>
      `モデルをダウンロード/展開中...（${label} / 初回のみ 約${mb}MB。以降はキャッシュ再利用）`,
    modelReady: (label) => `モデル準備完了: ${label}（オフラインでも利用できます）`,
    modelLoadFailed: (e) => `モデル読み込み失敗: ${e}`,
    loadModelFirst: "先にモデルを取得してください。",
    micDenied: "マイクを利用できません（権限が拒否されたか、非対応です）: ",
    listening: "認識中... マイクに向かって話してください。",
    stopped: "停止しました。",
    recording: "● 認識中",
    idle: "待機中",
    copied: "コピーしました。",
    copyFailed: "コピーに失敗しました。",
    nothingToCopy: "コピーする文字起こしがありません。",
    cleared: "文字起こしをクリアしました。",
    noSecure: "マイク入力には HTTPS 接続が必要です。",
  },
  en: {
    initModelStatus: 'Model not loaded. Choose a language and press "Fetch model".',
    loadingVosk: "Loading recognition engine (WASM)...",
    voskLoadFailed: (e) => `Failed to load recognition engine: ${e}`,
    downloading: (label, mb) =>
      `Downloading/extracting model... (${label} / ~${mb}MB on first run; cached afterwards)`,
    modelReady: (label) => `Model ready: ${label} (works offline too)`,
    modelLoadFailed: (e) => `Failed to load model: ${e}`,
    loadModelFirst: "Fetch the model first.",
    micDenied: "Microphone unavailable (permission denied or unsupported): ",
    listening: "Listening... please speak into the microphone.",
    stopped: "Stopped.",
    recording: "● Listening",
    idle: "Idle",
    copied: "Copied.",
    copyFailed: "Copy failed.",
    nothingToCopy: "No transcript to copy.",
    cleared: "Transcript cleared.",
    noSecure: "Microphone input requires an HTTPS connection.",
  },
}[locale];

const langSelectEl = document.getElementById("langSelect");
const loadModelBtn = document.getElementById("loadModelBtn");
const modelStatusEl = document.getElementById("modelStatus");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const recStateEl = document.getElementById("recState");
const partialEl = document.getElementById("partial");
const transcriptEl = document.getElementById("transcript");
const copyBtn = document.getElementById("copyBtn");
const clearBtn = document.getElementById("clearBtn");

const state = {
  model: null,
  modelLang: null,
  recognizer: null,
  audioContext: null,
  mediaStream: null,
  source: null,
  processor: null,
  recording: false,
};

function setStatus(el, message, isError = false) {
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function getErrorMessage(error) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function normalizeText(text, joiner) {
  const value = (text || "").trim();
  if (!value) return "";
  if (joiner === "") return value.replace(/\s+/g, "");
  return value;
}

async function ensureVoskLoaded() {
  if (window.Vosk) return window.Vosk;
  setStatus(modelStatusEl, uiText.loadingVosk);
  await new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${VOSK_SRC}"]`);
    if (existing) {
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", () => reject(new Error("network")), { once: true });
      if (window.Vosk) resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = VOSK_SRC;
    script.onload = resolve;
    script.onerror = () => reject(new Error("network"));
    document.head.appendChild(script);
  });
  if (!window.Vosk) throw new Error("Vosk global not available");
  return window.Vosk;
}

async function loadModel() {
  const langKey = langSelectEl.value in MODELS ? langSelectEl.value : DEFAULT_LANG;
  const cfg = MODELS[langKey];

  loadModelBtn.disabled = true;
  langSelectEl.disabled = true;
  startBtn.disabled = true;

  try {
    const Vosk = await ensureVoskLoaded();

    // Switch language: release any previous model.
    if (state.recording) stopRecording();
    if (state.model && typeof state.model.terminate === "function") {
      state.model.terminate();
      state.model = null;
      state.modelLang = null;
    }

    setStatus(modelStatusEl, uiText.downloading(cfg.label, cfg.approxMB));
    const model = await Vosk.createModel(cfg.url);
    state.model = model;
    state.modelLang = langKey;

    setStatus(modelStatusEl, uiText.modelReady(cfg.label));
    startBtn.disabled = false;
  } catch (error) {
    const msg = getErrorMessage(error) === "network"
      ? uiText.voskLoadFailed("network")
      : uiText.modelLoadFailed(getErrorMessage(error));
    setStatus(modelStatusEl, msg, true);
  } finally {
    loadModelBtn.disabled = false;
    langSelectEl.disabled = false;
  }
}

async function startRecording() {
  if (!state.model) {
    setStatus(modelStatusEl, uiText.loadModelFirst, true);
    return;
  }
  if (state.recording) return;

  if (!window.isSecureContext) {
    setStatus(modelStatusEl, uiText.noSecure, true);
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus(modelStatusEl, uiText.micDenied + "getUserMedia", true);
    return;
  }

  startBtn.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
      },
    });
    state.mediaStream = stream;

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioCtx();
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }
    state.audioContext = audioContext;

    const cfg = MODELS[state.modelLang];
    const recognizer = new state.model.KaldiRecognizer(audioContext.sampleRate);
    recognizer.setWords(false);

    // 無音区間の検出用。直近の発話イベント時刻を記録し、次の発話開始までの
    // 空白がしきい値を超えたら、その確定結果の手前に改行を入れる。
    let lastSpeechAt = performance.now();
    let pendingNewline = false;

    recognizer.on("result", (message) => {
      const text = normalizeText(message?.result?.text, cfg.joiner);
      if (text) {
        appendFinal(text, cfg.joiner, pendingNewline);
        pendingNewline = false;
      }
      lastSpeechAt = performance.now();
      partialEl.textContent = "";
    });
    recognizer.on("partialresult", (message) => {
      const partial = normalizeText(message?.result?.partial, cfg.joiner);
      if (partial) {
        // 直前の発話から一定時間空いていれば、新しい発話とみなし段落を分ける。
        const now = performance.now();
        if (now - lastSpeechAt > SILENCE_GAP_MS && transcriptEl.value.trim()) {
          pendingNewline = true;
        }
        lastSpeechAt = now;
      }
      partialEl.textContent = partial;
    });
    state.recognizer = recognizer;

    const source = audioContext.createMediaStreamSource(stream);
    // ScriptProcessorNode: 非推奨だが iOS Safari を含め広く動作する。
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (event) => {
      try {
        state.recognizer.acceptWaveform(event.inputBuffer);
      } catch (_error) {
        // 認識器が停止した後の残余イベントは無視する。
      }
    };
    source.connect(processor);
    // 出力バッファは書き込まないため無音。フィードバックは発生しない。
    processor.connect(audioContext.destination);
    state.source = source;
    state.processor = processor;

    state.recording = true;
    recStateEl.textContent = uiText.recording;
    recStateEl.classList.add("live");
    stopBtn.disabled = false;
    setStatus(modelStatusEl, uiText.listening);
  } catch (error) {
    setStatus(modelStatusEl, uiText.micDenied + getErrorMessage(error), true);
    stopRecording();
    startBtn.disabled = false;
  }
}

function stopRecording() {
  state.recording = false;

  if (state.processor) {
    state.processor.onaudioprocess = null;
    try { state.processor.disconnect(); } catch (_e) {}
    state.processor = null;
  }
  if (state.source) {
    try { state.source.disconnect(); } catch (_e) {}
    state.source = null;
  }
  if (state.recognizer) {
    try { state.recognizer.remove(); } catch (_e) {}
    state.recognizer = null;
  }
  if (state.audioContext) {
    try { state.audioContext.close(); } catch (_e) {}
    state.audioContext = null;
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null;
  }

  partialEl.textContent = "";
  recStateEl.textContent = uiText.idle;
  recStateEl.classList.remove("live");
  stopBtn.disabled = true;
  if (state.model) startBtn.disabled = false;
}

function appendFinal(text, joiner, newParagraph = false) {
  const current = transcriptEl.value;
  if (!current) {
    transcriptEl.value = text;
  } else if (newParagraph) {
    // 無音区間のあとの発話は改行して段落を分ける。
    transcriptEl.value = current + "\n" + text;
  } else {
    const sep = joiner === "" ? "" : joiner;
    transcriptEl.value = current + sep + text;
  }
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

async function copyTranscript() {
  const value = transcriptEl.value.trim();
  if (!value) {
    setStatus(modelStatusEl, uiText.nothingToCopy, true);
    return;
  }
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(value);
    } else {
      transcriptEl.focus();
      transcriptEl.select();
      document.execCommand("copy");
    }
    setStatus(modelStatusEl, uiText.copied);
  } catch (_error) {
    setStatus(modelStatusEl, uiText.copyFailed, true);
  }
}

function clearTranscript() {
  transcriptEl.value = "";
  partialEl.textContent = "";
  setStatus(modelStatusEl, uiText.cleared);
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js", { scope: "./" }).catch(() => {});
  });
}

function bootstrap() {
  if (!langSelectEl.value) langSelectEl.value = DEFAULT_LANG;
  setStatus(modelStatusEl, uiText.initModelStatus);
  recStateEl.textContent = uiText.idle;
  startBtn.disabled = true;
  stopBtn.disabled = true;

  loadModelBtn.addEventListener("click", loadModel);
  startBtn.addEventListener("click", startRecording);
  stopBtn.addEventListener("click", stopRecording);
  copyBtn.addEventListener("click", copyTranscript);
  clearBtn.addEventListener("click", clearTranscript);

  registerServiceWorker();
}

bootstrap();
