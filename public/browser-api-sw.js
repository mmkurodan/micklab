"use strict";

const API_PATH_RE = /^\/api\/(tags|generate|chat)$/;
const DEFAULT_MODEL = "default";

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting();
    return;
  }
  if (event.data && event.data.type === "CLAIM_CLIENTS") {
    event.waitUntil(self.clients.claim());
  }
});

self.addEventListener("fetch", (event) => {
  const requestUrl = new URL(event.request.url);
  if (requestUrl.origin !== self.location.origin) {
    return;
  }
  const match = requestUrl.pathname.match(API_PATH_RE);
  if (!match) {
    return;
  }
  event.respondWith(handleApiRequest(match[1], event.request));
});

function getCorsHeaders() {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type",
    vary: "origin",
  };
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
      ...getCorsHeaders(),
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...extraHeaders,
    },
  });
}

function ndjsonResponse(producer) {
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
      ...getCorsHeaders(),
      "content-type": "application/x-ndjson; charset=utf-8",
      "cache-control": "no-store",
      "x-accel-buffering": "no",
    },
  });
}

function methodNotAllowedResponse(allowedMethods) {
  return jsonResponse(
    { error: `method not allowed (allowed: ${allowedMethods.join(", ")})` },
    405,
    { allow: allowedMethods.join(",") }
  );
}

async function parseJsonBody(request) {
  const bodyText = await request.text();
  if (!bodyText) {
    return {};
  }
  return JSON.parse(bodyText);
}

function getModelName(body) {
  const model = body && typeof body.model === "string" ? body.model.trim() : "";
  return model || DEFAULT_MODEL;
}

function tokenizeText(text) {
  const chunks = text.match(/\S+\s*/g);
  if (!chunks || chunks.length === 0) {
    return text ? [text] : [];
  }
  return chunks;
}

function getTagsPayload() {
  return {
    models: [
      {
        name: DEFAULT_MODEL,
        model: DEFAULT_MODEL,
        modified_at: new Date().toISOString(),
        size: 0,
        details: {
          format: "browser",
          family: "local",
          parameter_size: "unknown",
          quantization_level: "unknown",
        },
      },
    ],
  };
}

function buildGenerateText(body) {
  const prompt = typeof body.prompt === "string" ? body.prompt.trim() : "";
  if (!prompt) {
    return { error: "prompt is required", text: "" };
  }
  const system = typeof body.system === "string" ? body.system.trim() : "";
  const text = system ? `${system}\n${prompt}` : prompt;
  return { error: null, text: `Echo: ${text}` };
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
    normalized.push({ role: message.role, content: message.content });
  }
  return normalized;
}

function buildChatText(messages) {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === "user") {
      return `Echo: ${messages[i].content}`;
    }
  }
  return `Echo: ${messages[messages.length - 1].content}`;
}

async function handleTagsRequest(request) {
  if (request.method !== "GET" && request.method !== "POST") {
    return methodNotAllowedResponse(["GET", "POST"]);
  }
  return jsonResponse(getTagsPayload());
}

async function handleGenerateRequest(request) {
  if (request.method !== "POST") {
    return methodNotAllowedResponse(["POST"]);
  }

  let body;
  try {
    body = await parseJsonBody(request.clone());
  } catch (_error) {
    return jsonResponse({ error: "invalid json body" }, 400);
  }

  const modelName = getModelName(body);
  const createdAt = new Date().toISOString();
  const stream = body.stream !== false;
  const generated = buildGenerateText(body);
  if (generated.error) {
    return jsonResponse({ error: generated.error }, 400);
  }

  if (!stream) {
    return jsonResponse({
      model: modelName,
      created_at: createdAt,
      response: generated.text,
      done: true,
    });
  }

  return ndjsonResponse(async (send) => {
    for (const piece of tokenizeText(generated.text)) {
      send({
        model: modelName,
        created_at: createdAt,
        response: piece,
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

async function handleChatRequest(request) {
  if (request.method !== "POST") {
    return methodNotAllowedResponse(["POST"]);
  }

  let body;
  try {
    body = await parseJsonBody(request.clone());
  } catch (_error) {
    return jsonResponse({ error: "invalid json body" }, 400);
  }

  const messages = normalizeChatMessages(body.messages);
  if (!messages) {
    return jsonResponse({ error: "messages must be a non-empty array of { role, content }" }, 400);
  }

  const modelName = getModelName(body);
  const createdAt = new Date().toISOString();
  const stream = body.stream !== false;
  const reply = buildChatText(messages);

  if (!stream) {
    return jsonResponse({
      model: modelName,
      created_at: createdAt,
      message: {
        role: "assistant",
        content: reply,
      },
      done: true,
    });
  }

  return ndjsonResponse(async (send) => {
    for (const piece of tokenizeText(reply)) {
      send({
        model: modelName,
        created_at: createdAt,
        message: {
          role: "assistant",
          content: piece,
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

async function handleApiRequest(endpoint, request) {
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: getCorsHeaders(),
    });
  }

  try {
    if (endpoint === "tags") {
      return await handleTagsRequest(request);
    }
    if (endpoint === "generate") {
      return await handleGenerateRequest(request);
    }
    if (endpoint === "chat") {
      return await handleChatRequest(request);
    }
    return jsonResponse({ error: "not found" }, 404);
  } catch (error) {
    return jsonResponse({ error: getErrorMessage(error) }, 500);
  }
}
