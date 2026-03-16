const functions = require('firebase-functions');
const express = require('express');
const app = express();
app.use(express.json());

app.post('/chat', (req, res) => {
  const { messages } = req.body || {};
  const last = Array.isArray(messages) && messages.length ? messages[messages.length - 1].content : '';
  const assistant = { role: 'assistant', content: `Echo: ${last}` };
  res.json({ choices: [{ message: assistant }], done: true });
});

app.post('/generate', (req, res) => {
  res.json({ text: 'Not implemented', done: true });
});

app.get('/tags', (req, res) => {
  res.json({ tags: [] });
});

exports.api = functions.https.onRequest((req, res) => {
  // Enable CORS for simple testing
  res.set('Access-Control-Allow-Origin', '*');
  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.status(204).send('');
    return;
  }
  app(req, res);
});
