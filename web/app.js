// Microphone in, speech out, and an honest clock on everything in between.
const MIC_RATE = 16000;

const els = id => document.getElementById(id);
const state = {
  ws: null, mic: null, micCtx: null, outCtx: null,
  playAt: 0, audioRate: 24000, turn: null, turns: [], speaking: false,
};

function log(kind, text, meta = "") {
  const row = document.createElement("div");
  row.className = `msg ${kind}`;
  row.innerHTML = `<span class="who">${kind}</span><span class="text"></span>` +
                  (meta ? `<span class="meta">${meta}</span>` : "");
  row.querySelector(".text").textContent = text;
  els("log").append(row);
  els("log").scrollTop = els("log").scrollHeight;
}

function setStatus(text, live) {
  els("status").textContent = text;
  els("status").classList.toggle("live", !!live);
}

// ---- latency -------------------------------------------------------------
// Every figure is measured from one zero point: the moment the turn detector
// decided the driver had stopped talking. That is what the person actually waits.
function renderLatency() {
  const done = state.turns.filter(t => t.first_audio_ms);
  els("turnCount").textContent = state.turns.length;
  if (!done.length) return;
  const pick = key => done.map(t => t[key]).filter(Number.isFinite).sort((a, b) => a - b);
  const q = (xs, p) => xs.length ? Math.round(xs[Math.min(xs.length - 1, Math.floor(p * xs.length))]) : "—";
  const audio = pick("first_audio_ms");
  els("p50").textContent = q(audio, 0.5);
  els("p95").textContent = q(audio, 0.95);
  const last = done[done.length - 1];
  const bar = (label, value, total) => `
    <div class="stage">
      <span class="stage-name">${label}</span>
      <span class="stage-bar"><i style="width:${Math.min(100, 100 * value / total)}%"></i></span>
      <span class="stage-ms">${Math.round(value)}</span>
    </div>`;
  const total = Math.max(last.first_audio_ms, 1);
  els("breakdown").innerHTML =
    bar("recognised", last.asr_ms ?? 0, total) +
    bar("first sentence", last.first_sentence_ms ?? 0, total) +
    bar("first audio", last.first_audio_ms ?? 0, total);
}

// ---- playback ------------------------------------------------------------
function playPcm(buffer) {
  const pcm = new Int16Array(buffer);
  const frame = state.outCtx.createBuffer(1, pcm.length, state.audioRate);
  const channel = frame.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32767;
  const src = state.outCtx.createBufferSource();
  src.buffer = frame;
  src.connect(state.outCtx.destination);
  const now = state.outCtx.currentTime;
  state.playAt = Math.max(state.playAt, now + 0.02);
  src.start(state.playAt);
  state.playAt += frame.duration;
  state.sources = (state.sources || []).concat(src);
}

function stopPlayback() {
  (state.sources || []).forEach(s => { try { s.stop(); } catch {} });
  state.sources = [];
  state.playAt = 0;
}

// ---- connection ----------------------------------------------------------
async function connect() {
  const url = els("wsUrl").value;
  state.ws = new WebSocket(url);
  state.ws.binaryType = "arraybuffer";
  setStatus("connecting…");

  state.ws.onopen = () => state.ws.send(JSON.stringify({
    type: "hello", device: els("device").value, vehicle: els("vehicle").value || null,
  }));

  state.ws.onmessage = ev => {
    if (ev.data instanceof ArrayBuffer) return playPcm(ev.data);
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case "ready":
        state.audioRate = m.audio_rate;
        setStatus("connected — hold the button and speak", true);
        els("talk").disabled = false;
        break;
      case "transcript":
        state.turn = { asr_ms: m.ms };
        log("you", m.text, `${m.ms} ms`);
        break;
      case "tool":
        log("tool", `${m.name}(${JSON.stringify(m.arguments ?? {})})`, `${m.duration_ms ?? "?"} ms`);
        break;
      case "sentence":
        log("dex", m.text, `${m.ms} ms`);
        if (state.turn && !state.turn.first_sentence_ms) state.turn.first_sentence_ms = m.ms;
        break;
      case "audio_start":
        if (state.turn) state.turn.first_audio_ms = m.ms;
        break;
      case "flush":
        stopPlayback();
        log("sys", "interrupted");
        break;
      case "turn_end":
        state.turns.push({ ...state.turn, ...m });
        renderLatency();
        break;
      case "error":
        log("sys", `error: ${m.message}`);
        break;
    }
  };

  state.ws.onclose = () => { setStatus("disconnected"); els("talk").disabled = true; };
  state.ws.onerror = () => setStatus("connection failed");
}

// ---- microphone ----------------------------------------------------------
// The browser only hands over a microphone on HTTPS or localhost, and the person can still
// refuse. Both are ordinary situations and have to be explained, not thrown.
async function startMic() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new Error("A microphone needs HTTPS (or localhost). Open this page over https.");
  }
  state.micCtx = new AudioContext({ sampleRate: MIC_RATE });
  state.outCtx = new AudioContext({ sampleRate: state.audioRate });
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const worklet = `
    class Tap extends AudioWorkletProcessor {
      process(inputs) {
        const ch = inputs[0][0];
        if (ch) this.port.postMessage(new Float32Array(ch));
        return true;
      }
    }
    registerProcessor("tap", Tap);`;
  await state.micCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([worklet], { type: "text/javascript" })));
  const node = new AudioWorkletNode(state.micCtx, "tap");
  node.port.onmessage = e => {
    if (!state.speaking || state.ws?.readyState !== WebSocket.OPEN) return;
    const f = e.data, pcm = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) pcm[i] = Math.max(-1, Math.min(1, f[i])) * 32767;
    state.ws.send(pcm.buffer);
  };
  state.micCtx.createMediaStreamSource(stream).connect(node);
  state.mic = node;
}

function wire() {
  els("connect").onclick = async () => {
    try {
      await connect();
      if (!state.mic) await startMic();
    } catch (err) {
      const denied = err?.name === "NotAllowedError";
      const message = denied
        ? "Microphone access was refused. Allow it in the address bar, then press Connect again."
        : err.message || String(err);
      setStatus("microphone unavailable");
      log("sys", message);
      els("talk").disabled = true;
    }
  };
  const talk = els("talk");
  const press = () => { state.speaking = true; talk.classList.add("held"); setStatus("listening…", true); };
  const release = () => { state.speaking = false; talk.classList.remove("held"); setStatus("thinking…", true); };
  talk.addEventListener("mousedown", press);
  talk.addEventListener("touchstart", e => { e.preventDefault(); press(); });
  addEventListener("mouseup", release);
  addEventListener("touchend", release);
  els("reset").onclick = () => {
    state.ws?.send(JSON.stringify({ type: "reset" }));
    els("log").innerHTML = "";
    state.turns = [];
    renderLatency();
  };
  els("wsUrl").value = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/voice`;
}

wire();
