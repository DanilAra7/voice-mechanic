// Microphone in, speech out, and an honest clock on everything in between.
const MIC_RATE = 16000;

const els = id => document.getElementById(id);
const state = {
  ws: null, mic: null, micCtx: null, outCtx: null,
  playAt: 0, audioRate: 24000, turn: null, turns: [], speaking: false,
  // performance.now() when the driver let go of the button. The one number the server cannot
  // know: everything it reports starts from its own clock, after the network.
  askedAt: null, rtts: [], handsFree: false,
  // Bytes still sitting in this tab's own send buffer when the button came up, and the round
  // trip measured at that same moment. The 60-71 ms the panel shows is sampled every four
  // seconds on an idle socket; a turn ends on a socket that has just carried four seconds of
  // microphone audio, which is not the same connection at all.
  sendQueued: null, turnRtt: null,
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
  // What the person waited, not what the server spent. Falls back to the server's own figure
  // in hands-free, where the browser never learns when the driver stopped talking.
  const waited = pick("client_ms");
  const audio = waited.length ? waited : pick("first_audio_ms");
  els("p50").textContent = q(audio, 0.5);
  els("p95").textContent = q(audio, 0.95);
  const last = done[done.length - 1];
  // Disjoint slices of one wait, not milestones on a timeline. The server times each where it
  // happens: subtracting milestones from one another gives negative model time on any turn where
  // a filler is spoken before the tools have finished, which is the turn worth explaining.
  const net = t => (Number.isFinite(t.client_ms) && Number.isFinite(t.first_audio_ms)
    ? t.client_ms - t.first_audio_ms : null);
  const STAGES = [
    ["silence", t => t.detection_ms, "hands-free only: waiting to be sure the sentence had ended"],
    ["recognition", t => t.asr_ms, "what you said, into words"],
    ["model", t => t.model_ms, "deciding what to answer"],
    ["tools", t => t.tool_ms_to_audio, "sensors, codes, search"],
    ["speech", t => t.tts_ms, "words into sound"],
    ["turn hold", t => t.hold_ms, "waiting to be sure you had finished"],
    ["network", net, "the wire, and the audio pipeline in this tab"],
  ];
  // Across this session, so one slow turn does not read as the shape of the thing.
  const median = get => {
    const xs = done.map(get).filter(Number.isFinite).sort((a, b) => a - b);
    return xs.length ? xs[Math.floor(xs.length / 2)] : null;
  };
  const total = Math.max(last.client_ms ?? last.first_audio_ms ?? 1, 1);
  const rows = STAGES.map(([label, get, why]) => {
    const value = get(last);
    if (!Number.isFinite(value)) return "";
    // Push-to-talk pays nothing for turn detection. A row of zeroes is noise, not honesty.
    if (label === "silence" && value === 0 && !median(get)) return "";
    const mid = median(get);
    return `
    <div class="stage${label === "network" ? " net" : ""}${label === "turn hold" || label === "silence" ? " hold" : ""}" title="${why}">
      <span class="stage-name">${label}</span>
      <span class="stage-bar"><i style="width:${Math.min(100, 100 * value / total)}%"></i></span>
      <span class="stage-ms">${Math.round(value)}</span>
      <span class="stage-mid">${mid === null ? "—" : Math.round(mid)}</span>
    </div>`;
  }).join("");
  const accounted = STAGES.reduce((sum, [, get]) => {
    const v = get(last);
    return sum + (Number.isFinite(v) ? v : 0);
  }, 0);
  // The tools row counts only what ran before the driver heard anything. On a slow lookup the
  // agent says a holding line and keeps searching while it talks, so most of the tool time costs
  // the driver nothing — which is invisible unless it is said.
  const covered = Math.round((last.tool_ms ?? 0) - (last.tool_ms_to_audio ?? 0));
  els("breakdown").innerHTML = `
    <div class="stage head"><span class="stage-name">stage</span><span></span>
      <span class="stage-ms">last</span><span class="stage-mid">med</span></div>`
    + rows + `
    <div class="stage total"><span class="stage-name">total</span><span></span>
      <span class="stage-ms">${Math.round(accounted)}</span>
      <span class="stage-mid">${Math.round(median(t => t.client_ms ?? t.first_audio_ms) ?? 0)}</span>
    </div>`
    + (covered > 20 ? `<div class="aside">a further <b>${covered} ms</b> of tool time ran while
       the agent was already talking, so you never waited for it</div>` : "");
}

/** Round trip to the server on the browser's own clock, so the network has a number of its own
 *  rather than being whatever is left over. */
// One line per turn, appended server-side: what this listener actually waited for sound, with
// the network round trip beside it so a slow answer can be told from a slow connection.
function report(clientMs) {
  const sorted = [...state.rtts].sort((a, b) => a - b);
  const rtt = sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
  try {
    state.ws.send(JSON.stringify({
      type: "client_latency",
      client_ms: Math.round(clientMs),
      rtt_ms: rtt === null ? null : Math.round(rtt),
      // Everything the server cannot see about the gap between its clock and this one.
      turn_rtt_ms: state.turnRtt ? Math.round(state.turnRtt) : null,
      send_queued_bytes: state.sendQueued,
      hands_free: state.handsFree,
    }));
  } catch { /* a closed socket must never cost the listener their answer */ }
}

function ping() {
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "ping", t: performance.now() }));
}

function notePong(sent) {
  // A pong that answers the ping fired on button release prices the network under the load the
  // turn actually ran on, rather than between turns.
  if (state.turnRtt === 0) state.turnRtt = performance.now() - sent;
  state.rtts.push(performance.now() - sent);
  const sorted = [...state.rtts].sort((a, b) => a - b);
  els("rtt").textContent = `${Math.round(sorted[Math.floor(sorted.length / 2)])} ms to the server`;
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
  await garage.ready;
  const url = els("wsUrl").value;
  state.ws = new WebSocket(url);
  state.ws.binaryType = "arraybuffer";
  setStatus("connecting…");

  state.ws.onopen = () => state.ws.send(JSON.stringify({
    type: "hello", device: deviceName(), vehicle: els("gVehicle").value || null,
  }));

  state.ws.onmessage = ev => {
    if (ev.data instanceof ArrayBuffer) {
      if (state.askedAt !== null && state.turn && state.turn.client_ms === undefined) {
        state.turn.client_ms = performance.now() - state.askedAt;
        state.askedAt = null;
        // Sent back so nobody has to read a number off a panel and remember it. The panel shows
        // this session; the server keeps every session, which is the only way a figure survives
        // the tab being closed.
        report(state.turn.client_ms);
      }
      return playPcm(ev.data);
    }
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case "ready":
        state.audioRate = m.audio_rate;
        setStatus("connected — hold the button and speak", true);
        els("talk").disabled = false;
        ping();
        state.pinger ??= setInterval(ping, 4000);
        break;
      case "pong":
        notePong(m.t);
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
        // A turn that produced no sound must not leave the stopwatch running into the next one.
        state.askedAt = null;
        renderLatency();
        break;
      case "vehicle":
        log("sys", `now driving a ${vehicleTitle(m.vehicle)} — conversation started over`);
        break;
      case "error":
        log("sys", `error: ${m.message}`);
        break;
    }
  };

  state.ws.onclose = () => {
    setStatus("disconnected");
    els("talk").disabled = true;
    clearInterval(state.pinger);
    state.pinger = null;
  };
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
    // Hands-free keeps the line open and lets the server's detector decide where a turn ends;
    // push-to-talk sends only while the button is down and then says so explicitly.
    if ((!state.speaking && !state.handsFree) || state.ws?.readyState !== WebSocket.OPEN) return;
    const f = e.data, pcm = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) pcm[i] = Math.max(-1, Math.min(1, f[i])) * 32767;
    state.ws.send(pcm.buffer);
  };
  state.micCtx.createMediaStreamSource(stream).connect(node);
  state.mic = node;
}

// ---- garage --------------------------------------------------------------
// The point of this panel: whoever is testing breaks the car themselves and then asks Dex about
// it. Dex is never told which fault was picked — it only ever sees the sensor stream — so
// whether it works the fault out is visible in the conversation rather than taken on trust.
const garage = { faults: {}, vehicles: {}, running: false, timer: null, ready: null };
const SPEEDS = [["1", "real time"], ["5", "5× faster"], ["20", "20× faster"]];
// Readings worth a glance while a fault develops, in the order they are shown. The names the
// car uploads with them ("Engine Coolant Temperature") are too long for a column this narrow,
// so the panel labels its own; anything unexpected still falls back to the uploaded name.
const SHOWN_PIDS = [
  [0x05, "coolant"], [0x0c, "revs"], [0x0d, "speed"], [0x04, "load"],
  [0x06, "fuel trim, short"], [0x07, "fuel trim, long"], [0x10, "air flow"], [0x42, "volts"],
];
// A reading turns red exactly where the agent's own tool result calls it abnormal
// (see normal_range in src/mechanic/agent/tools.py) — the panel must not claim more than Dex sees.
const ALERT = {
  0x05: v => v > 110,           // coolant, °C
  0x06: v => Math.abs(v) > 15,  // short term fuel trim, %
  0x07: v => Math.abs(v) > 15,  // long term fuel trim, %
  0x42: v => v < 13,            // system voltage with the engine running
};

const deviceName = () => els("device").value.trim() || "demo";
const vehicleTitle = id => garage.vehicles[id] ?? id ?? "unknown car";

// The front end can be served from anywhere; the socket address is the one thing the person
// already has to get right, so the REST calls follow it rather than guessing at the origin.
function apiBase() {
  const u = new URL(els("wsUrl").value, location.href);
  return `${u.protocol === "wss:" ? "https:" : "http:"}//${u.host}`;
}

async function api(path, options) {
  const r = await fetch(apiBase() + path, options);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

const send = (path, method, body) => api(path, {
  method, headers: { "content-type": "application/json" }, body: body && JSON.stringify(body),
});

function fill(id, pairs, selected) {
  const select = els(id);
  select.replaceChildren(...pairs.map(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }));
  if (selected !== undefined) select.value = selected;
}

async function loadCatalog() {
  const cat = await api("/api/catalog");
  garage.faults = Object.fromEntries(cat.faults.map(f => [f.id, f]));
  garage.vehicles = Object.fromEntries(cat.vehicles.map(v => [v.id, v.title]));
  fill("gVehicle", cat.vehicles.map(v => [v.id, `${v.make} ${v.model} (${v.generation})`]));
  // City driving, not idle: several faults only show their signature once the car is moving.
  fill("gMode", cat.modes.map(m => [m, m]), "city");
  fill("gFault", [["", "healthy car"], ...cat.faults.map(f => [f.id, f.title])]);
  fill("gScale", SPEEDS, "5");
  showTruth();
  await adoptRunningCar();
}

/** The car keeps running on the server when the page is reloaded; pick it back up rather than
 *  showing a garage that claims to be empty. */
async function adoptRunningCar() {
  let status;
  try {
    status = await api(`/api/sim/${encodeURIComponent(deviceName())}`);
  } catch {
    return;  // 404: nothing running for this device, which is the ordinary case
  }
  els("gVehicle").value = status.vehicle;
  els("gMode").value = status.mode;
  els("gFault").value = status.fault ?? "";
  showTruth();
  setRunning(true, `${status.mode} · ${status.fault ?? "healthy"}`);
  await refreshSensors();
}

/** What is really wrong, for the person testing only — revealed on request, never sent to Dex. */
function showTruth() {
  const fault = garage.faults[els("gFault").value];
  els("gTruth").hidden = !fault;
  els("gCause").hidden = true;
  els("gReveal").hidden = false;
  if (fault) {
    els("gCause").replaceChildren();
    const cause = document.createElement("b");
    cause.textContent = fault.hidden_cause;
    els("gCause").append(cause, document.createElement("br"),
                         document.createTextNode(`The driver would say: ${fault.symptom_hint}`));
  }
}

function setRunning(running, note) {
  garage.running = running;
  // Talking to Dex with no car running is the worst state this page has: every answer becomes
  // "check your adapter", which reads as the agent being stupid rather than as the garage being
  // empty. It happened to the first person who tried it, so it is said out loud now.
  els("noCar").hidden = running;
  els("gState").textContent = note;
  els("gState").classList.toggle("running", running);
  els("gStop").disabled = !running;
  els("gStart").textContent = running ? "Restart the car" : "Start the car";
  if (running) {
    garage.timer ??= setInterval(refreshSensors, 1500);
  } else {
    clearInterval(garage.timer);
    garage.timer = null;
    els("gReadings").replaceChildren();
    els("gCodes").hidden = true;
  }
}

async function refreshSensors() {
  let data;
  try {
    data = await api(`/api/sensors/${encodeURIComponent(deviceName())}`);
  } catch {
    return;  // a blip in polling is not worth a line in the conversation
  }
  const by = Object.fromEntries(data.readings.map(r => [r.pid, r]));
  // Names and units arrive with the uploads, so they are treated as text, never as markup.
  els("gReadings").replaceChildren(...SHOWN_PIDS.filter(([pid]) => by[pid]).map(([pid, label]) => {
    const r = by[pid];
    const cell = document.createElement("div");
    cell.className = ALERT[pid]?.(r.value) ? "reading alert" : "reading";
    const name = document.createElement("span");
    name.textContent = label || r.name;
    const value = document.createElement("b");
    value.textContent = `${Math.abs(r.value) >= 100 ? Math.round(r.value) : r.value.toFixed(1)} ${r.unit}`;
    cell.append(name, value);
    return cell;
  }));
  els("gCodes").hidden = !data.dtcs.length;
  const label = document.createElement("span");
  label.textContent = "codes";
  els("gCodes").replaceChildren(label, ...data.dtcs.map(code => {
    const chip = document.createElement("span");
    chip.className = "code";
    chip.textContent = code;
    return chip;
  }));
}

async function startCar() {
  const vehicle = els("gVehicle").value;
  const status = await send(`/api/sim/${encodeURIComponent(deviceName())}`, "POST", {
    vehicle,
    mode: els("gMode").value,
    fault: els("gFault").value || null,
    time_scale: Number(els("gScale").value),
  });
  setRunning(true, `${status.mode} · ${status.fault ?? "healthy"}`);
  log("sys", `${vehicleTitle(vehicle)} running: ${status.mode}, ${status.fault ?? "nothing wrong with it"}`);
  await refreshSensors();
}

/** Changing mode or fault on a running car, rather than restarting it: the engine stays warm and
 *  the readings keep their history, which is exactly what breaking a car mid-drive looks like. */
async function adjustCar(patch) {
  const status = await send(`/api/sim/${encodeURIComponent(deviceName())}`, "PATCH", patch);
  setRunning(true, `${status.mode} · ${status.fault ?? "healthy"}`);
}

function wireGarage() {
  const guard = fn => async (...args) => {
    try {
      await fn(...args);
    } catch (err) {
      log("sys", `garage: ${err.message || err}`);
    }
  };
  els("gStart").onclick = guard(startCar);
  els("gStop").onclick = guard(async () => {
    await send(`/api/sim/${encodeURIComponent(deviceName())}`, "DELETE");
    setRunning(false, "no car running");
    log("sys", "car switched off");
  });
  // The device name is the car's identity in the sensor stream, and the agent is told it once,
  // at the handshake. Changing it here would quietly point the garage at a different car from
  // the one Dex is reading.
  els("device").onchange = guard(async () => {
    if (state.ws?.readyState === WebSocket.OPEN) {
      log("sys", "device changed — press Connect again so Dex reads this car");
    }
    setRunning(false, "no car running");
    await adoptRunningCar();
  });
  els("gReveal").onclick = () => { els("gCause").hidden = false; els("gReveal").hidden = true; };
  els("gFault").onchange = guard(async () => {
    showTruth();
    const fault = els("gFault").value;
    if (garage.running) await adjustCar(fault ? { fault } : { clear_fault: true });
  });
  els("gMode").onchange = guard(async () => {
    if (garage.running) await adjustCar({ mode: els("gMode").value });
  });
  // A different car is a different conversation: the simulator restarts and the agent, which
  // states the current car in its system prompt, is told to start over.
  els("gVehicle").onchange = guard(async () => {
    if (state.ws?.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "vehicle", vehicle: els("gVehicle").value }));
    }
    if (garage.running) await startCar();
  });
  garage.ready = loadCatalog().catch(err => log("sys", `garage unavailable: ${err.message || err}`));
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
  const press = () => {
    state.speaking = true;
    talk.classList.add("held");
    setStatus("listening…", true);
    // While the button is down the agent must not decide the question is over, however long the
    // driver pauses to think. Ordinary speech holds pauses of most of a second.
    if (!state.handsFree) state.ws?.send(JSON.stringify({ type: "start_of_speech" }));
  };
  const release = () => {
    if (!state.speaking) return;  // a click elsewhere on the page is not the end of a question
    state.speaking = false;
    talk.classList.remove("held");
    setStatus("thinking…", true);
    if (state.handsFree) return;
    // Letting go is the driver stating the turn is over. Telling the server saves it waiting out
    // the silence to work that out for itself, and starts the clock this browser measures.
    state.askedAt = performance.now();
    // Read before the send, so it is what `end_of_speech` had to queue behind rather than what
    // it added. A non-zero figure here means the browser was still shipping the question when
    // the driver was already waiting for the answer.
    state.sendQueued = state.ws?.bufferedAmount ?? null;
    state.ws?.send(JSON.stringify({ type: "end_of_speech" }));
    state.turnRtt = 0;  // armed; the next pong fills it in
    ping();
  };
  talk.addEventListener("mousedown", press);
  talk.addEventListener("touchstart", e => { e.preventDefault(); press(); });
  addEventListener("mouseup", release);
  addEventListener("touchend", release);
  els("handsFree").onchange = e => {
    state.handsFree = e.target.checked;
    els("talk").disabled = state.handsFree || state.ws?.readyState !== WebSocket.OPEN;
    setStatus(state.handsFree ? "hands-free — just talk" : "hold the button and speak", true);
  };
  els("reset").onclick = () => {
    state.ws?.send(JSON.stringify({ type: "reset" }));
    els("log").innerHTML = "";
    state.turns = [];
    renderLatency();
  };
  els("wsUrl").value = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/voice`;
  wireGarage();
}

wire();
