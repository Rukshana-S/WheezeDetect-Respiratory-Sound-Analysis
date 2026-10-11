/* WheezeDetect frontend v3.1
   ML runs only in the Python backend. Frontend is display-only.
   Waveform uses Web Audio API + Canvas from actual file samples.
*/
'use strict';

// ===== 1. Configuration =====
const CONFIG = {
  API_URL:     'http://127.0.0.1:5000/api/analyze',
  HEALTH_URL:  'http://127.0.0.1:5000/api/health',
  MAX_MB:      50,
  TIMEOUT_MS:  60000,
  STORAGE_KEY: 'wheezedetect_history_v1',
  DEMO_KEY:    'wheezedetect_demo_pref',
  THRESHOLD:   0.225,
  // Waveform display parameters
  WF_DOWNSAMPLE: 2000,   // max data points drawn on canvas
  WF_COLOR:      '#22d3ee',
  WF_BG:         '#0a1628',
  WF_PLAYHEAD:   '#f59e0b',
  WF_GRID:       'rgba(30,51,88,.7)',
};

const $ = id => document.getElementById(id);
const show = (el, disp = '') => {
  if (typeof el === 'string') el = $(el);
  if (!el) return;
  el.hidden = false;
  el.style.display = disp;
};
const hide = el => {
  if (typeof el === 'string') el = $(el);
  if (!el) return;
  el.hidden = true;
  el.style.display = 'none';
};
const state = {
  file:       null,
  url:        null,
  duration:   null,
  busy:       false,
  current:    null,
  // Waveform state
  wfPeaks:    null,   // Float32Array of downsampled peaks
  wfCtx:      null,   // AudioContext
  wfRaf:      null,   // requestAnimationFrame handle
  wfPlayer:   null,   // reference to <audio> element
};

// ===== 2. Helpers =====
function toast(msg, type = 'info') {
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.textContent = msg;
  $('toasts').appendChild(t);
  setTimeout(() => t.remove(), 4500);
}
const fmtSize = b => (b / 1048576).toFixed(2) + ' MB';
const fmtDur  = s => Number(s).toFixed(1) + ' s';
const fmtTime = s => {
  const m = Math.floor(s / 60), sec = (s % 60).toFixed(1);
  return m > 0 ? `${m}:${sec.padStart(4,'0')}` : sec;
};
const icons = () => window.lucide && lucide.createIcons();

// ===== 3. Backend health check =====
async function checkBackendHealth() {
  const el = $('backendStatus');
  try {
    const res = await fetch(CONFIG.HEALTH_URL, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      el.classList.add('online');
      el.querySelector('.status-text').textContent = 'Backend online';
    } else { throw new Error(); }
  } catch (_) {
    el.classList.remove('online');
    el.querySelector('.status-text').textContent = 'Backend offline';
  }
}

// ===== 4. File handling =====
function setMsg(m) { $('fileMsg').textContent = m || ''; }

function handleFile(file) {
  if (!file) return;
  setMsg('');
  if (!/\.wav$/i.test(file.name)) return setMsg('Unsupported file type. Please choose a .wav file.');
  if (file.size === 0)            return setMsg('This file is empty.');
  if (file.size > CONFIG.MAX_MB * 1048576) return setMsg(`File is too large (max ${CONFIG.MAX_MB} MB).`);
  releaseFile();
  const url = URL.createObjectURL(file);
  const probe = new Audio();
  probe.preload = 'metadata';
  probe.onloadedmetadata = () => {
    state.file     = file;
    state.url      = url;
    state.duration = probe.duration;
    state.wfPlayer = $('player');

    $('player').src = url;
    $('fName').textContent = file.name;
    $('fMeta').textContent = `${fmtSize(file.size)} · ${fmtDur(probe.duration)}`;
    show('fileBox', 'block');
    $('analyzeBtn').disabled = false;
    icons();
    toast('File ready for analysis.', 'success');
    // Kick off waveform decode immediately
    decodeAndDrawWaveform(file);
  };
  probe.onerror = () => {
    URL.revokeObjectURL(url);
    setMsg('Could not read this audio file. It may be corrupted.');
  };
  probe.src = url;
}

function releaseFile() {
  stopWaveformLoop();
  resetWaveformUI();
  if (state.url) URL.revokeObjectURL(state.url);
  const player = $('player');
  player.pause();
  player.removeAttribute('src');
  player.load();
  Object.assign(state, { file: null, url: null, duration: null, wfPeaks: null, wfPlayer: null });
  hide('fileBox');
  $('analyzeBtn').disabled = true;
  $('fileInput').value   = '';
}

// ===== 5. Waveform Engine (Web Audio API + Canvas) =====

/**
 * Decode the audio file using Web Audio API, downsample to WF_DOWNSAMPLE
 * peak-envelope values, then draw and start the playhead animation loop.
 */
async function decodeAndDrawWaveform(file) {
  // Show loading state
  hide('wfEmpty');
  show('wfLoading', 'flex');
  hide('waveformWrap');
  hide('wfDisclaimer');
  hide('waveformControls');

  try {
    // Create or reuse AudioContext
    if (!state.wfCtx || state.wfCtx.state === 'closed') {
      state.wfCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const arrayBuffer  = await file.arrayBuffer();
    const audioBuffer  = await state.wfCtx.decodeAudioData(arrayBuffer);

    // Mix down to mono
    const numChannels = audioBuffer.numberOfChannels;
    const rawLength   = audioBuffer.length;
    const mono = new Float32Array(rawLength);
    for (let c = 0; c < numChannels; c++) {
      const chData = audioBuffer.getChannelData(c);
      for (let i = 0; i < rawLength; i++) mono[i] += chData[i] / numChannels;
    }

    // Downsample: compute peak per block
    const n       = CONFIG.WF_DOWNSAMPLE;
    const block   = Math.max(1, Math.floor(rawLength / n));
    const peaks   = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const start = i * block;
      const end   = Math.min(start + block, rawLength);
      let  pk     = 0;
      for (let j = start; j < end; j++) {
        const v = Math.abs(mono[j]);
        if (v > pk) pk = v;
      }
      peaks[i] = pk;
    }

    // Normalise to [0,1]
    const maxPeak = peaks.reduce((m, v) => Math.max(m, v), 0) || 1;
    for (let i = 0; i < n; i++) peaks[i] /= maxPeak;

    state.wfPeaks = peaks;

    // Build axis labels
    buildAxisLabels(audioBuffer.duration);

    // Show canvas, hide empty and loading indicators
    hide('wfEmpty');
    hide('wfLoading');
    show('waveformWrap', 'block');
    show('wfDisclaimer', 'flex');
    show('waveformControls', 'flex');

    // Update duration display
    $('wfDuration').textContent = fmtTime(audioBuffer.duration);

    // Size canvas to container
    resizeCanvas();

    // Render static waveform
    drawWaveform(0);

    // Start playhead animation
    startWaveformLoop();

  } catch (err) {
    console.error('Waveform decode error:', err);
    hide('wfLoading');
    show('wfEmpty', 'flex');
    $('wfEmpty').querySelector('p').textContent = 'Could not decode waveform for this file.';
  }
}

function buildAxisLabels(duration) {
  const axisEl = $('wfAxisX');
  axisEl.innerHTML = '';
  // Choose a reasonable tick interval
  const ticks = Math.min(10, Math.floor(duration));
  const step  = duration / ticks;
  for (let i = 0; i <= ticks; i++) {
    const t   = i * step;
    const span = document.createElement('span');
    span.textContent = fmtTime(t) + 's';
    axisEl.appendChild(span);
  }
}

function resizeCanvas() {
  const canvas = $('waveformCanvas');
  const wrap   = $('waveformWrap');
  // Set physical pixel dimensions for sharp rendering
  const dpr  = window.devicePixelRatio || 1;
  const rect = wrap.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = 140 * dpr;
  canvas.getContext('2d').scale(dpr, dpr);
}

/**
 * Draw the waveform bars + optional playhead at normalised position [0,1].
 */
function drawWaveform(progress) {
  const canvas = $('waveformCanvas');
  if (!canvas || !state.wfPeaks) return;

  const ctx    = canvas.getContext('2d');
  const dpr    = window.devicePixelRatio || 1;
  const W      = canvas.width  / dpr;
  const H      = canvas.height / dpr;
  const peaks  = state.wfPeaks;
  const n      = peaks.length;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = CONFIG.WF_BG;
  ctx.fillRect(0, 0, W, H);

  // Horizontal center line
  ctx.strokeStyle = CONFIG.WF_GRID;
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();

  // Bars
  const barW    = W / n;
  const played  = Math.floor(progress * n);
  const halfH   = H / 2;

  for (let i = 0; i < n; i++) {
    const x    = i * barW;
    const amp  = peaks[i];
    const bh   = Math.max(1, amp * halfH * 0.9);

    // Color: played portion in cyan, unplayed in indigo/blue
    if (i < played) {
      ctx.fillStyle = CONFIG.WF_COLOR;
      ctx.globalAlpha = 0.9;
    } else {
      ctx.fillStyle = '#6366f1';
      ctx.globalAlpha = 0.55;
    }

    // Draw symmetric bars around center
    ctx.fillRect(x, halfH - bh, Math.max(barW - 0.5, 0.5), bh * 2);
  }
  ctx.globalAlpha = 1;

  // Playhead line
  if (progress > 0 && progress < 1) {
    const px = progress * W;
    ctx.strokeStyle = CONFIG.WF_PLAYHEAD;
    ctx.lineWidth   = 2;
    ctx.shadowColor = CONFIG.WF_PLAYHEAD;
    ctx.shadowBlur  = 6;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

function startWaveformLoop() {
  stopWaveformLoop();
  const player = $('player');
  if (!player) return;

  function tick() {
    const dur  = player.duration || state.duration || 1;
    const ct   = player.currentTime || 0;
    const prog = ct / dur;

    drawWaveform(prog);
    $('wfCurrentTime').textContent = fmtTime(ct);

    state.wfRaf = requestAnimationFrame(tick);
  }
  state.wfRaf = requestAnimationFrame(tick);
}

function stopWaveformLoop() {
  if (state.wfRaf) {
    cancelAnimationFrame(state.wfRaf);
    state.wfRaf = null;
  }
}

function resetWaveformUI() {
  stopWaveformLoop();
  show('wfEmpty', 'flex');
  hide('wfLoading');
  hide('waveformWrap');
  hide('wfDisclaimer');
  hide('waveformControls');
  // Reset labels
  $('wfEmpty').querySelector('p').textContent = 'Waveform will appear here once you select a WAV file.';
}

/** Seek audio when user clicks the waveform canvas */
function onWaveformClick(e) {
  const player = $('player');
  if (!player || !player.duration || !state.wfPeaks) return;
  const canvas = $('waveformCanvas');
  const rect   = canvas.getBoundingClientRect();
  const relX   = (e.clientX - rect.left) / rect.width;
  player.currentTime = relX * player.duration;
}

// ===== 6. API layer =====
async function callBackend(file) {
  const fd    = new FormData();
  fd.append('file', file);
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), CONFIG.TIMEOUT_MS);
  try {
    const res = await fetch(CONFIG.API_URL, { method: 'POST', body: fd, signal: ctrl.signal });
    if (!res.ok) {
      let errMsg = `Server returned status ${res.status}.`;
      try { const d = await res.json(); if (d && d.error) errMsg = d.error; } catch (_) {}
      throw new Error(errMsg);
    }
    const d = await res.json();
    if (typeof d.prediction !== 'string' || typeof d.score !== 'number')
      throw new Error('Unexpected response format from the backend.');
    return normalize(d, 'backend');
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('The backend took too long to respond.');
    if (e instanceof TypeError) throw new Error(
      'ML backend is not connected. Start the Flask server at ' + CONFIG.API_URL + ' or use Demo mode.'
    );
    throw e;
  } finally { clearTimeout(timer); }
}

function normalize(d, source) {
  return {
    prediction: d.prediction,
    score:      d.score,
    threshold:  typeof d.threshold === 'number' ? d.threshold : CONFIG.THRESHOLD,
    segments:   d.segments_analyzed ?? (Array.isArray(d.segment_scores) ? d.segment_scores.length : null),
    duration:   d.audio_duration ?? null,
    scores:     Array.isArray(d.segment_scores) ? d.segment_scores : null,
    model:      d.model_name || 'Unknown',
    source,
  };
}

// DEMO ONLY — clearly labelled, never a real prediction
function demoResult(duration) {
  const n      = Math.max(1, Math.floor(duration - 2) + 1);
  const scores = Array.from({ length: n }, () => +(0.1 + Math.random() * 0.25).toFixed(3));
  const top    = Math.max(...scores);
  return normalize({
    prediction:        top >= CONFIG.THRESHOLD ? 'Wheeze-like Pattern Detected' : 'No Wheeze-like Pattern Detected',
    score:             top,
    threshold:         CONFIG.THRESHOLD,
    segments_analyzed: n,
    audio_duration:    duration,
    segment_scores:    scores,
    model_name:        'DEMO (simulated)',
  }, 'demo');
}

// ===== 7. Analyze flow =====
const sleep = ms => new Promise(r => setTimeout(r, ms));
function step(p, text) {
  $('bar').style.width  = p + '%';
  $('status').textContent = text;
}

async function analyze() {
  if (state.busy || !state.file) return;
  state.busy = true;
  $('analyzeBtn').disabled    = true;
  $('analyzeLabel').textContent = 'Analyzing…';
  show('progress', 'block');
  try {
    step(15, 'Preparing audio…'); await sleep(300);
    let result;
    if ($('demoToggle').checked) {
      step(50, 'Generating simulated demo data…'); await sleep(700);
      result = demoResult(state.duration);
    } else {
      step(45, 'Sending to ML backend…');
      result = await callBackend(state.file);
    }
    step(100, 'Done.');
    result.duration = result.duration ?? state.duration;
    showResult(result);
    saveRecord(result, state.file.name);
    checkBackendHealth();
    toast(result.source === 'demo' ? 'Demo result shown (simulated).' : 'Analysis complete.', 'success');
  } catch (e) {
    $('status').textContent = '';
    toast(e.message, 'error');
  } finally {
    state.busy = false;
    $('analyzeBtn').disabled    = !state.file;
    $('analyzeLabel').textContent = 'Analyze Audio';
    setTimeout(() => { hide('progress'); $('bar').style.width = 0; }, 800);
  }
}

// ===== 8. Results & segment chart =====
function isWheeze(p) { return /^wheeze|detected/i.test(p) && !/^no/i.test(p); }

function showResult(r, when) {
  state.current = r;
  hide('resultEmpty');
  show('resultBody', 'block');
  const w = isWheeze(r.prediction);

  $('verdict').className    = 'verdict' + (w ? '' : ' ok');
  $('vLabel').textContent   = r.prediction;
  $('vText').textContent    = w
    ? 'At least one segment scored at or above the threshold. This is a model output, not a diagnosis.'
    : 'No segment reached the threshold. This is a model output, not a diagnosis.';
  $('vScore').textContent   = Number(r.score).toFixed(3);
  $('vThr').textContent     = `(Threshold: ${r.threshold})`;
  $('sDur').textContent     = r.duration != null ? fmtDur(r.duration) : 'N/A';
  $('sSeg').textContent     = r.segments ?? 'N/A';
  $('sModel').textContent   = r.model;

  const tag        = $('vSource');
  tag.textContent  = r.source === 'demo' ? 'DEMO — simulated data' : 'Backend result';
  tag.className    = 'tag' + (r.source === 'demo' ? ' demo-tag' : '');

  $('when').textContent = when
    ? 'Analyzed ' + new Date(when).toLocaleString()
    : 'Analyzed ' + new Date().toLocaleString();

  // Update verdict icon
  const vi = $('verdictIcon');
  if (vi) {
    vi.setAttribute('data-lucide', w ? 'wind' : 'check-circle-2');
  }

  drawSegmentChart(r.scores, r.threshold, r.duration);
  icons();
}

/**
 * Draw the model segment score bar chart using inline SVG.
 * Only real scores returned by the backend (or demo scores) are shown.
 * No data is invented.
 */
function drawSegmentChart(scores, thr, duration) {
  const box = $('chart');
  if (!scores || !scores.length) {
    box.innerHTML = '<p class="empty-sm">Segment scores were not returned by the backend.</p>';
    return;
  }

  const W = 580, H = 200, L = 42, B = 30, T = 10;
  const maxY = Math.max(1, ...scores, thr);
  const bw   = (W - L - 8) / scores.length;
  const yPx  = v => T + (H - T - B) * (1 - v / maxY);

  // Work out time-per-segment if we have duration
  // The backend uses 2 s window / 1 s stride → segment i starts at i seconds
  const segDur = 1; // stride
  const hasTime = duration != null;

  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Model segment score bar chart">`;

  // Grid lines + Y labels
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    const yv = yPx(v);
    s += `<line x1="${L}" x2="${W - 8}" y1="${yv}" y2="${yv}" stroke="#1e3358" stroke-width="1"/>`;
    s += `<text x="${L - 5}" y="${yv + 4}" fill="#4a6d99" font-size="10" text-anchor="end">${v.toFixed(2)}</text>`;
  });

  // Bars
  scores.forEach((v, i) => {
    const x  = L + i * bw;
    const hi = v >= thr;
    const bh = Math.max(2, H - B - yPx(v));
    const t  = hasTime ? `t=${(i * segDur).toFixed(0)}–${(i * segDur + 2).toFixed(0)}s` : `Seg ${i + 1}`;
    s += `<rect x="${x + 1}" y="${yPx(v)}" width="${Math.max(bw - 2, 0.5)}" height="${bh}" rx="3"
            fill="${hi ? '#f43f5e' : '#6366f1'}" opacity="${hi ? '1' : '.75'}">
            <title>${t}: ${v.toFixed(4)}${hi ? ' — above threshold' : ' — below threshold'}</title>
          </rect>`;
    // X label every few bars
    if (scores.length <= 25 || i % Math.ceil(scores.length / 18) === 0) {
      const lbl = hasTime ? `${i * segDur}s` : `${i + 1}`;
      s += `<text x="${x + bw / 2}" y="${H - 12}" fill="#4a6d99" font-size="9" text-anchor="middle">${lbl}</text>`;
    }
  });

  // Threshold line
  const ty = yPx(thr);
  s += `<line x1="${L}" x2="${W - 8}" y1="${ty}" y2="${ty}" stroke="#f59e0b" stroke-dasharray="5 4" stroke-width="1.5"/>`;
  s += `<text x="${W - 8}" y="${ty - 4}" fill="#f59e0b" font-size="10" text-anchor="end">thr ${thr}</text>`;

  // Axis labels
  const xAxisLabel = hasTime ? 'Time (seconds)' : 'Segment number';
  s += `<text x="${(W + L) / 2}" y="${H}" fill="#4a6d99" font-size="10" text-anchor="middle">${xAxisLabel}</text>`;
  s += '<text x="12" y="60" fill="#4a6d99" font-size="10" text-anchor="middle" transform="rotate(-90,12,80)">Score</text>';
  s += '</svg>';
  box.innerHTML = s;
}

// ===== 9. History (LocalStorage; no audio or personal data stored) =====
function loadHistory() {
  try { const d = JSON.parse(localStorage.getItem(CONFIG.STORAGE_KEY) || '[]'); return Array.isArray(d) ? d : []; }
  catch (e) { return []; }
}
function writeHistory(list) {
  try { localStorage.setItem(CONFIG.STORAGE_KEY, JSON.stringify(list)); }
  catch (e) { toast('Could not save history (storage unavailable).', 'error'); }
}
function saveRecord(r, name) {
  const list = loadHistory();
  list.push({
    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    fileName: name, date: new Date().toISOString(), prediction: r.prediction, score: r.score,
    threshold: r.threshold, duration: r.duration, segments: r.segments, model: r.model,
    scores: r.scores, source: r.source,
  });
  writeHistory(list.slice(-200));
  renderHistory();
}

function renderHistory() {
  const q     = $('search').value.trim().toLowerCase();
  const f     = $('filter').value;
  const order = $('sort').value;
  let list = loadHistory().filter(h =>
    h.fileName.toLowerCase().includes(q) &&
    (f === 'all' || (f === 'wheeze') === isWheeze(h.prediction))
  );
  list.sort((a, b) => order === 'new' ? b.date.localeCompare(a.date) : a.date.localeCompare(b.date));

  const body = $('histBody');
  body.innerHTML = '';
  if (list.length > 0) hide('histEmpty'); else show('histEmpty', 'flex');
  const total = loadHistory().length;
  $('histCount').textContent = total + (total === 1 ? ' record' : ' records');
  renderDashboard();

  list.forEach(h => {
    const tr = document.createElement('tr');
    const w  = isWheeze(h.prediction);
    [h.fileName, new Date(h.date).toLocaleString(), null,
     Number(h.score).toFixed(3), h.duration != null ? fmtDur(h.duration) : '—',
     h.segments ?? '—', h.source === 'demo' ? 'Demo' : 'Backend']
    .forEach((c, i) => {
      const td = document.createElement('td');
      if (i === 2) td.innerHTML = `<span class="pill ${w ? 'w' : 'n'}">${w ? '⚠ Wheeze' : '✓ Clear'}</span>`;
      else td.textContent = c;
      tr.appendChild(td);
    });
    const act = document.createElement('td');
    act.innerHTML =
      `<button aria-label="Open record" data-open="${h.id}"><i data-lucide="eye"></i></button>` +
      `<button aria-label="Delete record" data-del="${h.id}"><i data-lucide="trash-2"></i></button>`;
    tr.appendChild(act);
    body.appendChild(tr);
  });
  icons();
}

function onHistoryClick(e) {
  const b = e.target.closest('button'); if (!b) return;
  const list = loadHistory();
  if (b.dataset.open) {
    const h = list.find(x => x.id === b.dataset.open); if (!h) return;
    showResult({ prediction: h.prediction, score: h.score, threshold: h.threshold, duration: h.duration,
      segments: h.segments, model: h.model, scores: h.scores, source: h.source }, h.date);
    navigate('analyze');
  } else if (b.dataset.del) {
    writeHistory(list.filter(x => x.id !== b.dataset.del));
    renderHistory(); toast('Record deleted.');
  }
}

function clearHistory() {
  if (!loadHistory().length) return toast('History is already empty.');
  if (confirm('Delete all saved analysis history? This cannot be undone.')) {
    writeHistory([]); renderHistory(); toast('History cleared.', 'success');
  }
}

function exportCsv() {
  const list = loadHistory();
  if (!list.length) return toast('Nothing to export yet.', 'error');
  const esc  = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
  const rows = [['id','file_name','date','prediction','score','threshold','duration_s','segments','source']];
  list.forEach(h => rows.push([h.id, h.fileName, h.date, h.prediction, h.score, h.threshold, h.duration, h.segments, h.source]));
  const blob = new Blob([rows.map(r => r.map(esc).join(',')).join('\n')], { type: 'text/csv' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'wheezedetect_history.csv';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ===== 10. Navigation =====
const VIEWS = ['dashboard', 'analyze', 'history', 'about'];
function navigate(v) { location.hash = v; }

function showView() {
  let v = location.hash.slice(1);
  if (!VIEWS.includes(v)) v = 'dashboard';
  VIEWS.forEach(id => $(id).classList.toggle('active', id === v));
  document.querySelectorAll('nav a').forEach(l => {
    const on = l.hash === '#' + v;
    l.classList.toggle('active', on);
    if (on) l.setAttribute('aria-current', 'page'); else l.removeAttribute('aria-current');
  });
  closeSidebar();
  window.scrollTo(0, 0);
  // Re-draw waveform when switching back to analyze (canvas may need redraw)
  if (v === 'analyze' && state.wfPeaks) {
    resizeCanvas();
    const player = $('player');
    const prog   = player.duration ? player.currentTime / player.duration : 0;
    drawWaveform(prog);
  }
}

function closeSidebar() {
  $('sidebar').classList.remove('open');
  $('sidebarOverlay').classList.remove('open');
  $('menuBtn').setAttribute('aria-expanded', 'false');
}

function initNav() {
  window.addEventListener('hashchange', showView);
  document.addEventListener('click', e => {
    const b = e.target.closest('[data-nav]');
    if (b) navigate(b.dataset.nav);
  });
  $('menuBtn').addEventListener('click', () => {
    const open = $('sidebar').classList.toggle('open');
    $('sidebarOverlay').classList.toggle('open', open);
    $('menuBtn').setAttribute('aria-expanded', open);
  });
  $('sidebarOverlay').addEventListener('click', closeSidebar);
  showView();
}

// ===== 11. Dashboard =====
function renderDashboard() {
  const all = loadHistory().sort((a, b) => b.date.localeCompare(a.date));
  const w   = all.filter(h => isWheeze(h.prediction)).length;
  $('stTotal').textContent  = all.length;
  $('stWheeze').textContent = w;
  $('stNone').textContent   = all.length - w;
  $('stScore').textContent  = all.length ? Number(all[0].score).toFixed(3) : '—';
  $('dashLatest').textContent = all.length
    ? `Latest: ${all[0].fileName} — ${all[0].prediction}${all[0].source === 'demo' ? ' (demo)' : ''}`
    : 'No analyses yet.';

  const body = $('dashBody');
  body.innerHTML = '';
  if (all.length > 0) hide('dashEmpty'); else show('dashEmpty', 'block');
  all.slice(0, 5).forEach(h => {
    const tr = document.createElement('tr');
    const w  = isWheeze(h.prediction);
    [h.fileName, new Date(h.date).toLocaleString(), h.prediction, Number(h.score).toFixed(3),
     h.source === 'demo' ? 'Demo' : 'Backend']
    .forEach((c, i) => {
      const td = document.createElement('td');
      if (i === 2) td.innerHTML = `<span class="pill ${w ? 'w' : 'n'}">${w ? '⚠ Wheeze' : '✓ Clear'}</span>`;
      else td.textContent = c;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

// ===== 12. Init =====
document.addEventListener('DOMContentLoaded', () => {
  icons();
  initNav();
  renderHistory();
  checkBackendHealth();
  setInterval(checkBackendHealth, 30000);

  // Restore demo preference
  try { $('demoToggle').checked = localStorage.getItem(CONFIG.DEMO_KEY) === '1'; } catch (e) {}
  $('demoToggle').addEventListener('change', e => {
    try { localStorage.setItem(CONFIG.DEMO_KEY, e.target.checked ? '1' : '0'); } catch (_) {}
    if (e.target.checked) toast('Demo mode on — results are simulated, not real predictions.');
  });

  // Drop zone
  const drop = $('drop');
  $('browseBtn').addEventListener('click', e => { e.stopPropagation(); $('fileInput').click(); });
  drop.addEventListener('click',   () => $('fileInput').click());
  drop.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('fileInput').click(); }
  });
  $('fileInput').addEventListener('change', e => handleFile(e.target.files[0]));
  ['dragenter', 'dragover'].forEach(ev =>
    drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); })
  );
  ['dragleave', 'drop'].forEach(ev =>
    drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); })
  );
  drop.addEventListener('drop', e => handleFile(e.dataTransfer.files[0]));

  // File controls
  $('removeBtn').addEventListener('click', () => { releaseFile(); setMsg(''); });
  $('analyzeBtn').addEventListener('click', analyze);

  // Waveform canvas click-to-seek
  $('waveformCanvas').addEventListener('click', onWaveformClick);

  // Sync waveform with audio player's own seek/time events
  $('player').addEventListener('seeked', () => {
    if (state.wfPeaks) {
      const player = $('player');
      drawWaveform(player.duration ? player.currentTime / player.duration : 0);
    }
  });

  // Resize waveform on window resize
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.wfPeaks) { resizeCanvas(); }
    }, 150);
  });

  // History controls
  ['search', 'filter', 'sort'].forEach(id => $(id).addEventListener('input', renderHistory));
  $('histBody').addEventListener('click', onHistoryClick);
  $('exportBtn').addEventListener('click', exportCsv);
  $('clearBtn').addEventListener('click', clearHistory);
  $('printBtn').addEventListener('click', () =>
    state.current ? window.print() : toast('Run an analysis first.', 'error')
  );

  window.addEventListener('beforeunload', () => {
    stopWaveformLoop();
    if (state.url) URL.revokeObjectURL(state.url);
  });
});
