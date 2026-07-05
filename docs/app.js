/* ==================================================================== *
 * Demo rozpoznawania emocji — logika przeglądarkowa
 *
 * Inferencja w całości po stronie klienta:
 *   - ONNX Runtime Web (WASM) uruchamia modele z docs/models/*.onnx,
 *   - MediaPipe (FaceDetector / FaceLandmarker) wykrywa twarz / punkty.
 *
 * Preprocessing odwzorowuje dokładnie realtime.py (EmotionPredictor) oraz
 * utils/landmarks.py, aby wyniki zgadzały się z wersją PyTorch.
 * ==================================================================== */

import {
  FilesetResolver, FaceDetector, FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const ort = window.ort;

// ---- ONNX Runtime Web: WASM z CDN, jeden wątek (GitHub Pages nie ma nagłówków COOP/COEP) ----
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

// ---- Zasoby MediaPipe (CDN Google) ----
const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const MP_FACE_DETECTOR =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const MP_FACE_LANDMARKER =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// ---- Kolory emocji (RGB, spójne z realtime.py EMOTION_COLORS) ----
const EMOTION_COLORS = {
  angry: "#dc3c3c", disgust: "#3ca03c", fear: "#a03ca0", happy: "#f0c828",
  sad: "#2878c8", surprise: "#dcdc28", neutral: "#b4b4b4",
};

// ---- Indeksy siatki twarzy MediaPipe (port utils/landmarks.py IDX) ----
const IDX = {
  left_eye_outer: 33, left_eye_inner: 133, right_eye_outer: 263, right_eye_inner: 362,
  left_eye_top: 159, left_eye_bottom: 145, right_eye_top: 386, right_eye_bottom: 374,
  left_brow_inner: 55, left_brow_outer: 105, right_brow_inner: 285, right_brow_outer: 334,
  mouth_left: 61, mouth_right: 291, mouth_top: 13, mouth_bottom: 14,
  upper_lip_top: 0, lower_lip_bottom: 17, nose_tip: 1, nose_bridge: 168,
  chin: 152, left_cheek: 50, right_cheek: 280,
};

const FRAME_INTERVAL_MS = 140;  // throttling pętli kamery

// --------------------------------------------------------------------- //
// Stan
// --------------------------------------------------------------------- //
const state = {
  manifest: null,
  classes: [], classesPl: [],
  current: null,                 // wpis modelu z manifestu
  sessions: new Map(),           // id -> InferenceSession (cache)
  fileset: null,
  faceDetector: null,
  faceLandmarker: null,
  mpDelegate: "GPU",
  stream: null,
  facing: "user",
  source: "camera",              // "camera" | "upload"
  running: false,
  paused: false,                 // "Zatrzymaj" — zamrożona klatka do podglądu wyników
  emaAlpha: 0.6,                 // waga nowej klatki w wygładzaniu (1 = brak wygładzania)
  emaProbs: null,
  lastFrame: null,               // { srcW, srcH, box, probs } — do przerysowania przy resize/fullscreen
  videoTs: 0,
  busy: false,
};

// ---- Skróty do elementów ----
const $ = (id) => document.getElementById(id);
const el = {
  select: $("model-select"), meta: $("model-meta"),
  progress: $("load-progress"), progressBar: $("load-progress-bar"), progressText: $("load-progress-text"),
  stage: $("stage"), video: $("video"), photo: $("photo"), overlay: $("overlay"), hint: $("stage-hint"),
  tabCamera: $("tab-camera"), tabUpload: $("tab-upload"),
  cameraControls: $("camera-controls"), uploadControls: $("upload-controls"),
  btnFlip: $("btn-flip"), btnFullscreen: $("btn-fullscreen"), btnCamToggle: $("btn-camera-toggle"),
  btnFsFlip: $("btn-fs-flip"), btnFsExit: $("btn-fs-exit"), fileInput: $("file-input"),
  smoothRange: $("smooth-range"), smoothVal: $("smooth-val"),
  dominant: $("dominant-label"), dominantPct: $("dominant-pct"),
  bars: $("bars"), inferTime: $("infer-time"),
  excludedList: $("excluded-list"),
};

// --------------------------------------------------------------------- //
// Inicjalizacja
// --------------------------------------------------------------------- //
async function init() {
  try {
    const resp = await fetch("models/models.json", { cache: "no-cache" });
    state.manifest = await resp.json();
  } catch (e) {
    setHint("Nie udało się wczytać listy modeli (models.json).");
    return;
  }
  state.classes = state.manifest.classes;
  state.classesPl = state.manifest.classesPl;

  buildModelSelect();
  buildBars();
  buildExcludedNote();
  bindControls();

  // pierwszy model + kamera równolegle
  await Promise.all([selectModel(state.manifest.models[0].id), startCamera()]);
  startLoop();
}

function buildModelSelect() {
  el.select.innerHTML = "";
  for (const m of state.manifest.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    const mb = (m.sizeBytes / 1e6).toFixed(m.sizeBytes < 1e6 ? 2 : 1);
    opt.textContent = `${m.label} — ${mb} MB`;
    el.select.appendChild(opt);
  }
  el.select.addEventListener("change", () => selectModel(el.select.value));
}

function buildBars() {
  el.bars.innerHTML = "";
  state.classes.forEach((cls, i) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.dataset.idx = i;
    row.innerHTML =
      `<span class="bar-name">${state.classesPl[i]}</span>` +
      `<span class="bar-track"><span class="bar-fill" style="background:${EMOTION_COLORS[cls]}"></span></span>` +
      `<span class="bar-pct">0%</span>`;
    el.bars.appendChild(row);
  });
}

function buildExcludedNote() {
  el.excludedList.innerHTML = "";
  for (const ex of state.manifest.excluded || []) {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${ex.label}</strong> (~${ex.sizeMB} MB) — ${ex.reason}.`;
    el.excludedList.appendChild(li);
  }
}

// --------------------------------------------------------------------- //
// Ładowanie modelu (lazy, z paskiem postępu)
// --------------------------------------------------------------------- //
async function selectModel(id) {
  const spec = state.manifest.models.find((m) => m.id === id);
  if (!spec) return;
  state.current = spec;
  state.emaProbs = null;
  updateModelMeta(spec);

  if (!state.sessions.has(id)) {
    try {
      showProgress(0, spec.sizeBytes, "Pobieranie modelu…");
      const bytes = await fetchWithProgress(`models/${spec.file}`, (r, t) =>
        showProgress(r, t || spec.sizeBytes, "Pobieranie modelu…"));
      showProgress(spec.sizeBytes, spec.sizeBytes, "Inicjalizacja…");
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      state.sessions.set(id, session);
    } catch (e) {
      hideProgress();
      setHint("Błąd ładowania modelu: " + e.message);
      return;
    }
    hideProgress();
  }

  // upewnij się, że właściwy detektor MediaPipe jest gotowy
  await ensureDetectors(spec.type === "landmarks");

  // przelicz aktualny (statyczny/zamrożony) obraz po zmianie modelu
  if (state.source === "upload" && !el.photo.hidden && el.photo.src) analyzeOnce();
  else if (state.source === "camera" && state.paused && el.video.videoWidth) analyzeOnce();
}

function updateModelMeta(spec) {
  const acc = spec.accuracy != null ? `${(spec.accuracy * 100).toFixed(1)}%` : "—";
  const f1 = spec.f1_macro != null ? spec.f1_macro.toFixed(3) : "—";
  const kind = spec.type === "landmarks" ? "cechy geometryczne (MediaPipe)" :
    `${spec.channels === 1 ? "skala szarości" : "RGB"} ${spec.inputSize}×${spec.inputSize}`;
  el.meta.innerHTML =
    `Dokładność (FER2013): <strong>${acc}</strong> · macro-F1: <strong>${f1}</strong> · wejście: ${kind}`;
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url, { cache: "force-cache" });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const total = +resp.headers.get("Content-Length") || 0;
  if (!resp.body) return new Uint8Array(await resp.arrayBuffer());
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress && onProgress(received, total);
  }
  const out = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) { out.set(c, pos); pos += c.length; }
  return out;
}

function showProgress(received, total, label) {
  el.progress.hidden = false;
  const pct = total ? Math.min(100, Math.round((received / total) * 100)) : 0;
  el.progressBar.style.width = pct + "%";
  const mb = (received / 1e6).toFixed(1);
  const totMb = (total / 1e6).toFixed(1);
  el.progressText.textContent = total ? `${label} ${mb} / ${totMb} MB (${pct}%)` : `${label} ${mb} MB`;
}
function hideProgress() { el.progress.hidden = true; }

// --------------------------------------------------------------------- //
// MediaPipe
// --------------------------------------------------------------------- //
async function ensureDetectors(needLandmarker) {
  if (!state.fileset) state.fileset = await FilesetResolver.forVisionTasks(MP_WASM);
  try {
    if (needLandmarker && !state.faceLandmarker) {
      state.faceLandmarker = await FaceLandmarker.createFromOptions(state.fileset, {
        baseOptions: { modelAssetPath: MP_FACE_LANDMARKER, delegate: state.mpDelegate },
        runningMode: "VIDEO", numFaces: 1,
      });
    } else if (!needLandmarker && !state.faceDetector) {
      state.faceDetector = await FaceDetector.createFromOptions(state.fileset, {
        baseOptions: { modelAssetPath: MP_FACE_DETECTOR, delegate: state.mpDelegate },
        runningMode: "VIDEO",
      });
    }
  } catch (e) {
    // fallback na CPU, jeśli GPU niedostępne
    if (state.mpDelegate !== "CPU") {
      state.mpDelegate = "CPU";
      return ensureDetectors(needLandmarker);
    }
    throw e;
  }
}

// --------------------------------------------------------------------- //
// Kamera
// --------------------------------------------------------------------- //
async function startCamera() {
  stopStream();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: state.facing, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    el.video.srcObject = state.stream;
    await el.video.play().catch(() => {});
    state.paused = false;
    el.btnCamToggle.textContent = "⏸ Zatrzymaj";
    updateMirror();
    setHint("");
  } catch (e) {
    setHint("Brak dostępu do kamery. Zezwól na kamerę lub użyj trybu „Zdjęcie”.");
  }
}

function stopStream() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
}

// --------------------------------------------------------------------- //
// Pętla analizy
// --------------------------------------------------------------------- //
function startLoop() {
  state.running = true;
  let last = 0;
  const tick = async (ts) => {
    if (state.running && !state.paused && state.source === "camera" && ts - last >= FRAME_INTERVAL_MS) {
      last = ts;
      await analyze(el.video, el.video.videoWidth, el.video.videoHeight, true);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function analyzeOnce() {
  const src = state.source === "upload" ? el.photo : el.video;
  const w = state.source === "upload" ? src.naturalWidth : src.videoWidth;
  const h = state.source === "upload" ? src.naturalHeight : src.videoHeight;
  state.emaProbs = null;
  await analyze(src, w, h, false);
}

async function analyze(source, w, h, smooth) {
  if (state.busy || !state.current || !w || !h) return;
  const session = state.sessions.get(state.current.id);
  if (!session) return;
  // odpowiedni detektor MediaPipe może jeszcze się inicjalizować — pomiń klatkę
  const needLandmarker = state.current.type === "landmarks";
  if (needLandmarker ? !state.faceLandmarker : !state.faceDetector) return;
  state.busy = true;
  const t0 = performance.now();
  try {
    let probs, box = null;
    if (state.current.type === "landmarks") {
      ({ probs, box } = await runLandmarks(source, w, h, session));
    } else {
      ({ probs, box } = await runImage(source, w, h, session));
    }
    if (probs) {
      if (smooth) probs = applyEma(probs);
      renderResults(probs);
    }
    state.lastFrame = { srcW: w, srcH: h, box, probs };
    drawOverlay();
  } catch (e) {
    // pojedynczy błąd klatki nie przerywa pętli
    console.error(e);
  } finally {
    el.inferTime.textContent = `czas: ${(performance.now() - t0).toFixed(0)} ms`;
    state.busy = false;
  }
}

// --------------------------------------------------------------------- //
// Ścieżka obrazowa (CNN / ViT-Lite / transfer)
// --------------------------------------------------------------------- //
const cropCanvas = document.createElement("canvas");
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

async function runImage(source, w, h, session) {
  state.videoTs = Math.max(state.videoTs + 1, Math.floor(performance.now()));
  const det = state.faceDetector.detectForVideo(source, state.videoTs);
  const raw = rawFaceBox(det, w, h);                                  // ciasny box do wyświetlenia
  const crop = cropFaceSquare(raw.x, raw.y, raw.w, raw.h, w, h);      // kwadrat z marginesem do modelu

  const spec = state.current;
  const S = spec.inputSize;
  cropCanvas.width = S; cropCanvas.height = S;
  cropCtx.drawImage(source, crop.x, crop.y, crop.w, crop.h, 0, 0, S, S);
  const rgba = cropCtx.getImageData(0, 0, S, S).data;

  const tensor = imageToTensor(rgba, S, spec);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const logits = out[session.outputNames[0]].data;
  return { probs: softmax(logits), box: raw };
}

function imageToTensor(rgba, S, spec) {
  const n = S * S;
  const gray = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255; // luma jak cv2.COLOR_BGR2GRAY, skala 0..1
  }
  const C = spec.channels;
  const data = new Float32Array(C * n);
  for (let c = 0; c < C; c++) {
    const mean = spec.mean[c], std = spec.std[c];
    const off = c * n;
    for (let i = 0; i < n; i++) data[off + i] = (gray[i] - mean) / std; // grayscale replikowane na kanały
  }
  return new ort.Tensor("float32", data, [1, C, S, S]);
}

// --------------------------------------------------------------------- //
// Ścieżka landmarkowa (SVM / XGBoost)
// --------------------------------------------------------------------- //
async function runLandmarks(source, w, h, session) {
  state.videoTs = Math.max(state.videoTs + 1, Math.floor(performance.now()));
  const res = state.faceLandmarker.detectForVideo(source, state.videoTs);
  if (!res.faceLandmarks || !res.faceLandmarks.length) return { probs: null, box: null };

  const lm = res.faceLandmarks[0];
  // współrzędne w pikselach (izotropia — jak kwadratowy crop w wersji PyTorch)
  const px = new Float32Array(lm.length), py = new Float32Array(lm.length);
  for (let i = 0; i < lm.length; i++) { px[i] = lm[i].x * w; py[i] = lm[i].y * h; }

  let feats = computeGeometricFeatures(px, py);
  // standaryzacja (StandardScaler przeniesiony z potoku sklearn do JS)
  const m = state.current.scalerMean, s = state.current.scalerScale;
  const scaled = new Float32Array(feats.length);
  for (let i = 0; i < feats.length; i++) scaled[i] = (feats[i] - m[i]) / s[i];

  const tensor = new ort.Tensor("float32", scaled, [1, feats.length]);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const probaName = session.outputNames[state.current.probaOutputIndex] || session.outputNames[1];
  const probs = Array.from(out[probaName].data); // to już prawdopodobieństwa (bez softmax)

  const box = landmarkBox(px, py, w, h);
  return { probs, box };
}

// Port utils/landmarks.compute_geometric_features (kolejność == FEATURE_NAMES)
function computeGeometricFeatures(px, py) {
  const dist = (a, b) => Math.hypot(px[a] - px[b], py[a] - py[b]);
  const eyeDist = Math.max(dist(IDX.left_eye_outer, IDX.right_eye_outer), 1e-6);
  const nd = (a, b) => dist(a, b) / eyeDist;

  const mouthOpen = nd(IDX.mouth_top, IDX.mouth_bottom);
  const mouthWidth = nd(IDX.mouth_left, IDX.mouth_right);
  const mouthCenterY = (py[IDX.mouth_top] + py[IDX.mouth_bottom]) / 2;
  const cornerY = (py[IDX.mouth_left] + py[IDX.mouth_right]) / 2;
  const mouthCornerLift = (mouthCenterY - cornerY) / eyeDist;

  const leftEyeOpen = nd(IDX.left_eye_top, IDX.left_eye_bottom);
  const rightEyeOpen = nd(IDX.right_eye_top, IDX.right_eye_bottom);
  const eyeOpenMean = (leftEyeOpen + rightEyeOpen) / 2;

  const leftBrowEye = nd(IDX.left_brow_inner, IDX.left_eye_top);
  const rightBrowEye = nd(IDX.right_brow_inner, IDX.right_eye_top);
  const browEyeMean = (leftBrowEye + rightBrowEye) / 2;
  const browInnerDist = nd(IDX.left_brow_inner, IDX.right_brow_inner);

  const mouthToNose = nd(IDX.mouth_top, IDX.nose_tip);
  const noseWrinkle = nd(IDX.nose_tip, IDX.nose_bridge);
  const lipPress = nd(IDX.upper_lip_top, IDX.lower_lip_bottom);
  const jawDrop = nd(IDX.nose_tip, IDX.chin);
  const mouthAspectRatio = mouthOpen / Math.max(mouthWidth, 1e-6);

  const leftBrowRaise = (py[IDX.left_eye_top] - py[IDX.left_brow_inner]) / eyeDist;
  const rightBrowRaise = (py[IDX.right_eye_top] - py[IDX.right_brow_inner]) / eyeDist;
  const smileAsymmetry = Math.abs((py[IDX.mouth_left] - py[IDX.mouth_right]) / eyeDist);
  const cheekRaise = nd(IDX.left_cheek, IDX.left_eye_bottom);
  const mouthCurvature =
    ((py[IDX.mouth_left] + py[IDX.mouth_right]) / 2 - py[IDX.mouth_top]) / eyeDist;

  return [
    mouthOpen, mouthWidth, mouthCornerLift,
    leftEyeOpen, rightEyeOpen, eyeOpenMean,
    leftBrowEye, rightBrowEye, browEyeMean,
    browInnerDist, mouthToNose, noseWrinkle,
    lipPress, jawDrop, mouthAspectRatio,
    leftBrowRaise, rightBrowRaise, smileAsymmetry,
    cheekRaise, mouthCurvature,
  ];
}

// --------------------------------------------------------------------- //
// Geometria twarzy / nakładka
// --------------------------------------------------------------------- //
function rawFaceBox(det, w, h) {
  if (det && det.detections && det.detections.length) {
    const bb = det.detections[0].boundingBox; // ciasny box detekcji (px w układzie źródła)
    return { x: bb.originX, y: bb.originY, w: bb.width, h: bb.height };
  }
  // brak twarzy -> centralny kwadrat kadru (do wycinka modelu)
  const side = Math.min(w, h) * 0.7;
  return { x: (w - side) / 2, y: (h - side) / 2, w: side, h: side };
}

// Port realtime.crop_face_square (margines 0.25, przycięcie do kadru)
function cropFaceSquare(x, y, w, h, W, H, margin = 0.25) {
  const cx = x + w / 2, cy = y + h / 2;
  const half = (Math.max(w, h) * (1 + margin)) / 2;
  const x0 = Math.max(0, cx - half), y0 = Math.max(0, cy - half);
  const x1 = Math.min(W, cx + half), y1 = Math.min(H, cy + half);
  return { x: x0, y: y0, w: Math.max(1, x1 - x0), h: Math.max(1, y1 - y0) };
}

function landmarkBox(px, py, w, h) {
  let minX = w, minY = h, maxX = 0, maxY = 0;
  for (let i = 0; i < px.length; i++) {
    if (px[i] < minX) minX = px[i]; if (px[i] > maxX) maxX = px[i];
    if (py[i] < minY) minY = py[i]; if (py[i] > maxY) maxY = py[i];
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

const overlayCtx = el.overlay.getContext("2d");

// Dopasowuje bufor canvasa do rzeczywistego rozmiaru kontenera (z uwzględnieniem DPR).
function sizeOverlay() {
  const rect = el.stage.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = Math.max(1, Math.round(rect.width));
  const ch = Math.max(1, Math.round(rect.height));
  const bw = Math.round(cw * dpr), bh = Math.round(ch * dpr);
  if (el.overlay.width !== bw || el.overlay.height !== bh) {
    el.overlay.width = bw; el.overlay.height = bh;
  }
  return { cw, ch, dpr };
}

// Mapowanie współrzędnych źródła (klatka/zdjęcie) na wyświetlanie przy object-fit: cover.
function coverMap(srcW, srcH, cw, ch) {
  const scale = Math.max(cw / srcW, ch / srcH);
  return { scale, offX: (cw - srcW * scale) / 2, offY: (ch - srcH * scale) / 2 };
}

// Rysuje nakładkę z ostatniej analizy (state.lastFrame). Wołana po każdej klatce oraz
// przy zmianie rozmiaru kontenera / pełnym ekranie — dzięki temu pozycje się dopasowują.
function drawOverlay() {
  const ctx = overlayCtx;
  const { cw, ch, dpr } = sizeOverlay();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);   // rysujemy w pikselach CSS
  ctx.clearRect(0, 0, cw, ch);

  const f = state.lastFrame;
  if (!f) return;
  const { srcW, srcH, box, probs } = f;
  if (!box || !probs || !srcW || !srcH) {
    if (state.current && state.current.type === "landmarks" && !box) drawNoFace(ctx, cw, ch);
    return;
  }

  const u = Math.min(cw, ch) / 480;
  const top = argmax(probs);
  const color = EMOTION_COLORS[state.classes[top]] || "#6c8cff";

  // ramka twarzy: współrzędne źródła -> ekran (cover), z lustrem dla przedniej kamery
  const { scale, offX, offY } = coverMap(srcW, srcH, cw, ch);
  const mir = state.source === "camera" && state.facing === "user";
  const xContent = mir ? (srcW - box.x - box.w) : box.x;
  const dx = offX + xContent * scale, dy = offY + box.y * scale;
  const dw = box.w * scale, dh = box.h * scale;

  ctx.lineWidth = Math.max(2, 3 * u);
  ctx.strokeStyle = color;
  ctx.strokeRect(dx, dy, dw, dh);

  // etykieta nad ramką: "<emocja> NN%" (przycięta do kadru)
  const labelTxt = `${state.classesPl[top]} ${Math.round(probs[top] * 100)}%`;
  const lf = Math.round(20 * u);
  ctx.font = `700 ${lf}px system-ui, sans-serif`;
  const padX = 8 * u, padY = 5 * u;
  const lw = ctx.measureText(labelTxt).width + 2 * padX, lh = lf + 2 * padY;
  const lx = Math.max(0, Math.min(dx, cw - lw));
  const ly = dy - lh >= 0 ? dy - lh : dy;
  ctx.fillStyle = color;
  roundRect(ctx, lx, ly, lw, lh, 6 * u); ctx.fill();
  ctx.fillStyle = contrastColor(color);
  ctx.textBaseline = "middle"; ctx.textAlign = "left";
  ctx.fillText(labelTxt, lx + padX, ly + lh / 2);

  drawBarsPanel(ctx, cw, u, probs, top);
}

// Panel słupków wszystkich klas (prawy górny róg) — odpowiednik draw_overlay z realtime.py
function drawBarsPanel(ctx, w, u, probs, top) {
  const font = Math.round(13 * u);
  const rowH = 20 * u, pad = 8 * u;
  const labelW = 84 * u, pctW = 40 * u, barW = Math.min(w * 0.24, 150 * u);
  const panelW = labelW + barW + pctW + pad * 2;
  const panelH = probs.length * rowH + pad * 2;
  const x0 = Math.max(pad, w - panelW - pad), y0 = pad;

  ctx.fillStyle = "rgba(10,12,22,0.55)";
  roundRect(ctx, x0, y0, panelW, panelH, 10 * u); ctx.fill();

  ctx.textBaseline = "middle";
  for (let i = 0; i < probs.length; i++) {
    const cy = y0 + pad + i * rowH + rowH / 2;
    const isTop = i === top;
    ctx.font = `${isTop ? "700 " : ""}${font}px system-ui, sans-serif`;
    ctx.fillStyle = isTop ? "#ffffff" : "rgba(230,233,245,0.82)";
    ctx.textAlign = "right";
    ctx.fillText(state.classesPl[i], x0 + pad + labelW - 6 * u, cy);

    const trackX = x0 + pad + labelW, trackY = cy - 6 * u, trackH = 12 * u;
    ctx.fillStyle = "rgba(255,255,255,0.14)";
    roundRect(ctx, trackX, trackY, barW, trackH, trackH / 2); ctx.fill();
    const fw = Math.max(0, barW * probs[i]);
    if (fw > 0) {
      ctx.fillStyle = EMOTION_COLORS[state.classes[i]];
      roundRect(ctx, trackX, trackY, fw, trackH, trackH / 2); ctx.fill();
    }
    ctx.textAlign = "left";
    ctx.fillStyle = isTop ? "#ffffff" : "rgba(230,233,245,0.82)";
    ctx.fillText(`${Math.round(probs[i] * 100)}%`, trackX + barW + 6 * u, cy);
  }
}

function drawNoFace(ctx, w, h) {
  const u = Math.min(w, h) / 480;
  ctx.font = `600 ${Math.round(18 * u)}px system-ui, sans-serif`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const t = "nie wykryto twarzy";
  const tw = ctx.measureText(t).width;
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  roundRect(ctx, w / 2 - tw / 2 - 12 * u, h / 2 - 18 * u, tw + 24 * u, 36 * u, 8 * u); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.fillText(t, w / 2, h / 2);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); return; }
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function contrastColor(hex) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.62 ? "#0b0d16" : "#ffffff";
}

// --------------------------------------------------------------------- //
// Wyniki
// --------------------------------------------------------------------- //
function renderResults(probs) {
  const top = argmax(probs);
  el.dominant.textContent = state.classesPl[top];
  el.dominant.style.color = EMOTION_COLORS[state.classes[top]];
  el.dominantPct.textContent = `${(probs[top] * 100).toFixed(0)}%`;
  const rows = el.bars.children;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    row.classList.toggle("top", i === top);
    row.querySelector(".bar-fill").style.width = `${(probs[i] * 100).toFixed(1)}%`;
    row.querySelector(".bar-pct").textContent = `${(probs[i] * 100).toFixed(0)}%`;
  }
}

function applyEma(probs) {
  if (!state.emaProbs) { state.emaProbs = probs.slice(); return probs; }
  const a = state.emaAlpha;
  const out = new Array(probs.length);
  let sum = 0;
  for (let i = 0; i < probs.length; i++) {
    out[i] = a * probs[i] + (1 - a) * state.emaProbs[i];
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  state.emaProbs = out;
  return out;
}

// --------------------------------------------------------------------- //
// Narzędzia
// --------------------------------------------------------------------- //
function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const exps = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { exps[i] = Math.exp(logits[i] - max); sum += exps[i]; }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum;
  return exps;
}

function argmax(arr) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > bv) { bv = arr[i]; bi = i; }
  return bi;
}

function setHint(text) { el.hint.textContent = text; el.hint.hidden = !text; }

// --------------------------------------------------------------------- //
// Sterowanie UI
// --------------------------------------------------------------------- //
function bindControls() {
  el.tabCamera.addEventListener("click", () => setSource("camera"));
  el.tabUpload.addEventListener("click", () => setSource("upload"));

  el.btnFlip.addEventListener("click", flipCamera);
  el.btnFsFlip.addEventListener("click", flipCamera);
  el.btnCamToggle.addEventListener("click", togglePause);
  el.btnFullscreen.addEventListener("click", toggleFullscreen);
  el.btnFsExit.addEventListener("click", exitFullscreen);
  document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement) el.stage.classList.remove("pseudo-fs");
    drawOverlay();
  });

  // Nakładka dopasowuje się do rozmiaru kontenera (fullscreen, resize okna, obrót telefonu).
  if (window.ResizeObserver) new ResizeObserver(() => drawOverlay()).observe(el.stage);
  window.addEventListener("orientationchange", () => setTimeout(drawOverlay, 200));

  // Regulacja wygładzania predykcji (suwak): 0 = brak, 100 = maksymalne.
  const applySmooth = () => {
    const v = +el.smoothRange.value;
    state.emaAlpha = 1 - (v / 100) * 0.9;   // v=0 -> alpha 1 (brak), v=100 -> alpha 0.1
    el.smoothVal.textContent = v === 0 ? "wył." : `${v}%`;
  };
  el.smoothRange.addEventListener("input", applySmooth);
  applySmooth();

  el.fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    el.photo.onload = () => { el.photo.hidden = false; setHint(""); analyzeOnce(); URL.revokeObjectURL(url); };
    el.photo.src = url;
  });
}

async function flipCamera() {
  state.facing = state.facing === "user" ? "environment" : "user";
  await startCamera();
}

// „Zatrzymaj" = zamrożenie klatki (podgląd wyników); „Wznów" = powrót do pętli.
// Pauzujemy element video (pokazuje ostatnią klatkę) i NIE czyścimy nakładki/wyników.
function togglePause() {
  if (state.paused) {
    state.paused = false;
    el.btnCamToggle.textContent = "⏸ Zatrzymaj";
    el.video.play().catch(() => {});
    setHint("");
  } else {
    state.paused = true;
    el.video.pause();
    el.btnCamToggle.textContent = "▶ Wznów";
  }
}

function updateMirror() {
  const mir = state.source === "camera" && state.facing === "user";
  el.video.classList.toggle("mirror", mir);
}

function toggleFullscreen() {
  if (el.stage.classList.contains("pseudo-fs")) exitFullscreen();
  else enterFullscreen();
}
function enterFullscreen() {
  el.stage.classList.add("pseudo-fs");
  if (el.stage.requestFullscreen) el.stage.requestFullscreen().catch(() => {});
}
function exitFullscreen() {
  el.stage.classList.remove("pseudo-fs");
  if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {});
}

async function setSource(mode) {
  if (mode === state.source) return;
  state.source = mode;
  state.emaProbs = null;
  const cam = mode === "camera";
  el.tabCamera.classList.toggle("active", cam);
  el.tabUpload.classList.toggle("active", !cam);
  el.tabCamera.setAttribute("aria-selected", cam);
  el.tabUpload.setAttribute("aria-selected", !cam);
  el.cameraControls.hidden = !cam;
  el.uploadControls.hidden = cam;
  el.video.hidden = !cam;
  el.photo.hidden = cam || !el.photo.src;   // nie pokazuj pustego <img> (ikona błędu)
  updateMirror();
  overlayCtx.clearRect(0, 0, el.overlay.width, el.overlay.height);

  if (cam) {
    if (!state.stream) {
      await startCamera();
    } else {
      state.paused = false;
      el.btnCamToggle.textContent = "⏸ Zatrzymaj";
      el.video.play().catch(() => {});
    }
  } else {
    stopStream();
    if (el.photo.src) analyzeOnce();
    else setHint("Wybierz zdjęcie z twarzą.");
  }
}

init();
