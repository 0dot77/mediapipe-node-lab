import { FilesetResolver, FaceLandmarker, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';
const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const graph = new LGraph();
const graphCanvas = new LGraphCanvas('#graph-canvas', graph);

const ui = {
  startBtn: document.querySelector('#start-btn'),
  stopBtn: document.querySelector('#stop-btn'),
  resetBtn: document.querySelector('#reset-btn'),
  statusPill: document.querySelector('#status-pill'),
  debug: document.querySelector('#debug-output'),
  graphWrap: document.querySelector('#graph-wrap'),
  stage: document.querySelector('#stage-canvas'),
  webcam: document.querySelector('#webcam'),
};

const state = {
  modelsReady: false,
  cameraReady: false,
  stream: null,
  faceLandmarker: null,
  handLandmarker: null,
  sourceCanvas: document.createElement('canvas'),
  overlayCanvas: document.createElement('canvas'),
  stageCtx: ui.stage.getContext('2d'),
  rafId: null,
  trail: [],
  frameCounter: 0,
  lastFaceMetrics: null,
  lastHandMetrics: null,
};

function setStatus(text, tone = 'info') {
  ui.statusPill.textContent = text;
  if (tone === 'ok') {
    ui.statusPill.style.background = '#dcfce7';
    ui.statusPill.style.color = '#166534';
    return;
  }
  if (tone === 'error') {
    ui.statusPill.style.background = '#fee2e2';
    ui.statusPill.style.color = '#991b1b';
    return;
  }
  ui.statusPill.style.background = '#dbeafe';
  ui.statusPill.style.color = '#1e40af';
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function distance2d(a, b) {
  if (!a || !b) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function resizeGraphCanvas() {
  const rect = ui.graphWrap.getBoundingClientRect();
  const width = Math.max(240, Math.floor(rect.width));
  const height = Math.max(220, Math.floor(rect.height));
  const canvasEl = graphCanvas.canvas;
  canvasEl.width = width;
  canvasEl.height = height;
  graphCanvas.setDirty(true, true);
}

function resizeStageCanvas() {
  const rect = ui.stage.getBoundingClientRect();
  ui.stage.width = Math.max(200, Math.floor(rect.width));
  ui.stage.height = Math.max(160, Math.floor(rect.height));
}

function drawCoverImage(ctx, image, targetWidth, targetHeight) {
  if (!image) return;
  const sourceWidth = image.width;
  const sourceHeight = image.height;
  if (!sourceWidth || !sourceHeight) return;

  const sourceRatio = sourceWidth / sourceHeight;
  const targetRatio = targetWidth / targetHeight;
  let drawWidth;
  let drawHeight;
  let dx = 0;
  let dy = 0;

  if (sourceRatio > targetRatio) {
    drawHeight = targetHeight;
    drawWidth = drawHeight * sourceRatio;
    dx = (targetWidth - drawWidth) * 0.5;
  } else {
    drawWidth = targetWidth;
    drawHeight = drawWidth / sourceRatio;
    dy = (targetHeight - drawHeight) * 0.5;
  }

  ctx.drawImage(image, dx, dy, drawWidth, drawHeight);
}

function drawPoints(ctx, landmarks, color, pointSize = 2, step = 6) {
  if (!landmarks || landmarks.length === 0) return;
  ctx.fillStyle = color;
  for (let i = 0; i < landmarks.length; i += step) {
    const p = landmarks[i];
    ctx.beginPath();
    ctx.arc(p.x * ctx.canvas.width, p.y * ctx.canvas.height, pointSize, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawHandConnections(ctx, landmarks, color) {
  if (!landmarks || landmarks.length === 0) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [from, to] of HAND_CONNECTIONS) {
    const a = landmarks[from];
    const b = landmarks[to];
    if (!a || !b) continue;
    ctx.moveTo(a.x * ctx.canvas.width, a.y * ctx.canvas.height);
    ctx.lineTo(b.x * ctx.canvas.width, b.y * ctx.canvas.height);
  }
  ctx.stroke();
}

async function ensureModelsLoaded() {
  if (state.modelsReady) return;
  setStatus('Loading models...');

  const vision = await FilesetResolver.forVisionTasks(WASM_URL);

  state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: FACE_MODEL_URL },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
  });

  state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL_URL },
    runningMode: 'VIDEO',
    numHands: 2,
  });

  state.modelsReady = true;
  setStatus('Models ready', 'ok');
}

async function startCamera() {
  if (state.stream) return;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: 'user',
    },
    audio: false,
  });

  ui.webcam.srcObject = stream;
  await ui.webcam.play();
  state.stream = stream;
  state.cameraReady = true;
  setStatus('Camera running', 'ok');
}

function stopCamera() {
  if (!state.stream) return;
  for (const track of state.stream.getTracks()) {
    track.stop();
  }
  state.stream = null;
  state.cameraReady = false;
  ui.webcam.srcObject = null;
  setStatus('Camera stopped');
}

function updateDebugOutput(controls) {
  const payload = {
    modelsReady: state.modelsReady,
    cameraReady: state.cameraReady,
    frame: state.frameCounter,
    face: state.lastFaceMetrics,
    hand: state.lastHandMetrics,
    controls,
  };
  ui.debug.textContent = JSON.stringify(payload, null, 2);
}

function drawStage(baseImage, controls = {}) {
  const ctx = state.stageCtx;
  const w = ui.stage.width;
  const h = ui.stage.height;
  const time = performance.now() * 0.001;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0b1020';
  ctx.fillRect(0, 0, w, h);

  if (baseImage) {
    ctx.save();
    ctx.globalAlpha = 0.95;
    drawCoverImage(ctx, baseImage, w, h);
    ctx.restore();
  }

  const tilt = controls.tilt ?? 0;
  const lift = controls.lift ?? 0;
  const pinch = controls.pinch ?? 0;
  const jaw = controls.jaw ?? 0;
  const presence = controls.presence ?? 0;

  const cx = w * clamp(0.5 + tilt * 1.1, 0.05, 0.95);
  const cy = h * clamp(0.7 - lift * 0.8, 0.08, 0.92);
  const radius = 40 + pinch * 240 + jaw * 2000;
  const hue = (time * 85 + pinch * 220) % 360;

  if (presence > 0) {
    state.trail.push({ x: cx, y: cy, life: 1 });
    if (state.trail.length > 45) {
      state.trail.shift();
    }
  }

  for (const p of state.trail) {
    p.life = Math.max(0, p.life - 0.02);
  }
  state.trail = state.trail.filter((p) => p.life > 0);

  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  for (const p of state.trail) {
    ctx.fillStyle = `hsla(${(hue + p.life * 80) % 360}, 95%, 70%, ${0.12 * p.life})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 10 + 80 * p.life * (0.2 + pinch), 0, Math.PI * 2);
    ctx.fill();
  }

  const gradient = ctx.createRadialGradient(cx, cy, 6, cx, cy, radius);
  gradient.addColorStop(0, `hsla(${hue}, 96%, 72%, 0.55)`);
  gradient.addColorStop(0.5, `hsla(${(hue + 40) % 360}, 96%, 62%, 0.18)`);
  gradient.addColorStop(1, 'hsla(240, 80%, 20%, 0)');
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  ctx.strokeStyle = `hsla(${hue}, 90%, 75%, 0.8)`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, Math.max(20, radius * 0.4 + Math.sin(time * 5) * 8), 0, Math.PI * 2);
  ctx.stroke();
}

function WebcamSourceNode() {
  this.addOutput('frame', 'canvas');
  this.addOutput('timeMs', 'number');
  this.properties = { mirror: true };
  this.size = [220, 90];
}
WebcamSourceNode.title = 'Webcam Source';
WebcamSourceNode.prototype.onExecute = function onExecute() {
  if (!state.cameraReady || ui.webcam.readyState < 2) {
    this.setOutputData(0, null);
    this.setOutputData(1, performance.now());
    return;
  }

  const videoWidth = ui.webcam.videoWidth || 1280;
  const videoHeight = ui.webcam.videoHeight || 720;
  if (state.sourceCanvas.width !== videoWidth || state.sourceCanvas.height !== videoHeight) {
    state.sourceCanvas.width = videoWidth;
    state.sourceCanvas.height = videoHeight;
  }

  const ctx = state.sourceCanvas.getContext('2d');
  ctx.save();
  ctx.clearRect(0, 0, videoWidth, videoHeight);
  if (this.properties.mirror) {
    ctx.translate(videoWidth, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(ui.webcam, 0, 0, videoWidth, videoHeight);
  ctx.restore();

  this.setOutputData(0, state.sourceCanvas);
  this.setOutputData(1, performance.now());
};

function FaceExtractNode() {
  this.addInput('frame', 'canvas');
  this.addInput('timeMs', 'number');
  this.addOutput('landmarks', 'array');
  this.addOutput('metrics', 'object');
  this.size = [210, 100];
}
FaceExtractNode.title = 'Face Extract';
FaceExtractNode.prototype.onExecute = function onExecute() {
  const frame = this.getInputData(0);
  const timeMs = this.getInputData(1) ?? performance.now();
  if (!frame || !state.modelsReady || !state.faceLandmarker) {
    this.setOutputData(0, null);
    this.setOutputData(1, null);
    state.lastFaceMetrics = null;
    return;
  }

  const result = state.faceLandmarker.detectForVideo(frame, timeMs);
  const faces = result.faceLandmarks || [];

  const metrics = {
    count: faces.length,
    noseX: 0.5,
    noseY: 0.5,
    jaw: 0,
  };

  if (faces[0]) {
    const nose = faces[0][1] || faces[0][4] || faces[0][0];
    const upperLip = faces[0][13];
    const lowerLip = faces[0][14];
    if (nose) {
      metrics.noseX = nose.x;
      metrics.noseY = nose.y;
    }
    metrics.jaw = distance2d(upperLip, lowerLip);
  }

  state.lastFaceMetrics = metrics;
  this.setOutputData(0, faces);
  this.setOutputData(1, metrics);
};

function HandExtractNode() {
  this.addInput('frame', 'canvas');
  this.addInput('timeMs', 'number');
  this.addOutput('landmarks', 'array');
  this.addOutput('metrics', 'object');
  this.size = [210, 100];
}
HandExtractNode.title = 'Hand Extract';
HandExtractNode.prototype.onExecute = function onExecute() {
  const frame = this.getInputData(0);
  const timeMs = this.getInputData(1) ?? performance.now();
  if (!frame || !state.modelsReady || !state.handLandmarker) {
    this.setOutputData(0, null);
    this.setOutputData(1, null);
    state.lastHandMetrics = null;
    return;
  }

  const result = state.handLandmarker.detectForVideo(frame, timeMs);
  const hands = result.landmarks || [];
  const hand = hands[0];

  const metrics = {
    count: hands.length,
    pinch: 0,
    palmY: 0.5,
  };

  if (hand) {
    const thumbTip = hand[4];
    const indexTip = hand[8];
    const wrist = hand[0];
    const pinchDistance = distance2d(thumbTip, indexTip);
    metrics.pinch = clamp(1 - pinchDistance * 8, 0, 1);
    metrics.palmY = wrist?.y ?? 0.5;
  }

  state.lastHandMetrics = metrics;
  this.setOutputData(0, hands);
  this.setOutputData(1, metrics);
};

function LandmarkOverlayNode() {
  this.addInput('frame', 'canvas');
  this.addInput('faceLandmarks', 'array');
  this.addInput('handLandmarks', 'array');
  this.addOutput('image', 'canvas');
  this.properties = { showFace: true, showHands: true };
  this.size = [250, 125];
}
LandmarkOverlayNode.title = 'Landmark Overlay';
LandmarkOverlayNode.prototype.onExecute = function onExecute() {
  const frame = this.getInputData(0);
  const faces = this.getInputData(1) || [];
  const hands = this.getInputData(2) || [];
  if (!frame) {
    this.setOutputData(0, null);
    return;
  }

  if (state.overlayCanvas.width !== frame.width || state.overlayCanvas.height !== frame.height) {
    state.overlayCanvas.width = frame.width;
    state.overlayCanvas.height = frame.height;
  }

  const ctx = state.overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, state.overlayCanvas.width, state.overlayCanvas.height);
  ctx.drawImage(frame, 0, 0);

  if (this.properties.showFace && faces[0]) {
    drawPoints(ctx, faces[0], '#f97316', 1.6, 5);
  }

  if (this.properties.showHands) {
    for (const hand of hands) {
      drawHandConnections(ctx, hand, '#38bdf8');
      drawPoints(ctx, hand, '#22d3ee', 2.4, 1);
    }
  }

  this.setOutputData(0, state.overlayCanvas);
};

function GestureMapperNode() {
  this.addInput('faceMetrics', 'object');
  this.addInput('handMetrics', 'object');
  this.addOutput('controls', 'object');
  this.size = [215, 85];
}
GestureMapperNode.title = 'Gesture Mapper';
GestureMapperNode.prototype.onExecute = function onExecute() {
  const face = this.getInputData(0) || {};
  const hand = this.getInputData(1) || {};

  const controls = {
    tilt: (face.noseX ?? 0.5) - 0.5,
    lift: 1 - (hand.palmY ?? 0.5),
    pinch: hand.pinch ?? 0,
    jaw: face.jaw ?? 0,
    presence: Math.max(face.count ? 1 : 0, hand.count ? 1 : 0),
  };

  this.setOutputData(0, controls);
};

function StageOutputNode() {
  this.addInput('image', 'canvas');
  this.addInput('controls', 'object');
  this.size = [210, 85];
}
StageOutputNode.title = 'Stage Output';
StageOutputNode.prototype.onExecute = function onExecute() {
  const image = this.getInputData(0);
  const controls = this.getInputData(1) || {};
  drawStage(image, controls);
  updateDebugOutput(controls);
};

LiteGraph.registerNodeType('input/webcam_source', WebcamSourceNode);
LiteGraph.registerNodeType('extract/face', FaceExtractNode);
LiteGraph.registerNodeType('extract/hand', HandExtractNode);
LiteGraph.registerNodeType('draw/overlay', LandmarkOverlayNode);
LiteGraph.registerNodeType('map/gesture', GestureMapperNode);
LiteGraph.registerNodeType('output/stage', StageOutputNode);

function createDefaultGraph() {
  graph.clear();

  const webcam = LiteGraph.createNode('input/webcam_source');
  const face = LiteGraph.createNode('extract/face');
  const hand = LiteGraph.createNode('extract/hand');
  const overlay = LiteGraph.createNode('draw/overlay');
  const mapper = LiteGraph.createNode('map/gesture');
  const stage = LiteGraph.createNode('output/stage');

  webcam.pos = [40, 90];
  face.pos = [320, 30];
  hand.pos = [320, 220];
  overlay.pos = [610, 90];
  mapper.pos = [610, 290];
  stage.pos = [900, 190];

  graph.add(webcam);
  graph.add(face);
  graph.add(hand);
  graph.add(overlay);
  graph.add(mapper);
  graph.add(stage);

  webcam.connect(0, face, 0);
  webcam.connect(1, face, 1);
  webcam.connect(0, hand, 0);
  webcam.connect(1, hand, 1);

  webcam.connect(0, overlay, 0);
  face.connect(0, overlay, 1);
  hand.connect(0, overlay, 2);

  face.connect(1, mapper, 0);
  hand.connect(1, mapper, 1);

  overlay.connect(0, stage, 0);
  mapper.connect(0, stage, 1);
}

function startGraphLoop() {
  if (state.rafId) return;
  const loop = () => {
    state.frameCounter += 1;
    graph.runStep(1, false);
    state.rafId = requestAnimationFrame(loop);
  };
  state.rafId = requestAnimationFrame(loop);
}

function bootstrapEvents() {
  ui.startBtn.addEventListener('click', async () => {
    ui.startBtn.disabled = true;
    try {
      await ensureModelsLoaded();
      await startCamera();
      startGraphLoop();
    } catch (error) {
      console.error(error);
      setStatus('Failed to start', 'error');
    } finally {
      ui.startBtn.disabled = false;
    }
  });

  ui.stopBtn.addEventListener('click', () => {
    stopCamera();
  });

  ui.resetBtn.addEventListener('click', () => {
    createDefaultGraph();
    setStatus('Graph reset');
  });

  window.addEventListener('resize', () => {
    resizeGraphCanvas();
    resizeStageCanvas();
  });
}

function bootstrap() {
  graphCanvas.allow_dragnodes = true;
  graphCanvas.allow_dragcanvas = true;
  graphCanvas.zoom_modify_alpha = false;
  graphCanvas.background_image = null;

  resizeGraphCanvas();
  resizeStageCanvas();
  createDefaultGraph();
  drawStage(null, {});
  updateDebugOutput({});
  startGraphLoop();
  bootstrapEvents();
  setStatus('Idle');
}

bootstrap();
