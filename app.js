/* ===================================================
   ASL PORTAL — Custom Web Frontend Script
   Pure client-side MediaPipe + Custom Neural Net
   =================================================== */

// Constants
const CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],        // Index
  [5, 9], [9, 10], [10, 11], [11, 12],   // Middle
  [9, 13], [13, 14], [14, 15], [15, 16], // Ring
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
];

const ASL_DICTIONARY = {
  "A": ["✊", "Closed fist, thumb resting flat along the index finger."],
  "B": ["✋", "Open flat hand, four fingers upright, thumb crossed over palm."],
  "C": ["👌", "Fingers curved to form a semi-circle shape, resembling 'C'."],
  "D": ["☝️", "Index finger pointing straight up, other fingers touching thumb."],
  "E": ["✊", "Fist with all fingers curled tight, touching the thumb."],
  "F": ["👌", "Index and thumb form a circle, other three fingers extended."],
  "G": ["👉", "Index and thumb extended parallel, pointing to the side."],
  "H": ["👉", "Index and middle fingers extended side-by-side pointing horizontally."],
  "I": ["🤙", "Pinky finger extended straight up, other fingers curled."],
  "J": ["🤙", "Pinky finger extended, tracing a 'J' hook curve in the air."],
  "K": ["✌️", "Index and middle finger pointing up in 'V', thumb touching middle finger."],
  "L": ["☝️", "Index pointing up and thumb pointing out forming an 'L' shape."],
  "M": ["✊", "Fist with thumb tucked under the index, middle, and ring fingers."],
  "N": ["✊", "Fist with thumb tucked under the index and middle fingers."],
  "O": ["👌", "All fingers curved to touch thumb, forming an 'O' circle."],
  "P": ["✌️", "Index and middle finger extended down, thumb touching middle finger."],
  "Q": ["👉", "Index and thumb pointing down, parallel to each other."],
  "R": ["🤞", "Index and middle fingers crossed tightly together."],
  "S": ["✊", "Closed fist, thumb wrapped across the front of the fingers."],
  "T": ["✊", "Fist with thumb tucked under the index finger only."],
  "U": ["✌️", "Index and middle fingers extended straight up, pressed together."],
  "V": ["✌️", "Index and middle fingers extended up and spread apart in 'V'."],
  "W": ["🖐️", "Index, middle, and ring fingers extended up in 'W' shape."],
  "X": ["☝️", "Index finger curled into a hook, other fingers closed in fist."],
  "Y": ["🤙", "Thumb and pinky extended, middle three fingers closed."],
  "Z": ["☝️", "Index finger extended, tracing a 'Z' path in the air."]
};

const LANDMARKS_REF = [
  "WRIST (Anchor Point)",
  "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
  "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
  "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
  "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
  "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
];

// App Global States
let modelWeights = null;
let activePage = "dashboard";
let transcribedText = "";
let historyLog = [
  { time: "18:01:05", letter: "A", hand: "Right", confidence: 0.98 },
  { time: "18:01:15", letter: "S", hand: "Right", confidence: 0.94 },
  { time: "18:01:25", letter: "L", hand: "Left", confidence: 0.96 }
];

// Settings Thresholds
let confThreshold = 0.60;
let debounceFrames = 10;

// Typing Debouncer States
let lastStableLetter = "";
let stableCount = 0;

// Quiz States
let quizScore = 0;
let quizLetter = "A";
let quizStableFrames = 0;

// Camera Loops
let activeCamera = null;
let isWebcamRunning = false;
let isQuizWebcamRunning = false;

// DOM Elements
const videoElement = document.getElementById("webcam-video");
const canvasElement = document.getElementById("output-canvas");
const canvasCtx = canvasElement.getContext("2d");
const placeholderOverlay = document.getElementById("placeholder-overlay");
const liveStatusBar = document.getElementById("live-status-bar");
const transcribedTextArea = document.getElementById("transcribed-text");

const quizCanvas = document.getElementById("quiz-canvas");
const quizCanvasCtx = quizCanvas.getContext("2d");
const quizPlaceholderOverlay = document.getElementById("quiz-placeholder-overlay");
const quizLiveStatusBar = document.getElementById("quiz-live-status-bar");
const quizScoreVal = document.getElementById("quiz-score");
const quizTargetLetter = document.getElementById("quiz-target-letter");
const quizTargetEmoji = document.getElementById("quiz-target-emoji");
const quizTargetHint = document.getElementById("quiz-target-hint");
const quizFeedbackBox = document.getElementById("quiz-feedback-box");

// ── INITIALIZATION ────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  // 1. Setup navigation
  setupNavigation();

  // 2. Render ASL Dictionary & Landmark Map listings & Contributors
  renderASLDictionary();
  renderLandmarkMap();
  renderContributors();

  // 3. Setup Event Listeners
  setupEventListeners();

  // 4. Render Initial Logs
  renderLogs();

  // 5. Load weights asynchronously in the background
  loadModelWeights();
});

async function loadModelWeights() {
  try {
    const response = await fetch("model_weights.json");
    modelWeights = await response.json();
    console.log("Neural Net weights loaded successfully:", modelWeights);
  } catch (err) {
    console.error("Failed to load neural network weights:", err);
    alert("Warning: model_weights.json not found! Running in mock classification fallback.");
  }
}

// ── NAVIGATION MENU ROUTER ───────────────────────────────────────────────────
function setupNavigation() {
  const navButtons = document.querySelectorAll(".nav-btn");
  const sections = document.querySelectorAll(".page-section");

  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      // Remove active states
      navButtons.forEach(b => b.classList.remove("active"));
      sections.forEach(s => s.classList.remove("active"));

      // Set active
      btn.classList.add("active");
      const targetPage = btn.getAttribute("data-page");
      document.getElementById(`${targetPage}-page`).classList.add("active");
      
      activePage = targetPage;
      console.log("Navigated to page:", activePage);

      // Manage Webcams depending on page
      if (activePage !== "translation" && isWebcamRunning) {
        document.getElementById("webcam-checkbox").checked = false;
        toggleWebcam(false);
      }
      if (activePage !== "quiz" && isQuizWebcamRunning) {
        document.getElementById("quiz-webcam-checkbox").checked = false;
        toggleQuizWebcam(false);
      }
    });
  });
}

// ── EVENT LISTENERS SETUP ─────────────────────────────────────────────────────
function setupEventListeners() {
  // Transcribe actions
  document.getElementById("space-btn").addEventListener("click", () => {
    transcribedText += " ";
    transcribedTextArea.value = transcribedText;
  });
  document.getElementById("backspace-btn").addEventListener("click", () => {
    transcribedText = transcribedText.slice(0, -1);
    transcribedTextArea.value = transcribedText;
  });
  document.getElementById("clear-btn").addEventListener("click", () => {
    transcribedText = "";
    transcribedTextArea.value = transcribedText;
  });

  // TTS Voice Synthesis
  document.getElementById("speak-btn").addEventListener("click", () => {
    if (transcribedText) {
      const utterance = new SpeechSynthesisUtterance(transcribedText);
      window.speechSynthesis.speak(utterance);
    } else {
      alert("Transcription sentence is empty. Sign some letters first!");
    }
  });

  // Text file download
  document.getElementById("download-btn").addEventListener("click", () => {
    if (!transcribedText) {
      alert("Output text is empty. Nothing to download!");
      return;
    }
    const blob = new Blob([transcribedText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "asl_translation.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

  // Sliders
  const confSlider = document.getElementById("conf-slider");
  const confVal = document.getElementById("conf-val");
  const debounceSlider = document.getElementById("debounce-slider");
  const debounceVal = document.getElementById("debounce-val");

  const settingsConfSlider = document.getElementById("settings-conf-slider");
  const settingsConfVal = document.getElementById("settings-conf-val");
  const settingsDebounceSlider = document.getElementById("settings-debounce-slider");
  const settingsDebounceVal = document.getElementById("settings-debounce-val");

  // Sync sliders
  function updateConf(val) {
    confThreshold = val / 100;
    confSlider.value = val;
    settingsConfSlider.value = val;
    confVal.textContent = `${val}%`;
    settingsConfVal.textContent = `${val}%`;
  }
  function updateDebounce(val) {
    debounceFrames = parseInt(val);
    debounceSlider.value = val;
    settingsDebounceSlider.value = val;
    debounceVal.textContent = val;
    settingsDebounceVal.textContent = val;
  }

  confSlider.addEventListener("input", (e) => updateConf(e.target.value));
  settingsConfSlider.addEventListener("input", (e) => updateConf(e.target.value));
  debounceSlider.addEventListener("input", (e) => updateDebounce(e.target.value));
  settingsDebounceSlider.addEventListener("input", (e) => updateDebounce(e.target.value));

  // Webcam checkbox toggles
  document.getElementById("webcam-checkbox").addEventListener("change", (e) => {
    toggleWebcam(e.target.checked);
  });
  document.getElementById("quiz-webcam-checkbox").addEventListener("change", (e) => {
    toggleQuizWebcam(e.target.checked);
  });

  // Quiz HUD buttons
  document.getElementById("quiz-skip-btn").addEventListener("click", loadNewQuizLetter);
  document.getElementById("quiz-reset-btn").addEventListener("click", () => {
    quizScore = 0;
    quizScoreVal.textContent = "0 pts";
    quizFeedbackBox.textContent = "Score counter reset. Start signing!";
  });

  document.getElementById("clear-history-btn").addEventListener("click", () => {
    historyLog = [];
    renderLogs();
  });
}

// ── WEBCAM TOGGLE CONTROLLER ──────────────────────────────────────────────────
function toggleWebcam(start) {
  isWebcamRunning = start;
  if (start) {
    placeholderOverlay.style.display = "none";
    liveStatusBar.style.display = "flex";
    videoElement.style.display = "block";
    startMediaPipe(canvasElement, onResultsTranslation);
  } else {
    stopCamera();
    placeholderOverlay.style.display = "flex";
    liveStatusBar.style.display = "none";
    videoElement.style.display = "none";
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  }
}

function toggleQuizWebcam(start) {
  isQuizWebcamRunning = start;
  if (start) {
    quizPlaceholderOverlay.style.display = "none";
    quizLiveStatusBar.style.display = "flex";
    videoElement.style.display = "block";
    startMediaPipe(quizCanvas, onResultsQuiz);
    loadNewQuizLetter();
  } else {
    stopCamera();
    quizPlaceholderOverlay.style.display = "flex";
    quizLiveStatusBar.style.display = "none";
    videoElement.style.display = "none";
    quizCanvasCtx.clearRect(0, 0, quizCanvas.width, quizCanvas.height);
  }
}

// ── MEDIAPIPE INITIALIZATION ─────────────────────────────────────────────────
let mpHands = null;

function startMediaPipe(canvasTarget, resultsCallback) {
  if (!mpHands) {
    mpHands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    mpHands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
  }

  // Clear previous callback and set the new one
  mpHands.onResults(resultsCallback);

  if (!activeCamera) {
    activeCamera = new Camera(videoElement, {
      onFrame: async () => {
        if (isWebcamRunning || isQuizWebcamRunning) {
          await mpHands.send({ image: videoElement });
        }
      },
      width: 640,
      height: 480
    });
    activeCamera.start().catch(err => {
      alert("Error starting camera: Please check browser camera permissions.");
      console.error(err);
    });
  }
}

function stopCamera() {
  // We keep the camera running internally, but stop processing predictions
  // If we want a hard stop to turn the webcam green light off:
  if (activeCamera) {
    try {
      const stream = videoElement.srcObject;
      if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
      }
      videoElement.srcObject = null;
    } catch (e) {
      console.error(e);
    }
    activeCamera = null;
  }
}

// ── NEURAL NETWORK FEEDFORWARD INFERENCE ──────────────────────────────────────
function relu(arr) {
  return arr.map(x => Math.max(0, x));
}

function mish(x) {
  return x * Math.tanh(x > 20 ? x : Math.log1p(Math.exp(x)));
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / (sum || 1));
}

function batchNormalize(x, gamma, beta, mean, variance, epsilon = 0.001) {
  let output = [];
  for (let i = 0; i < x.length; i++) {
    output[i] = ((x[i] - mean[i]) / Math.sqrt(variance[i] + epsilon)) * gamma[i] + beta[i];
  }
  return output;
}

function dense(x, weights, biases, activation = "mish") {
  let output = [];
  let inputDim = x.length;
  let outputDim = biases.length;
  for (let j = 0; j < outputDim; j++) {
    let sum = biases[j];
    for (let i = 0; i < inputDim; i++) {
      sum += x[i] * weights[i][j];
    }
    if (activation === "mish") {
      output[j] = mish(sum);
    } else if (activation === "relu") {
      output[j] = Math.max(0, sum);
    } else {
      output[j] = sum; // softmax / linear
    }
  }
  return output;
}

// Full inference pipeline forward pass
function predictHandSign(features) {
  if (!modelWeights) {
    // If weights are missing, fall back to a random class prediction
    let mockOutput = Array.from({ length: 26 }, () => Math.random());
    return softmax(mockOutput);
  }

  // Layer 0: BatchNormalization
  let x = features;
  let bn = modelWeights[0].params;
  x = batchNormalize(x, bn[0], bn[1], bn[2], bn[3]);

  // Layer 1: Dense (128 units, mish)
  let d1 = modelWeights[1].params;
  x = dense(x, d1[0], d1[1], "mish");

  // Layer 2: Dense (64 units, mish)
  let d2 = modelWeights[2].params;
  x = dense(x, d2[0], d2[1], "mish");

  // Layer 3: Dense (32 units, mish)
  let d3 = modelWeights[3].params;
  x = dense(x, d3[0], d3[1], "mish");

  // Layer 4: Dense (26 units, softmax)
  let d4 = modelWeights[4].params;
  x = dense(x, d4[0], d4[1], "softmax");
  return softmax(x);
}

// ── MEDIAPIPE TRANSLATION CALLBACK ──────────────────────────────────────────
function onResultsTranslation(results) {
  const canvasCtx = canvasElement.getContext("2d");
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Mirror camera frame on canvas
  canvasCtx.translate(canvasElement.width, 0);
  canvasCtx.scale(-1, 1);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.restore(); // Restore coordinates for HUD annotations

  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let idx = 0; idx < results.multiHandLandmarks.length; idx++) {
      const landmarks = results.multiHandLandmarks[idx];
      const handedness = results.multiHandedness[idx].label; // "Left" or "Right"

      // 1. Preprocess & extract features
      const features = processLandmarks(landmarks, handedness);

      // 2. Classify
      const predictions = predictHandSign(features);
      const classIdx = predictions.indexOf(Math.max(...predictions));
      const predictedLetter = CLASSES[classIdx];
      const confidence = predictions[classIdx];

      // 3. Render landmarks overlay
      const color = handedness === "Right" ? "#4F8EF7" : "#2DD4BF";
      drawLandmarksHUD(canvasCtx, landmarks, color);

      // 4. Threshold & Typing debouncer logic
      if (confidence >= confThreshold) {
        const labelText = `${handedness[0]}  ${predictedLetter} (${(confidence * 100).toFixed(0)}%)`;
        drawBoundingBoxHUD(canvasCtx, landmarks, labelText, color);

        // Typing logic
        if (predictedLetter === lastStableLetter) {
          stableCount++;
        } else {
          lastStableLetter = predictedLetter;
          stableCount = 0;
        }

        if (stableCount === debounceFrames) {
          if (!transcribedText || transcribedText[transcribedText.length - 1] !== predictedLetter) {
            transcribedText += predictedLetter;
            transcribedTextArea.value = transcribedText;

            // Log history
            const time = new Date().toTimeString().split(" ")[0];
            historyLog.unshift({
              time: time,
              letter: predictedLetter,
              hand: handedness,
              confidence: confidence
            });
            renderLogs();
          }
        }
      } else {
        // Unsure prediction
        drawBoundingBoxHUD(canvasCtx, landmarks, `${handedness[0]} ? (${(confidence * 100).toFixed(0)}%)`, "#6B7280");
      }
    }
  } else {
    // Hint
    canvasCtx.fillStyle = "#8891AA";
    canvasCtx.font = "14px Inter";
    canvasCtx.fillText("Show your hand to the camera", 20, canvasElement.height - 20);
    
    // Reset stable typing count if no hand
    stableCount = 0;
    lastStableLetter = "";
  }
}

// ── MEDIAPIPE QUIZ CALLBACK ──────────────────────────────────────────────────
function onResultsQuiz(results) {
  const quizCanvasCtx = quizCanvas.getContext("2d");
  quizCanvasCtx.save();
  quizCanvasCtx.clearRect(0, 0, quizCanvas.width, quizCanvas.height);

  // Mirror camera frame
  quizCanvasCtx.translate(quizCanvas.width, 0);
  quizCanvasCtx.scale(-1, 1);
  quizCanvasCtx.drawImage(results.image, 0, 0, quizCanvas.width, quizCanvas.height);
  quizCanvasCtx.restore();

  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let idx = 0; idx < results.multiHandLandmarks.length; idx++) {
      const landmarks = results.multiHandLandmarks[idx];
      const handedness = results.multiHandedness[idx].label;

      const features = processLandmarks(landmarks, handedness);
      const predictions = predictHandSign(features);
      const classIdx = predictions.indexOf(Math.max(...predictions));
      const predictedLetter = CLASSES[classIdx];
      const confidence = predictions[classIdx];

      // Draw outlines
      const color = predictedLetter === quizLetter ? "#2DD4BF" : "#4F8EF7";
      drawLandmarksHUD(quizCanvasCtx, landmarks, color);

      if (confidence >= 0.70) {
        const labelText = `${handedness[0]}  ${predictedLetter} (${(confidence * 100).toFixed(0)}%)`;
        drawBoundingBoxHUD(quizCanvasCtx, landmarks, labelText, color);

        // Quiz check
        if (predictedLetter === quizLetter) {
          quizStableFrames++;
          
          // Render progress HUD text
          quizCanvasCtx.fillStyle = "#2DD4BF";
          quizCanvasCtx.font = "bold 16px Space Grotesk";
          quizCanvasCtx.fillText(`Verifying Sign: ${quizStableFrames * 10}%`, 20, 40);

          if (quizStableFrames === 10) {
            quizScore++;
            quizScoreVal.textContent = `${quizScore} pts`;
            quizFeedbackBox.innerHTML = `<span style="color:var(--accent-green);font-weight:700;">🎉 Excellent! Correctly signed '${quizLetter}' (+1 point).</span>`;
            quizStableFrames = 0;
            loadNewQuizLetter();
          }
        } else {
          quizStableFrames = 0;
        }
      } else {
        drawBoundingBoxHUD(quizCanvasCtx, landmarks, `${handedness[0]} ?`, "#6B7280");
        quizStableFrames = 0;
      }
    }
  } else {
    quizStableFrames = 0;
    quizCanvasCtx.fillStyle = "#8891AA";
    quizCanvasCtx.font = "14px Inter";
    quizCanvasCtx.fillText(`Make ASL sign for '${quizLetter}'`, 20, quizCanvas.height - 20);
  }
}

// ── COORDINATE PREPROCESSING NORMALIZATION ────────────────────────────────────
function processLandmarks(landmarks, handedness) {
  const wristX = landmarks[0].x;
  const wristY = landmarks[0].y;

  // 1. Shift
  let shifted = landmarks.map(lm => [lm.x - wristX, lm.y - wristY]);

  // 2. Scale
  let distances = shifted.map(coord => Math.sqrt(coord[0]*coord[0] + coord[1]*coord[1]));
  let maxDist = Math.max(...distances);

  if (maxDist > 0) {
    shifted = shifted.map(coord => [coord[0] / maxDist, coord[1] / maxDist]);
  }

  // 3. Mirror Left Hand
  if (handedness === "Left") {
    shifted = shifted.map(coord => [-coord[0], coord[1]]);
  }

  // 4. Flatten
  let flat = [];
  shifted.forEach(coord => {
    flat.push(coord[0]);
    flat.push(coord[1]);
  });
  return flat;
}

// ── CANVAS ANNOTATIONS DRAWING HUD ────────────────────────────────────────────
function drawLandmarksHUD(ctx, landmarks, color) {
  const h = ctx.canvas.height;
  const w = ctx.canvas.width;

  // Draw joints
  landmarks.forEach(lm => {
    // Coordinates are mirrored back because canvas frame is mirrored
    // landmarks coordinates are normalized [0, 1] relative to video frame width
    // To match our HUD coords, we must map them to canvas resolution
    // Since the camera frame is drawn mirrored, the pixel X coordinate should be (1 - lm.x) * w
    const x = int((1 - lm.x) * w);
    const y = int(lm.y * h);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  });

  // Draw connections
  ctx.strokeStyle = "rgba(167, 139, 250, 0.4)";
  ctx.lineWidth = 2;
  HAND_CONNECTIONS.forEach(([start, end]) => {
    const x1 = int((1 - landmarks[start].x) * w);
    const y1 = int(landmarks[start].y * h);
    const x2 = int((1 - landmarks[end].x) * w);
    const y2 = int(landmarks[end].y * h);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  });
}

function drawBoundingBoxHUD(ctx, landmarks, label, color) {
  const h = ctx.canvas.height;
  const w = ctx.canvas.width;

  const xs = landmarks.map(lm => int((1 - lm.x) * w));
  const ys = landmarks.map(lm => int(lm.y * h));

  const x1 = Math.max(0, Math.min(...xs) - 20);
  const y1 = Math.max(0, Math.min(...ys) - 20);
  const x2 = Math.min(w, Math.max(...xs) + 20);
  const y2 = Math.min(h, Math.max(...ys) + 20);

  // Draw tech corner brackets
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  const corner = 12;

  // Top-left
  ctx.beginPath(); ctx.moveTo(x1, y1 + corner); ctx.lineTo(x1, y1); ctx.lineTo(x1 + corner, y1); ctx.stroke();
  // Top-right
  ctx.beginPath(); ctx.moveTo(x2 - corner, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + corner); ctx.stroke();
  // Bottom-left
  ctx.beginPath(); ctx.moveTo(x1, y2 - corner); ctx.lineTo(x1, y2); ctx.lineTo(x1 + corner, y2); ctx.stroke();
  // Bottom-right
  ctx.beginPath(); ctx.moveTo(x2 - corner, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - corner); ctx.stroke();

  // Label pill background
  ctx.font = "bold 14px Inter";
  const textWidth = ctx.measureText(label).width;
  const textHeight = 14;

  ctx.fillStyle = color;
  ctx.fillRect(x1, y1 - textHeight - 16, textWidth + 14, textHeight + 12);

  // White text
  ctx.fillStyle = "#FFFFFF";
  ctx.fillText(label, x1 + 7, y1 - 8);
}

function int(val) {
  return Math.floor(val);
}

// ── QUIZ LETTER LOADER ────────────────────────────────────────────────────────
function loadNewQuizLetter() {
  const filtered = CLASSES.filter(x => x !== quizLetter);
  quizLetter = filtered[Math.floor(Math.random() * filtered.length)];
  
  const [emoji, desc] = ASL_DICTIONARY[quizLetter];
  
  quizTargetLetter.textContent = quizLetter;
  quizTargetEmoji.textContent = emoji;
  quizTargetHint.innerHTML = `<b>Hint:</b> ${desc}`;
  quizStableFrames = 0;
}

// ── LOG TABLES RENDERING ──────────────────────────────────────────────────────
function renderLogs() {
  // Render dashboard list (recent 5)
  const dashBody = document.querySelector("#dashboard-recent-table tbody");
  dashBody.innerHTML = "";
  
  if (historyLog.length > 0) {
    historyLog.slice(0, 5).forEach(item => {
      const confClass = item.confidence > 0.90 ? "conf-high" : "conf-mid";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${item.time}</td>
        <td style="font-weight:700;color:#4F8EF7;">${item.letter}</td>
        <td>${item.hand}</td>
        <td><span class="conf-pill ${confClass}">${(item.confidence * 100).toFixed(1)}%</span></td>
      `;
      dashBody.appendChild(tr);
    });
  } else {
    dashBody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--text-muted);">No activity logs yet.</td></tr>`;
  }

  // Render full history page table
  const histBody = document.querySelector("#history-table tbody");
  const histEmpty = document.getElementById("history-empty-placeholder");
  histBody.innerHTML = "";

  if (historyLog.length > 0) {
    histEmpty.style.display = "none";
    document.getElementById("history-table").style.display = "table";
    
    historyLog.forEach(item => {
      const confClass = item.confidence > 0.90 ? "conf-high" : "conf-mid";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${item.time}</td>
        <td style="font-weight:700;color:#4F8EF7;">${item.letter}</td>
        <td>${item.hand}</td>
        <td><span class="conf-pill ${confClass}">${(item.confidence * 100).toFixed(1)}%</span></td>
      `;
      histBody.appendChild(tr);
    });
  } else {
    document.getElementById("history-table").style.display = "none";
    histEmpty.style.display = "flex";
  }

  // Update total translated stat on dashboard
  document.getElementById("stat-total-translated").textContent = 1482 + historyLog.length;
}

// ── REFERENCE PAGES RENDERING ─────────────────────────────────────────────────
function renderASLDictionary() {
  const container = document.getElementById("dict-grid-container");
  container.innerHTML = "";
  
  for (const [letter, [emoji, desc]] of Object.entries(ASL_DICTIONARY)) {
    const card = document.createElement("div");
    card.className = "dict-card";
    card.innerHTML = `
      <div class="dict-letter">${letter}</div>
      <div class="dict-emoji">${emoji}</div>
      <div class="dict-desc">${desc}</div>
    `;
    container.appendChild(card);
  }
}

function renderLandmarkMap() {
  const container = document.getElementById("landmark-list-container");
  container.innerHTML = "";

  LANDMARKS_REF.forEach((name, idx) => {
    const item = document.createElement("div");
    item.className = "landmark-item";
    item.innerHTML = `
      <span class="landmark-idx">${idx}</span>
      <span>${name}</span>
    `;
    container.appendChild(item);
  });
}

// ── PROJECT CONTRIBUTORS RENDERING ───────────────────────────────────────────
const CONTRIBUTORS = [
  {
    name: "Dipta Kishan Dutta",
    projects: ["Image Acquisition from Camera", "Frontend Development & Integration"],
    photo: "Photos/Dipta Kishan Dutta.jpeg"
  },
  {
    name: "Priyam Chhetri",
    projects: ["Preprocessed Dataset Preparation", "Hand Detection & Tracking"],
    photo: "Photos/Priyam Chhetri.jpeg"
  },
  {
    name: "SK Nayeemur Rahman",
    projects: ["Backend Server Development", "Text-to-Speech Integration"],
    photo: "Photos/SK Nayeemur Rahman.jpeg"
  },
  {
    name: "Souvik Bandopadhyaya",
    projects: ["Backend Server Development", "Backend Integration & Deployment Support"],
    photo: "Photos/Souvik Bandopadhyaya.jpeg"
  },
  {
    name: "Sagnik Chakraborty",
    projects: ["Noise Reduction", "Image Enhancement Support"],
    photo: "Photos/Sagnik Chakraborty.jpeg"
  },
  {
    name: "Soumyajit Dey",
    projects: ["Image Preprocessing", "Noise Reduction"],
    photo: "Photos/Soumyajit Dey.jpeg"
  },
  {
    name: "Surya Hati",
    projects: ["Preprocessed Dataset Preparation Support", "Dataset Validation Support"],
    photo: "Photos/Surya Hati.jpeg"
  },
  {
    name: "Swambrata Dey",
    projects: ["Preprocessed Dataset Preparation Support", "Dataset Organization Support"],
    photo: "Photos/Swambrata Dey.jpeg"
  },
  {
    name: "Swagata Bhunia",
    projects: ["Webpage Design & UI Planning", "Frontend Workflow Documentation"],
    photo: "Photos/Swagata Bhunia.jpeg"
  },
  {
    name: "Toufik Ali",
    projects: ["Report Documentation", "Project Documentation Support"],
    photo: "Photos/Toufik Ali.jpeg"
  },
  {
    name: "Sayan Mondal",
    projects: ["Presentation Design & Compilation", "Project Report Preparation", "Slide Layout & Formatting"],
    photo: "Photos/Sayan Mondal.jpeg"
  }
];

function renderContributors() {
  const container = document.getElementById("contributors-grid-container");
  if (!container) return;
  container.innerHTML = "";
  
  CONTRIBUTORS.forEach(c => {
    const card = document.createElement("div");
    card.className = "contributor-card";
    
    const img = document.createElement("img");
    img.className = "contributor-avatar";
    img.src = c.photo;
    img.alt = c.name;
    
    // Auto-healing fallback path loader
    let attempt = 0;
    img.onerror = () => {
      attempt++;
      if (attempt === 1) {
        // Try Photos/Photos/Name.jpeg (incase zip created nested folder)
        img.src = c.photo.replace("Photos/", "Photos/Photos/");
      } else if (attempt === 2) {
        // Try .jpg in Photos/Photos/
        img.src = c.photo.replace("Photos/", "Photos/Photos/").replace(".jpeg", ".jpg");
      } else if (attempt === 3) {
        // Try .jpg in root Photos/
        img.src = c.photo.replace(".jpeg", ".jpg");
      } else if (attempt === 4) {
        // Try .png in Photos/Photos/
        img.src = c.photo.replace("Photos/", "Photos/Photos/").replace(".jpeg", ".png");
      } else if (attempt === 5) {
        // Try .png in root Photos/
        img.src = c.photo.replace(".jpeg", ".png");
      } else {
        // Ultimate fallback to default avatar
        img.src = "https://cdn-icons-png.flaticon.com/512/149/149071.png";
      }
    };
    
    const nameEl = document.createElement("div");
    nameEl.className = "contributor-name";
    nameEl.textContent = c.name;
    
    const projectsList = document.createElement("ul");
    projectsList.className = "contributor-projects";
    c.projects.forEach(p => {
      const li = document.createElement("li");
      li.textContent = p;
      projectsList.appendChild(li);
    });
    
    card.appendChild(img);
    card.appendChild(nameEl);
    card.appendChild(projectsList);
    container.appendChild(card);
  });
}
