// --- Main Application Logic ---

const statusDiv = document.getElementById("status");
const interactionStatusDiv = document.getElementById("interaction-status");
const authSection = document.getElementById("auth-section");
const appSection = document.getElementById("app-section");
const sessionEndSection = document.getElementById("session-end-section");
const restartBtn = document.getElementById("restartBtn");
const micBtn = document.getElementById("micBtn");
const cameraBtn = document.getElementById("cameraBtn");
const screenBtn = document.getElementById("screenBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const videoPreview = document.getElementById("video-preview");
const videoPlaceholder = document.getElementById("video-placeholder");
const connectBtn = document.getElementById("connectBtn");
const languageSelect = document.getElementById("languageSelect");
const usageSummaryDiv = document.getElementById("usage-summary");
const chatLog = document.getElementById("chat-log");

let currentGeminiMessageDiv = null;
let currentUserMessageDiv = null;
let errorShown = false;
// Latest running token usage from the server, shown when the session ends
let latestUsage = null;

const toastContainer = document.createElement("div");
toastContainer.className = "toast-container";
document.body.appendChild(toastContainer);

// Error toasts stay until dismissed so the message can be read/copied
function showToast(message, { title = "Error", type = "error", duration = 0 } = {}) {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.setAttribute("role", type === "error" ? "alert" : "status");

  const body = document.createElement("div");
  body.className = "toast-body";
  const titleEl = document.createElement("strong");
  titleEl.textContent = title;
  const messageEl = document.createElement("p");
  messageEl.textContent = message;
  body.append(titleEl, messageEl);

  const closeBtn = document.createElement("button");
  closeBtn.className = "toast-close";
  closeBtn.setAttribute("aria-label", "Dismiss");
  closeBtn.textContent = "×";
  closeBtn.onclick = () => toast.remove();

  toast.append(body, closeBtn);
  toastContainer.appendChild(toast);
  if (duration) setTimeout(() => toast.remove(), duration);
}

const mediaHandler = new MediaHandler();
const geminiClient = new GeminiClient({
  onOpen: () => {
    statusDiv.textContent = "Connected";
    statusDiv.className = "status connected";
    authSection.classList.add("hidden");
    appSection.classList.remove("hidden");

    // Send hidden instruction to kick off the interview
    geminiClient.sendText(
      `System: The candidate has joined. Start the interview now with your opening greeting.`
    );
  },
  onMessage: (event) => {
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        handleJsonMessage(msg);
      } catch (e) {
        console.error("Parse error:", e);
      }
    } else {
      mediaHandler.playAudio(event.data);
    }
  },
  onClose: (e) => {
    console.log("WS Closed:", e);
    statusDiv.textContent = "Disconnected";
    statusDiv.className = "status disconnected";
    if (interactionStatusDiv) {
      interactionStatusDiv.className = "interaction-status hidden";
    }
    if (!errorShown && e.code !== 1000 && e.code !== 1005) {
      showToast(
        `The connection to the server closed unexpectedly (code ${e.code}).`,
        { title: "Connection lost" }
      );
    }
    errorShown = false;
    connectBtn.disabled = false;
    showSessionEnd();
  },
  onError: (e) => {
    console.error("WS Error:", e);
    statusDiv.textContent = "Connection Error";
    statusDiv.className = "status error";
  },
});

function handleJsonMessage(msg) {
  if (msg.type === "error") {
    errorShown = true;
    statusDiv.textContent = "Error";
    statusDiv.className = "status error";
    showToast(msg.error || "Unknown error", { title: msg.title || "Error" });
  } else if (msg.type === "usage") {
    latestUsage = msg;
  } else if (msg.type === "interaction_status") {
    if (interactionStatusDiv) {
      if (msg.status === "IN_PROGRESS") {
        interactionStatusDiv.textContent = "Processing...";
        interactionStatusDiv.className = "interaction-status in-progress";
      } else if (msg.status === "REQUIRES_ACTION") {
        interactionStatusDiv.textContent = "Ready";
        interactionStatusDiv.className = "interaction-status requires-action";
        currentGeminiMessageDiv = null;
        currentUserMessageDiv = null;
      }
    }
  } else if (msg.type === "interrupted") {
    mediaHandler.stopAudioPlayback();
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
  } else if (msg.type === "turn_complete") {
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
  } else if (msg.type === "user") {
    if (currentUserMessageDiv) {
      currentUserMessageDiv.textContent += msg.text;
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentUserMessageDiv = appendMessage("user", msg.text);
    }
  } else if (msg.type === "gemini") {
    if (currentGeminiMessageDiv) {
      currentGeminiMessageDiv.textContent += msg.text;
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentGeminiMessageDiv = appendMessage("gemini", msg.text);
    }
  }
}

function appendMessage(type, text) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${type}`;
  msgDiv.textContent = text;
  chatLog.appendChild(msgDiv);
  chatLog.scrollTop = chatLog.scrollHeight;
  return msgDiv;
}

// Connect Button Handler
connectBtn.onclick = async () => {
  statusDiv.textContent = "Connecting...";
  connectBtn.disabled = true;

  try {
    // Initialize audio context on user gesture
    await mediaHandler.initializeAudio();

    geminiClient.connect(languageSelect.value);
  } catch (error) {
    console.error("Connection error:", error);
    statusDiv.textContent = "Connection Failed: " + error.message;
    statusDiv.className = "status error";
    connectBtn.disabled = false;
  }
};

// UI Controls
disconnectBtn.onclick = () => {
  geminiClient.disconnect();
};

micBtn.onclick = async () => {
  if (mediaHandler.isRecording) {
    mediaHandler.stopAudio();
    micBtn.textContent = "Start Mic";
  } else {
    try {
      await mediaHandler.startAudio((data) => {
        if (geminiClient.isConnected()) {
          geminiClient.send(data);
        }
      });
      micBtn.textContent = "Stop Mic";
    } catch (e) {
      showToast(e.message || "Microphone permission denied.", { title: "Could not start audio capture" });
    }
  }
};

cameraBtn.onclick = async () => {
  if (cameraBtn.textContent === "Stop Camera") {
    mediaHandler.stopVideo(videoPreview);
    cameraBtn.textContent = "Start Camera";
    screenBtn.textContent = "Share Screen";
    videoPlaceholder.classList.remove("hidden");
  } else {
    // If another stream is active (e.g. Screen), stop it first
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      screenBtn.textContent = "Share Screen";
    }

    try {
      await mediaHandler.startVideo(videoPreview, (base64Data) => {
        if (geminiClient.isConnected()) {
          geminiClient.sendImage(base64Data);
        }
      });
      cameraBtn.textContent = "Stop Camera";
      screenBtn.textContent = "Share Screen";
      videoPlaceholder.classList.add("hidden");
    } catch (e) {
      showToast(e.message || "Camera permission denied.", { title: "Could not access camera" });
    }
  }
};

screenBtn.onclick = async () => {
  if (screenBtn.textContent === "Stop Sharing") {
    mediaHandler.stopVideo(videoPreview);
    screenBtn.textContent = "Share Screen";
    cameraBtn.textContent = "Start Camera";
    videoPlaceholder.classList.remove("hidden");
  } else {
    // If another stream is active (e.g. Camera), stop it first
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      cameraBtn.textContent = "Start Camera";
    }

    try {
      await mediaHandler.startScreen(
        videoPreview,
        (base64Data) => {
          if (geminiClient.isConnected()) {
            geminiClient.sendImage(base64Data);
          }
        },
        () => {
          // onEnded callback (e.g. user stopped sharing from browser)
          screenBtn.textContent = "Share Screen";
          videoPlaceholder.classList.remove("hidden");
        }
      );
      screenBtn.textContent = "Stop Sharing";
      cameraBtn.textContent = "Start Camera";
      videoPlaceholder.classList.add("hidden");
    } catch (e) {
      showToast(e.message || "Screen sharing was blocked.", { title: "Could not share screen" });
    }
  }
};

sendBtn.onclick = sendText;
textInput.onkeypress = (e) => {
  if (e.key === "Enter") sendText();
};

function sendText() {
  const text = textInput.value;
  if (text && geminiClient.isConnected()) {
    geminiClient.sendText(text);
    appendMessage("user", text);
    textInput.value = "";
  }
}

function resetUI() {
  authSection.classList.remove("hidden");
  appSection.classList.add("hidden");
  sessionEndSection.classList.add("hidden");

  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
  videoPlaceholder.classList.remove("hidden");

  micBtn.textContent = "Start Mic";
  cameraBtn.textContent = "Start Camera";
  screenBtn.textContent = "Share Screen";
  chatLog.innerHTML = "";
  connectBtn.disabled = false;
  latestUsage = null;
  usageSummaryDiv.classList.add("hidden");
}

function renderUsageSummary() {
  if (!latestUsage) {
    usageSummaryDiv.classList.add("hidden");
    return;
  }
  const labels = { AUDIO: "Audio", IMAGE: "Video frames", VIDEO: "Video", TEXT: "Text", THOUGHTS: "Thinking" };
  const rows = (direction, counts) =>
    Object.entries(counts)
      .map(([modality, n]) => `<tr><td>${direction} — ${labels[modality] || modality}</td><td>${n.toLocaleString()}</td></tr>`)
      .join("");

  usageSummaryDiv.innerHTML = `
    <div class="usage-cost">Estimated cost: $${latestUsage.cost_usd.toFixed(4)}</div>
    <table>
      ${rows("Input", latestUsage.input_tokens)}
      ${rows("Output", latestUsage.output_tokens)}
    </table>
    <p class="note">Token counts reported by Gemini; cost is an estimate from list prices.</p>`;
  usageSummaryDiv.classList.remove("hidden");
}

function showSessionEnd() {
  appSection.classList.add("hidden");
  sessionEndSection.classList.remove("hidden");
  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
  renderUsageSummary();
}

restartBtn.onclick = () => {
  resetUI();
};
