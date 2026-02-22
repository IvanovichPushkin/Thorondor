let pc = null;
let progressInterval = null;

// Initialize everything once the DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
  initLogStream();
  startWebRTC();

  // enable buttons immediately since we use hardcoded dirs
  const startVideoBtn = document.getElementById("startB");
  const startLogBtn = document.getElementById("startLogB");
  if (startVideoBtn) startVideoBtn.disabled = false;
  if (startLogBtn) startLogBtn.disabled = false;
});

// --- WebRTC Logic ---
async function startWebRTC() {
  const statusEl = document.getElementById("webrtcStatus");
  const videoEl = document.getElementById("videoStream");
  // const fallbackEl = document.getElementById("fallbackStream");

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  pc.addTransceiver("video", { direction: "recvonly" });

  pc.ontrack = function (evt) {
    if (evt.streams && evt.streams[0]) {
      videoEl.srcObject = evt.streams[0];
      videoEl.style.display = "block";
      // fallbackEl.style.display = "none";
      statusEl.textContent = "WebRTC Active";
    }
  };

  pc.oniceconnectionstatechange = () => {
    if (pc.iceConnectionState === "failed") {
      statusEl.textContent = "WebRTC Failed - Using MJPEG";
      videoEl.style.display = "none";
      // fallbackEl.style.display = "block";
    }
  };

  try {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const resp = await fetch("/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cam_name: "cam1",
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      }),
    });

    const answer = await resp.json();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));
  } catch (e) {
    console.error("WebRTC Error:", e);
    statusEl.textContent = "WebRTC Error";
  }
}

// --- Directory and Recording Logic ---
function saveDir() {
  // video folder is fixed at project recordings/
  alert("Videos will be saved to the project recordings folder.");
  document.getElementById("startB").disabled = false;
}

function saveLogDir() {
  // logs folder is fixed at project logs/
  alert("Logs will be saved to the project logs folder.");
  document.getElementById("startLogB").disabled = false;
}

function doRec(action) {
  let isStart = action === "start";
  fetch("/" + action + "_record")
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("startB").style.display = isStart
        ? "none"
        : "block";
      document.getElementById("stopB").style.display = isStart
        ? "block"
        : "none";
      document.getElementById("status").style.display = isStart
        ? "inline"
        : "none";

      if (!isStart) handleProgress();
      else if (progressInterval) {
        clearInterval(progressInterval);
        toggleProgressUI(false);
      }
    });
}

function handleProgress() {
  document.getElementById("startB").disabled = true;
  toggleProgressUI(true);

  const inner = document.getElementById("progressBar");
  const text = document.getElementById("progressText");

  progressInterval = setInterval(() => {
    fetch("/record_progress")
      .then((r) => r.json())
      .then((p) => {
        inner.style.width = p.percent + "%";
        text.textContent = p.percent + "%";
        if (p.done) {
          clearInterval(progressInterval);
          toggleProgressUI(false);
          document.getElementById("startB").disabled = false;
          alert("Video Saved");
        }
      });
  }, 500);
}

function toggleProgressUI(show) {
  const display = show ? "block" : "none";
  document.getElementById("progress").style.display = display;
  document.getElementById("progressText").style.display = display;
  document.getElementById("saveWarning").style.display = display;
}

function doLogRec(action) {
  let isStart = action === "start";
  fetch("/" + action + "_log_record")
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("startLogB").style.display = isStart
        ? "none"
        : "block";
      document.getElementById("stopLogB").style.display = isStart
        ? "block"
        : "none";
      const logStatus = document.getElementById("logStatus");
      if (isStart) {
        logStatus.style.display = "inline";
        logStatus.style.color = "#ef4444";
        logStatus.style.fontWeight = "bold";
        logStatus.style.animation = "blinker 1s linear infinite";
      } else {
        logStatus.style.display = "none";
        pollLogSaved();
      }
    });
}

function pollLogSaved() {
  const interval = setInterval(() => {
    fetch("/log_record_status")
      .then((r) => r.json())
      .then((s) => {
        if (s.saved) {
          clearInterval(interval);
          alert("Log Saved: " + s.file);
        }
      });
  }, 500);
}

function initLogStream() {
  const logDiv = document.getElementById("log");
  const MAX_LINES = 200; // hard DOM cap — older lines pruned automatically
  const evtSource = new EventSource("/log_stream");

  // Buffer incoming SSE events and flush them in a single rAF batch.
  // Without this, each event does a DOM append + layout + scroll individually,
  // which causes visible jank when many events arrive in the same millisecond.
  let pending = [];
  let rafId = null;

  function flushPending() {
    rafId = null;
    if (!pending.length) return;

    // Build a DocumentFragment — one reflow instead of one per line
    const frag = document.createDocumentFragment();
    for (const { text, color, bold } of pending) {
      const line = document.createElement("div");
      line.textContent = text;
      if (color) line.style.color = color;
      if (bold) line.style.fontWeight = "600";
      frag.appendChild(line);
    }
    pending = [];

    logDiv.appendChild(frag);

    // Prune oldest lines so the DOM never grows past MAX_LINES.
    // Unbounded growth = layout thrash on every future append.
    while (logDiv.children.length > MAX_LINES) {
      logDiv.removeChild(logDiv.firstChild);
    }

    // Instant scroll — smooth scroll triggers layout recalc on every frame
    logDiv.scrollTop = logDiv.scrollHeight;
  }

  evtSource.onmessage = function (e) {
    const text = e.data;
    let color = null,
      bold = false;

    if (text.includes("Cheating")) {
      color = "#dc2626";
      bold = true;
    } else if (text.includes("Normal")) {
      color = "#16a34a";
    } else if (text.includes("Desk")) {
      color = "#2563eb";
    } else if (text.includes("Object")) {
      color = "#f97316";
    }

    pending.push({ text, color, bold });

    // Coalesce bursts — one DOM update per animation frame max
    if (!rafId) rafId = requestAnimationFrame(flushPending);
  };
}
