let currentCam = document.body.dataset.defaultCam || "Camera 1";
let pc = null;
let progressInterval = null;

// Initialize everything once the DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
  initLogStream();
  startWebRTC(currentCam);

  // enable buttons immediately since we use hardcoded dirs
  const startVideoBtn = document.getElementById("startB");
  const startLogBtn = document.getElementById("startLogB");
  if (startVideoBtn) startVideoBtn.disabled = false;
  if (startLogBtn) startLogBtn.disabled = false;
});

// --- WebRTC Logic ---
async function startWebRTC(camName) {
  if (pc) {
    pc.close();
    pc = null;
  }

  const statusEl = document.getElementById("webrtcStatus");
  const videoEl = document.getElementById("videoStream");

  statusEl.style.fontSize = "0.6em";
  statusEl.style.color = "#666";
  statusEl.style.marginLeft = "10px";
  statusEl.textContent = "connecting...";

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    bundlePolicy: "max-bundle",
    rtcpMuxPolicy: "require",
  });

  pc.addTransceiver("video", { direction: "recvonly" });

  pc.ontrack = function (evt) {
    if (evt.streams && evt.streams[0]) {
      videoEl.srcObject = evt.streams[0];
      videoEl.style.display = "block";
      statusEl.textContent = "WebRTC Active";

      // Minimize latency — disable jitter buffer buffering
      videoEl.play();
      if (typeof videoEl.jitterBufferTarget !== "undefined") {
        videoEl.jitterBufferTarget = 0; // Chrome 113+
      }
    }
  };

  pc.oniceconnectionstatechange = function () {
    const state = pc.iceConnectionState;
    if (state === "failed" || state === "disconnected") {
      statusEl.textContent = "WebRTC Failed — retrying...";
      setTimeout(() => startWebRTC(camName), 3000);
    }
  };

  try {
    const offer = await pc.createOffer();
    // Request high bitrate in SDP — forces browser to ask for quality
    offer.sdp = offer.sdp.replace(/a=mid:video/, "a=mid:video\r\nb=AS:4000");
    await pc.setLocalDescription(offer);

    const resp = await fetch("/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cam_name: camName,
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      }),
    });

    if (!resp.ok) throw new Error("Offer rejected");

    const answer = await resp.json();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));
  } catch (err) {
    console.error("WebRTC Error:", err);
    statusEl.textContent = "WebRTC Error — retrying...";
    setTimeout(() => startWebRTC(camName), 3000);
  }
}

function switchCam(camName) {
  currentCam = camName;
  startWebRTC(camName);

  document.querySelectorAll(".cam-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.textContent.trim() === camName);
  });
}

// --- Recording ---
function doRec(action) {
  let isStart = action === "start";
  const url = isStart
    ? `/start_record?cam_name=${encodeURIComponent(currentCam)}`
    : "/stop_record";
  fetch(url)
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
  const inner = document.getElementById("progressBar");
  const text = document.getElementById("progressText");

  document.getElementById("startB").disabled = true;
  toggleProgressUI(true);
  inner.style.width = "0%";
  text.textContent = "0%";

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
      }
    });
}

// --- Live Log Stream ---
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
