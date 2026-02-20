let currentCam = "{{default_cam}}";
let pc = null;
let progressInterval = null;

// -----------------------------------------------
// WebRTC (matches webcam.js style)
// -----------------------------------------------
async function startWebRTC(camName) {
  // Close any existing connection
  if (pc) {
    pc.close();
    pc = null;
  }

  const statusEl = document.getElementById("webrtcStatus");
  const videoEl = document.getElementById("videoStream");
  const fallbackEl = document.getElementById("fallbackStream");

  // Initial UI state
  videoEl.style.display = "none";
  fallbackEl.src = "/video/" + camName;
  fallbackEl.style.display = "block";
  statusEl.style.fontSize = "0.6em";
  statusEl.style.color = "#666";
  statusEl.style.marginLeft = "10px";
  statusEl.textContent = "connecting...";

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  pc.addTransceiver("video", { direction: "recvonly" });

  pc.ontrack = function (evt) {
    if (evt.streams && evt.streams[0]) {
      videoEl.srcObject = evt.streams[0];
      videoEl.style.display = "block";
      fallbackEl.style.display = "none";
      statusEl.textContent = "WebRTC Active";
    }
  };

  pc.oniceconnectionstatechange = function () {
    const state = pc.iceConnectionState;
    if (state === "failed" || state === "disconnected") {
      statusEl.textContent = "WebRTC Failed - Using MJPEG";
      videoEl.style.display = "none";
      fallbackEl.style.display = "block";
    }
  };

  try {
    const offer = await pc.createOffer();
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
    statusEl.textContent = "WebRTC Error";
    videoEl.style.display = "none";
    fallbackEl.style.display = "block";
  }
}

// Switch camera (multi-camera support)
function switchCam(camName) {
  currentCam = camName;
  document.getElementById("videoStream").style.display = "none";
  document.getElementById("fallbackStream").style.display = "block";
  startWebRTC(camName);
}

// Start initial camera
window.addEventListener("load", () => startWebRTC(currentCam));

// -----------------------------------------------
// Recording / Log controls (same as webcam.js)
// -----------------------------------------------
function saveDir() {
  fetch("/set_dir", { method: "POST" })
    .then((r) => r.json())
    .then((data) => {
      if (data.path) {
        alert("Saving videos to: " + data.path);
        document.getElementById("startB").disabled = false;
      }
    });
}

function saveLogDir() {
  fetch("/set_log_dir", { method: "POST" })
    .then((r) => r.json())
    .then((data) => {
      if (data.path) {
        alert("Saving logs to: " + data.path);
        document.getElementById("startLogB").disabled = false;
      }
    });
}

function doRec(action) {
  let isStart = action === "start";
  fetch("/" + action + "_record")
    .then((response) => {
      if (!response.ok) {
        alert("Please click SET VIDEO DIR first!");
        return;
      }
      return response.json();
    })
    .then((data) => {
      if (!data) return;

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
    .then((response) => {
      if (!response.ok) {
        alert("Please click SET LOG DIR first!");
        return;
      }
      return response.json();
    })
    .then((data) => {
      if (!data) return;
      document.getElementById("startLogB").style.display = isStart
        ? "none"
        : "block";
      document.getElementById("stopLogB").style.display = isStart
        ? "block"
        : "none";
      alert(isStart ? "Log recording started!" : "Log saved!");
    });
}

// -----------------------------------------------
// Live Log Stream (like webcam.js)
// -----------------------------------------------
function initLogStream() {
  const logDiv = document.getElementById("log");
  const evtSource = new EventSource("/log_stream");

  evtSource.onmessage = function (e) {
    const line = document.createElement("div");
    line.textContent = e.data;

    if (e.data.includes("Cheating")) {
      line.style.color = "#dc2626";
      line.style.fontWeight = "600";
    } else if (e.data.includes("Normal")) {
      line.style.color = "#16a34a";
    } else if (e.data.includes("Object")) {
      line.style.color = "#2563eb";
    }

    logDiv.appendChild(line);
    logDiv.scrollTo({ top: logDiv.scrollHeight, behavior: "smooth" });
  };
}

document.addEventListener("DOMContentLoaded", initLogStream);
