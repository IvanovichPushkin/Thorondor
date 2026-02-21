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
    });
}

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
    } else if (e.data.includes("Desk")) {
      line.style.color = "#2563eb"; // blue for desk
    } else if (e.data.includes("Object")) {
      line.style.color = "#f97316"; // orange for object
    }

    logDiv.appendChild(line);
    logDiv.scrollTo({ top: logDiv.scrollHeight, behavior: "smooth" });
  };
}
