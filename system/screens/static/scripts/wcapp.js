let pc = null;
let progressInterval = null;

// Initialize everything once the DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
  initLogStream();
  startWebRTC(); // Attempt WebRTC handshake
});

// --- WebRTC Logic ---
async function startWebRTC() {
  const statusEl = document.getElementById("webrtcStatus");
  const videoEl = document.getElementById("videoStream");
  const fallbackEl = document.getElementById("fallbackStream");

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  pc.addTransceiver("video", { direction: "recvonly" });

  pc.ontrack = function (evt) {
    if (evt.streams && evt.streams[0]) {
      videoEl.srcObject = evt.streams[0];
      videoEl.style.display = "block";
      fallbackEl.style.display = "none";
      statusEl.textContent = "🟢 WebRTC Active";
    }
  };

  pc.oniceconnectionstatechange = () => {
    if (pc.iceConnectionState === "failed") {
      statusEl.textContent = "🔴 WebRTC Failed - Using MJPEG";
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
        cam_name: "cam1",
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      }),
    });

    const answer = await resp.json();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));
  } catch (e) {
    console.error("WebRTC Error:", e);
    statusEl.textContent = "🔴 WebRTC Error";
  }
}

// --- Directory and Recording Logic ---
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
        alert("⚠️ Please click SET VIDEO DIR first!");
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

      if (!isStart) {
        handleProgress();
      } else if (progressInterval) {
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

function initLogStream() {
  const logDiv = document.getElementById("log");
  const evtSource = new EventSource("/log_stream");
  evtSource.onmessage = function (e) {
    logDiv.innerHTML += e.data + "<br>";
    logDiv.scrollTop = logDiv.scrollHeight;
  };
}
