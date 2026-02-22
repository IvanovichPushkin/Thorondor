let currentCam = document.body.dataset.defaultCam || "cam1";
let pc = null;
let progressInterval = null;

// --- WebRTC ---
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
  });

  pc.addTransceiver("video", { direction: "recvonly" });

  pc.ontrack = function (evt) {
    if (evt.streams && evt.streams[0]) {
      videoEl.srcObject = evt.streams[0];
      statusEl.textContent = "WebRTC Active";
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

window.addEventListener("load", () => startWebRTC(currentCam));

// --- Recording ---

// function saveDir() {
//   fetch("/set_dir", { method: "POST" })
//     .then((r) => r.json())
//     .then((data) => {
//       if (data.path) {
//         alert("Saving videos to: " + data.path);
//         document.getElementById("startB").disabled = false;
//       }
//     });
// }

// function saveLogDir() {
//   fetch("/set_log_dir", { method: "POST" })
//     .then((r) => r.json())
//     .then((data) => {
//       if (data.path) {
//         alert("Saving logs to: " + data.path);
//         document.getElementById("startLogB").disabled = false;
//       }
//     });
// }

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

// function doRec(action) {
//   let isStart = action === "start";
//   fetch("/" + action + "_record")
//     .then((response) => {
//       if (!response.ok) {
//         alert("Please click SET VIDEO DIR first!");
//         return;
//       }
//       return response.json();
//     })
//     .then((data) => {
//       if (!data) return;
//       document.getElementById("startB").style.display = isStart
//         ? "none"
//         : "block";
//       document.getElementById("stopB").style.display = isStart
//         ? "block"
//         : "none";
//       document.getElementById("status").style.display = isStart
//         ? "inline"
//         : "none";

//       if (!isStart) handleProgress();
//       else if (progressInterval) {
//         clearInterval(progressInterval);
//         toggleProgressUI(false);
//       }
//     });
// }

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

// function doLogRec(action) {
//   let isStart = action === "start";
//   fetch("/" + action + "_log_record")
//     .then((response) => {
//       if (!response.ok) {
//         alert("Please click SET LOG DIR first!");
//         return;
//       }
//       return response.json();
//     })
//     .then((data) => {
//       if (!data) return;
//       document.getElementById("startLogB").style.display = isStart
//         ? "none"
//         : "block";
//       document.getElementById("stopLogB").style.display = isStart
//         ? "block"
//         : "none";
//       alert(isStart ? "Log recording started!" : "Log saved!");
//     });
// }

// --- Live Log Stream ---
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

document.addEventListener("DOMContentLoaded", () => {
  initLogStream();
  startWebRTC();

  // enable buttons immediately since we use hardcoded dirs
  const startVideoBtn = document.getElementById("startB");
  const startLogBtn = document.getElementById("startLogB");
  if (startVideoBtn) startVideoBtn.disabled = false;
  if (startLogBtn) startLogBtn.disabled = false;
});
