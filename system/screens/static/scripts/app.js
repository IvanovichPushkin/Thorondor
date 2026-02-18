let currentCam = "{{default_cam}}";
let pc = null;

// -----------------------------------------------
// WebRTC
// -----------------------------------------------
async function startWebRTC(camName) {
  // Close any existing connection
  if (pc) {
    pc.close();
    pc = null;
  }

  document.getElementById("webrtcStatus").textContent = "⏳ connecting...";

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  // Add a transceiver so the server knows we want video
  pc.addTransceiver("video", { direction: "recvonly" });

  pc.ontrack = function (evt) {
    const videoEl = document.getElementById("videoStream");
    if (evt.streams && evt.streams[0]) {
      videoEl.srcObject = evt.streams[0];
    }
  };

  pc.oniceconnectionstatechange = function () {
    const state = pc.iceConnectionState;
    const el = document.getElementById("webrtcStatus");
    if (state === "connected" || state === "completed") {
      el.textContent = "🟢 WebRTC";
      document.getElementById("fallbackStream").style.display = "none";
    } else if (state === "failed" || state === "disconnected") {
      el.textContent = "🔴 WebRTC failed – using MJPEG";
      useFallback(camName);
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
    console.warn("WebRTC setup failed:", err);
    document.getElementById("webrtcStatus").textContent =
      "🔴 WebRTC unavailable – MJPEG";
    useFallback(camName);
  }
}

function useFallback(camName) {
  document.getElementById("videoStream").style.display = "none";
  const fb = document.getElementById("fallbackStream");
  fb.src = "/video/" + camName;
  fb.style.display = "block";
}

function switchCam(camName) {
  currentCam = camName;
  // Reset fallback
  document.getElementById("videoStream").style.display = "block";
  document.getElementById("fallbackStream").style.display = "none";
  startWebRTC(camName);
}

// Start on load
window.addEventListener("load", () => startWebRTC(currentCam));

// -----------------------------------------------
// Recording / Log controls (unchanged from before)
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

let progressInterval = null;

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
        document.getElementById("startB").disabled = true;
        let bar = document.getElementById("progress");
        let inner = document.getElementById("progressBar");
        let text = document.getElementById("progressText");
        let warning = document.getElementById("saveWarning");
        bar.style.display = "block";
        text.style.display = "block";
        warning.style.display = "block";
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
                bar.style.display = "none";
                text.style.display = "none";
                warning.style.display = "none";
                document.getElementById("startB").disabled = false;
                alert("Video Saved ✅");
              }
            });
        }, 500);
      } else if (progressInterval) {
        clearInterval(progressInterval);
        document.getElementById("progress").style.display = "none";
        document.getElementById("progressText").style.display = "none";
        document.getElementById("saveWarning").style.display = "none";
      }
    });
}

function doLogRec(action) {
  let isStart = action === "start";
  fetch("/" + action + "_log_record")
    .then((response) => {
      if (!response.ok) {
        alert("⚠️ Please click SET LOG DIR first!");
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
      alert(isStart ? "Log recording started! 📝" : "Log saved! ✅");
    });
}

var evtSource = new EventSource("/log_stream");
var logDiv = document.getElementById("log");
evtSource.onmessage = function (e) {
  logDiv.innerHTML += e.data + "<br>";
  logDiv.scrollTop = logDiv.scrollHeight;
};
