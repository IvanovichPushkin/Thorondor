let currentCam = document.body.dataset.defaultCam || "Camera 1";
let progressInterval = null;

// ─── MJPEG stream ───────────────────────────────────────────────────────────
function startStream(camName) {
  const img = document.getElementById("videoStream");
  // Append timestamp to bust any browser cache
  img.src = "/video_feed/" + encodeURIComponent(camName) + "?t=" + Date.now();
}

function switchCam(camName) {
  currentCam = camName;
  startStream(camName);
  document.querySelectorAll(".cam-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.textContent.trim() === camName);
  });
}

// ─── Recording ───────────────────────────────────────────────────────────────
function doRec(action) {
  let isStart = action === "start";
  fetch("/" + action + "_record")
    .then((r) => r.json())
    .then(() => {
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
          alert("Videos Saved!");
        }
      });
  }, 500);
}

function toggleProgressUI(show) {
  const d = show ? "block" : "none";
  document.getElementById("progress").style.display = d;
  document.getElementById("progressText").style.display = d;
  document.getElementById("saveWarning").style.display = d;
}

// ─── Log recording ───────────────────────────────────────────────────────────
function doLogRec(action) {
  let isStart = action === "start";
  fetch("/" + action + "_log_record")
    .then((r) => r.json())
    .then(() => {
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

// ─── Live log stream ─────────────────────────────────────────────────────────
function initLogStream() {
  const logDiv = document.getElementById("log");
  const MAX_LINES = 200;
  const evtSource = new EventSource("/log_stream");

  let pending = [];
  let rafId = null;

  function flushPending() {
    rafId = null;
    if (!pending.length) return;
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
    while (logDiv.children.length > MAX_LINES)
      logDiv.removeChild(logDiv.firstChild);
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
    } else if (text.includes("Object")) {
      color = "#f97316";
    } else if (text.includes("Desk")) {
      color = "#2563eb";
    }
    pending.push({ text, color, bold });
    if (!rafId) rafId = requestAnimationFrame(flushPending);
  };
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initLogStream();
  startStream(currentCam);

  // Highlight default cam button
  document.querySelectorAll(".cam-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.textContent.trim() === currentCam);
  });

  document.getElementById("startB").disabled = false;
  document.getElementById("startLogB").disabled = false;
});
