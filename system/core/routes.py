import os
import asyncio
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse, Response

def register_routes(app, recorder, log_recorder, generate_frames, frames,
                    CAMERA_SOURCES, handle_offer, LOG_FILE, follow,
                    templates=None, template_name="app.html"):

    @app.get("/", response_class=Response)
    async def index(request: Request):
        default_cam = list(CAMERA_SOURCES.keys())[0]
        return templates.TemplateResponse(
            template_name,
            {"request": request, "cams": list(CAMERA_SOURCES.keys()), "default_cam": default_cam}
        )

    @app.post("/offer")
    async def offer(request: Request):
        try:
            data = await request.json()
            cam_name = data.get("cam_name", list(CAMERA_SOURCES.keys())[0])
            local_desc = await handle_offer(cam_name, data["sdp"], data["type"])
            return JSONResponse({"sdp": local_desc.sdp, "type": local_desc.type})
        except Exception as e:
            print(f"[ERROR] /offer failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/set_dir")
    async def set_dir():
        path = recorder.set_directory_popup()
        if path:
            recorder.directory_set = True
            return JSONResponse({"status": "success", "path": path})
        return JSONResponse({"status": "cancelled"})

    @app.get("/start_record")
    async def start_record(request: Request):
        if not getattr(recorder, "directory_set", False):
            return JSONResponse({"status": "error", "message": "Please set directory first"}, status_code=400)
        cam_name = request.query_params.get("cam_name", list(CAMERA_SOURCES.keys())[0])
        recorder.start(cam_name=cam_name)
        return JSONResponse({"status": "Started"})

    @app.get("/stop_record")
    async def stop_record():
        recorder.stop()
        return JSONResponse({"status": "Stop requested"})

    @app.get("/record_progress")
    async def record_progress():
        return JSONResponse({
            "status":  recorder.status_msg,
            "file":    os.path.basename(recorder.current_file) if recorder.current_file else "None",
            "percent": 100 if not recorder.finalizing else 50,
            "done":    not recorder.finalizing and not recorder.recording
        })

    @app.get("/recorder_status")
    async def recorder_status():
        return JSONResponse({
            "recording": recorder.recording,
            "status":    recorder.status_msg,
            "file":      os.path.basename(recorder.current_file) if recorder.current_file else "None",
            "path":      recorder.output_dir
        })

    @app.post("/set_log_dir")
    async def set_log_dir():
        path = log_recorder.set_directory_popup()
        if path:
            return JSONResponse({"status": "success", "path": path})
        return JSONResponse({"status": "cancelled"})

    @app.get("/start_log_record")
    async def start_log_record():
        if not getattr(log_recorder, "directory_set", False):
            return JSONResponse({"status": "error", "message": "Please set log directory first"}, status_code=400)
        log_recorder.start()
        return JSONResponse({"status": "Started"})

    @app.get("/stop_log_record")
    async def stop_log_record():
        log_recorder.stop()
        return JSONResponse({"status": "Stop requested"})

    @app.get("/log_stream")
    async def log_stream():
        if not os.path.exists(LOG_FILE):
            open(LOG_FILE, "w").close()

        async def event_generator():
            logfile = open(LOG_FILE, "r")
            logfile.seek(0, 2)
            try:
                while True:
                    line = logfile.readline()
                    if not line:
                        await asyncio.sleep(0.1)
                        continue
                    if log_recorder.recording:
                        log_recorder.write(line.strip())
                    yield f"data: {line}\n\n"
            finally:
                logfile.close()

        return StreamingResponse(event_generator(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})