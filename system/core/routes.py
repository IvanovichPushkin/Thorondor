from flask import Response, render_template, jsonify, request
import os
import time

def register_routes(app, recorder, log_recorder, generate_frames, frames, CAMERA_SOURCES, run_async, handle_offer, LOG_FILE, follow, template_name='app.html'):
    
    @app.route('/')
    def index():
        default_cam = list(CAMERA_SOURCES.keys())[0]
        return render_template(template_name, cams=CAMERA_SOURCES.keys(), default_cam=default_cam)

    @app.route('/offer', methods=['POST'])
    def offer():
        try:
            data = request.get_json()
            cam_name = data.get('cam_name', list(CAMERA_SOURCES.keys())[0])
            local_desc = run_async(handle_offer(cam_name, data['sdp'], data['type']))
            return jsonify({'sdp': local_desc.sdp, 'type': local_desc.type})
        except Exception as e:
            print(f"[ERROR] /offer failed: {e}")
            return jsonify({'error': str(e)}), 500

    # MJPEG fallback
    @app.route('/video/<cam_name>')
    def video(cam_name):
        return Response(generate_frames(cam_name, frames_override=frames, recorder=recorder),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/set_dir', methods=['POST'])
    def set_dir():
        path = recorder.set_directory_popup()
        if path:
            recorder.directory_set = True
            return jsonify({"status": "success", "path": path})
        return jsonify({"status": "cancelled"}), 200

    @app.route('/start_record')
    def start_record():
        if not hasattr(recorder, 'directory_set') or not recorder.directory_set:
            return jsonify({"status": "error", "message": "Please set directory first"}), 400
        recorder.start()
        return jsonify({"status": "Started"})

    @app.route('/stop_record')
    def stop_record():
        recorder.stop()
        return jsonify({"status": "Stop requested"})

    @app.route('/set_log_dir', methods=['POST'])
    def set_log_dir():
        path = log_recorder.set_directory_popup()
        if path:
            return jsonify({"status": "success", "path": path})
        return jsonify({"status": "cancelled"}), 200

    @app.route('/start_log_record')
    def start_log_record():
        if not log_recorder.directory_set:
            return jsonify({"status": "error", "message": "Please set log directory first"}), 400
        log_recorder.start()
        return jsonify({"status": "Started"})

    @app.route('/stop_log_record')
    def stop_log_record():
        log_recorder.stop()
        return jsonify({"status": "Stop requested"})

    @app.route('/log_stream')
    def log_stream():
        if not os.path.exists(LOG_FILE):
            open(LOG_FILE, 'w').close()
        return Response(follow(open(LOG_FILE, "r")), mimetype="text/event-stream")

    @app.route('/recorder_status')
    def recorder_status():
        return jsonify({
            "recording": recorder.recording,
            "status": recorder.status_msg,
            "file": os.path.basename(recorder.current_file) if recorder.current_file else "None",
            "path": recorder.output_dir
        })

    @app.route('/record_progress')
    def record_progress():
        return {
            "status": recorder.status_msg,
            "file": os.path.basename(recorder.current_file) if recorder.current_file else "None",
            "percent": 100 if not recorder.finalizing else 50,
            "done": not recorder.finalizing and not recorder.recording
        }