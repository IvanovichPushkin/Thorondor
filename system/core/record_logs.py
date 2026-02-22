import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import threading
import queue
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from core.config import LOG_FILE

class LogRecorder:
    def __init__(self, max_queue=1000):
        self.recording = False
        self.finalizing = False
        self.saved = False
        self.output_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.filename = None
        self.log_queue = queue.Queue(maxsize=max_queue)
        self.worker = None
        self.lock = threading.Lock()
        self._stop_signal = object()
        self.directory_set = True
        self.log_entries = []

    # def set_directory_popup(self):
    #     root = tk.Tk()
    #     root.withdraw()
    #     root.attributes("-topmost", True)
    #     selected_dir = filedialog.askdirectory(title="Select Log Save Directory")
    #     root.destroy()
    #     if selected_dir:
    #         self.output_dir = os.path.normpath(selected_dir)
    #         os.makedirs(self.output_dir, exist_ok=True)
    #         self.directory_set = True
    #         return self.output_dir
    #     return None

    def start(self):
        if self.recording:
            return
        self.recording = True
        self.finalizing = False
        self.saved = False
        os.makedirs(self.output_dir, exist_ok=True)
        with self.log_queue.mutex:
            self.log_queue.queue.clear()

        # Pre-load every line already in detections.log so the PDF contains
        # the full session history, not just what happened after Record was pressed.
        self.log_entries = []
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.log_entries.append(line)
        except Exception as e:
            print(f"[WARN] Could not pre-load existing logs: {e}")

        now = datetime.now()
        friendly_name = f"Argus Report Log - {now.strftime('%b')} {now.strftime('%d')}, {now.strftime('%Y')} {now.strftime('%I-%M-%S %p')}.pdf"
        self.filename = os.path.join(self.output_dir, friendly_name)

        self.worker = threading.Thread(target=self._record_worker, daemon=True)
        self.worker.start()
        print(f"[INFO] Log recording started: {self.filename}")

    def _record_worker(self):
        start_time = datetime.now()

        while self.recording or not self.log_queue.empty():
            try:
                log_entry = self.log_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if log_entry is self._stop_signal:
                continue

            self.log_entries.append(log_entry)

        self._generate_pdf(start_time)
        self.saved = True
        self.finalizing = False
        print(f"[INFO] Log recording stopped and saved: {self.filename}")

    def _generate_pdf(self, start_time):
        doc = SimpleDocTemplate(self.filename, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        elements = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'],
            fontSize=24, textColor='darkblue', spaceAfter=30,
            alignment=TA_CENTER, fontName='Helvetica-Bold'
        )
        heading_style = ParagraphStyle(
            'CustomHeading', parent=styles['Heading2'],
            fontSize=14, textColor='darkblue', spaceAfter=12,
            spaceBefore=12, fontName='Helvetica-Bold'
        )
        body_style = ParagraphStyle(
            'CustomBody', parent=styles['BodyText'],
            fontSize=10, fontName='Courier', leftIndent=20
        )

        elements.append(Paragraph("ARGUS DETECTION REPORT", title_style))
        elements.append(Spacer(1, 0.2 * inch))

        stop_time = datetime.now()
        duration = stop_time - start_time

        metadata = f"""
        <b>Report Generated:</b> {stop_time.strftime("%B %d, %Y at %I:%M:%S %p")}<br/>
        <b>Recording Started:</b> {start_time.strftime("%B %d, %Y at %I:%M:%S %p")}<br/>
        <b>Recording Stopped:</b> {stop_time.strftime("%B %d, %Y at %I:%M:%S %p")}<br/>
        <b>Duration:</b> {str(duration).split('.')[0]}<br/>
        <b>Total Events:</b> {len(self.log_entries)}
        """

        elements.append(Paragraph(metadata, styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("DETECTION LOG", heading_style))
        elements.append(Spacer(1, 0.1 * inch))

        if self.log_entries:
            for entry in self.log_entries:
                safe_entry = entry.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Paragraph(safe_entry, body_style))
                elements.append(Spacer(1, 0.05 * inch))
        else:
            elements.append(Paragraph("No detection events recorded.", styles['Normal']))

        doc.build(elements)

    def write(self, log_message):
        if self.recording:
            try:
                self.log_queue.put_nowait(log_message)
            except queue.Full:
                pass

    def stop(self):
        if not self.recording:
            return
        self.recording = False
        self.finalizing = True
        print("[INFO] Stopping log recording...")
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=5)