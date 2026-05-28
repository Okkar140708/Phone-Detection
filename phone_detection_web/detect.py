#!/usr/bin/env python3
import cv2
import time
import threading
from ultralytics import YOLO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket



# ---------------- CONFIG ----------------
MODEL_PATH   = '/home/okkar-aung/ros2_ws/src/phone_detection_web/best.pt'
CAMERA_INDEX = 0
CONFIDENCE   = 0.5
PORT         = 8080

IMG_SIZE     = 320       # faster inference
FRAME_SKIP   = 1        # process every 1st frame
JPEG_QUALITY = 70        # lower = faster streaming
REAL_WIDTH = 0.07           # phone width in meters (7 cm)
FOCAL_LENGTH = 1428  

# ---------------- INIT ----------------
print("Loading model...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ---------------- Trying to get the Local IP address -------------------
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()
print(f"Model loaded! Open http://{LOCAL_IP}:{PORT}")

# print(f"Model loaded! Open http://192.168.179.235:8080")


latest_frame = None
lock = threading.Lock()



# ---------------- DETECTION LOOP ----------------
def detection_loop():
    global latest_frame

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # Skip frames for speed
        if frame_count % FRAME_SKIP != 0:
            continue

        # YOLO inference (optimized)
        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            verbose=False
        )

        # Draw detections
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                label = f"{result.names.get(cls_id, cls_id)} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Count detections
        count = sum(len(r.boxes) for r in results if r.boxes)

        color = (0, 255, 0) if count > 0 else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (240, 35), (0, 0, 0), -1)
        cv2.putText(frame, f"Detections: {count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Encode JPEG (faster streaming)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        _, jpeg = cv2.imencode('.jpg', frame, encode_param)

        with lock:
            latest_frame = jpeg.tobytes()

# ---------------- HTTP SERVER ----------------
class StreamHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
            <html>
            <body style="margin:0;background:#000">
                <h3 style="color:white;text-align:center">Phone Detector Stream</h3>
                <img src="/stream" style="display:block;margin:auto;max-width:100%">
            </body>
            </html>
            """)

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header(
                'Content-type',
                'multipart/x-mixed-replace; boundary=frame'
            )
            self.end_headers()

            try:
                while True:
                    with lock:
                        frame = latest_frame

                    if frame:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')

                    time.sleep(0.03)  # prevents lag + CPU overload

            except Exception:
                pass

# ---------------- RUN ----------------
t = threading.Thread(target=detection_loop, daemon=True)
t.start()

server = ThreadingHTTPServer(('0.0.0.0', PORT), StreamHandler)

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Stopped.")
    cap.release()