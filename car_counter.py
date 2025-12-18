import cv2
from ultralytics import YOLO
import math

VIDEO_SOURCE = 0 # use 0 for webcam or provide video file path like "path/para/seu/video.mp4"
MODEL_NAME = "yolov8m.pt"
CONF_THRES = 0.4

CLASS_CAR = [2, 3, 5, 7]

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO_SOURCE)

LINE_Y = 100

# tracking
tracks = {}     # id -> (cx, cy)
history = {}    # id -> lista de últimos y
next_id = 0

entrou = 0
saiu = 0

def distancia(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRES)

    detections = []  # (cx, cy, x1, y1, x2, y2)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in CLASS_CAR:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append((cx, cy, x1, y1, x2, y2))

    updated_tracks = {}

    for det in detections:
        cx, cy, x1, y1, x2, y2 = det

        #limite de associação
        min_dist = 99999
        best_id = None

        for tid, (tx, ty) in tracks.items():
            d = distancia((cx, cy), (tx, ty))
            if d < min_dist and d < 90: 
                min_dist = d
                best_id = tid

        if best_id is not None:
            updated_tracks[best_id] = (cx, cy)

            # Histórico com até 3 posições
            if best_id not in history:
                history[best_id] = []
            history[best_id].append(cy)
            if len(history[best_id]) > 3:
                history[best_id].pop(0)

            # Detecta cruzamento
            if len(history[best_id]) >= 2:
                old_y = history[best_id][-2]

                if old_y < LINE_Y <= cy:
                    entrou += 1

                # Sai (vem de baixo para cima)
                if old_y > LINE_Y >= cy:
                    saiu += 1

        else:
            # Novo veículo
            updated_tracks[next_id] = (cx, cy)
            history[next_id] = [cy]
            best_id = next_id
            next_id += 1

        # Desenha caixas e ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {best_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    tracks = updated_tracks

    # Linha de contagem: cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (255, 0, 0), 2)

    cv2.putText(frame, f"Entraram: {entrou}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.putText(frame, f"Sairam: {saiu}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()