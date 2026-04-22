import cv2
import numpy as np
import time

from config import *
from tracker import CentroidTracker
from utils import merge_boxes
from classifier import classify_object
from embedding import Embedding, find_similar_id
from database import VectorDatabase

db = VectorDatabase()

embedder = Embedding()

classified_ids = {}
embeddings_ids = {}

# carregar embeddings do banco
dados_db = db.get_all_vectors()

for rid, oid, arquivo, data_hora, vec, classes in dados_db:
    if oid not in embeddings_ids:
        embeddings_ids[oid] = vec

classified_ids = {}
embeddings_ids = {}

id_map = {}
SIMILARITY_THRESHOLD = 0.98

count_in = 0
count_out = 0

last_positions = {}
counted_ids = set()

measurements = {}
last_radar_text = None

cv2.setUseOptimized(True)

cap = cv2.VideoCapture(VIDEO_SOURCE)

bg = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=40,
    detectShadows=False
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

tracker = CentroidTracker()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    h, w = frame.shape[:2]

    MERGE_DISTANCE = int(w * MERGE_DISTANCE_RATIO)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = bg.apply(gray)

    fgmask = cv2.medianBlur(fgmask, 5)

    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        fgmask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    raw_boxes = [
        cv2.boundingRect(cnt)
        for cnt in contours
        if cv2.contourArea(cnt) >= MIN_AREA
    ]

    merged_boxes = merge_boxes(raw_boxes, MERGE_DISTANCE)

    detections = [
        ((x + w // 2, y + h // 2), (x, y, w, h))
        for (x, y, w, h) in merged_boxes
    ]

    objects = tracker.update(detections)

    cv2.line(frame, (0, LINE_Y), (FRAME_WIDTH, LINE_Y), (255, 0, 0), 2)
    cv2.line(frame, (0, LINE1_Y), (FRAME_WIDTH, LINE1_Y), (0, 255, 0), 2)
    cv2.line(frame, (0, LINE2_Y), (FRAME_WIDTH, LINE2_Y), (0, 0, 255), 2)

    for oid, (cx, cy) in objects.items():

        x, y, w, h = tracker.boxes[oid]

        real_id = id_map.get(oid, oid)

        prev_y = last_positions.get(real_id, cy)
        now = time.time()

        if oid not in classified_ids:

            if prev_y < LINE2_Y <= cy or prev_y > LINE1_Y >= cy:

                classificacoes = classify_object(frame, (x, y, w, h))
                classified_ids[oid] = classificacoes

                h_frame, w_frame = frame.shape[:2]

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_frame, x + w)
                y2 = min(h_frame, y + h)

                crop = frame[y1:y2, x1:x2]

                if crop.size != 0:

                    embedding = embedder.get_image_embedding(crop)

                    similar_id, score = find_similar_id(
                        embedding,
                        embeddings_ids,
                        SIMILARITY_THRESHOLD
                    )

                    if similar_id is not None:
                        print(f"[MATCH] ID {oid} → {similar_id} | score={score:.2f}")
                        id_map[oid] = similar_id
                        real_id = similar_id
                    else:
                        embeddings_ids[oid] = embedding

                        db.insert_vector(
                            object_id=oid,
                            vector=embedding,
                            arquivo=VIDEO_SOURCE,
                            classificacoes=classificacoes
                        )

                        id_map[oid] = oid
                        real_id = oid

        classes = classified_ids.get(oid, [("...", 0.0)])

        labels = [l for l, _ in classes]
        is_vehicle = any(label in vehicle_keywords for label in labels)

        texto_classes = "veiculo" if is_vehicle else "nao veiculo"

        if real_id not in counted_ids:

            if prev_y < LINE_Y <= cy:
                count_in += 1
                counted_ids.add(real_id)

            elif prev_y > LINE_Y >= cy:
                count_out += 1
                counted_ids.add(real_id)

        crossed_line = None

        if prev_y < LINE1_Y <= cy or prev_y > LINE1_Y >= cy:
            crossed_line = LINE1_Y

        elif prev_y < LINE2_Y <= cy or prev_y > LINE2_Y >= cy:
            crossed_line = LINE2_Y

        if crossed_line is not None:

            if real_id not in measurements:

                direction = "ENTRANDO" if cy > prev_y else "SAINDO"

                measurements[real_id] = {
                    "time": now,
                    "line": crossed_line,
                    "direction": direction
                }

            else:

                first = measurements[real_id]

                if crossed_line != first["line"]:

                    dt = now - first["time"]

                    if dt > 0:

                        velocidade = (DISTANCIA_METROS / dt) * 3.6

                        last_radar_text = (
                            f"[RADAR] ID {real_id} | {first['direction']} | "
                            f"{velocidade:.2f} km/h | {texto_classes}"
                        )

                        print(last_radar_text)

                    del measurements[real_id]

        last_positions[real_id] = cy

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame,
                    f"ID {real_id} | {texto_classes}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2)

    cv2.putText(frame, f"Entraram: {count_in}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Sairam: {count_out}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if last_radar_text:
        cv2.putText(frame, last_radar_text, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Mask", fgmask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
db.fechar()
cv2.destroyAllWindows()