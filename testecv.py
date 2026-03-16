import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
from ultralytics import YOLO

VIDEO_SOURCE = "videstrada.mp4"

FRAME_WIDTH = 1700
FRAME_HEIGHT = 870

MIN_AREA = 1000
MAX_DISTANCE = 100
MAX_DISAPPEARED = 5
MERGE_DISTANCE = 200

LINE_Y = 180
LINE1_Y = 200
LINE2_Y = 500

DISTANCIA_METROS = 8.0

cls_model = YOLO("yolo11m-cls.pt")

classified_ids = {}

vehicle_keywords = {
"minivan",
"sports_car",
"police_van",
"minibus",
"garbage_truck",
"racer",
"jeep",
"convertible",
"cab",
"beach_wagon",
"pickup",
"ambulance",
"moped",
"go-kart",
"passenger_car",
"limousine",
"moving_van",
"trailer_truck",
"tow_truck",
"mountain_bike"
}

def classify_object(frame, bbox):

    x, y, w, h = bbox
    padding = 10

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return [("desconhecido", 0.0)]

    results = cls_model(crop, verbose=False)
    probs = results[0].probs

    top3_idx = probs.top5[:3]
    top3_conf = probs.top5conf[:3]

    labels = [results[0].names[i] for i in top3_idx]
    confs = [float(c) for c in top3_conf]

    return list(zip(labels, confs))


cv2.setUseOptimized(True)

count_in = 0
count_out = 0

last_positions = {}
counted_ids = set()

measurements = {}
last_radar_text = None


class CentroidTracker:

    def __init__(self):

        self.next_id = 0
        self.objects = {}
        self.boxes = {}
        self.disappeared = {}

    def register(self, centroid, bbox):

        oid = self.next_id

        self.objects[oid] = centroid
        self.boxes[oid] = bbox
        self.disappeared[oid] = 0

        self.next_id += 1

    def deregister(self, oid):

        del self.objects[oid]
        del self.boxes[oid]
        del self.disappeared[oid]

    def update(self, detections):

        if not detections:

            for oid in list(self.disappeared.keys()):

                self.disappeared[oid] += 1

                if self.disappeared[oid] > MAX_DISAPPEARED:
                    self.deregister(oid)

            return self.objects

        input_centroids = np.array([d[0] for d in detections])

        if not self.objects:

            for c, b in detections:
                self.register(c, b)

            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        D = dist.cdist(object_centroids, input_centroids)

        used_rows = set()
        used_cols = set()

        for row in range(D.shape[0]):

            col = D[row].argmin()

            if row in used_rows or col in used_cols:
                continue

            if D[row, col] < MAX_DISTANCE:

                oid = object_ids[row]

                self.objects[oid] = input_centroids[col]
                self.boxes[oid] = detections[col][1]
                self.disappeared[oid] = 0

                used_rows.add(row)
                used_cols.add(col)

        for row, oid in enumerate(object_ids):

            if row not in used_rows:

                self.disappeared[oid] += 1

                if self.disappeared[oid] > MAX_DISAPPEARED:
                    self.deregister(oid)

        for col in range(len(detections)):

            if col not in used_cols:
                self.register(detections[col][0], detections[col][1])

        return self.objects

def merge_boxes(boxes, thresh):

    merged = []

    for x, y, w, h in boxes:

        cx = x + w // 2
        cy = y + h // 2

        merged_flag = False

        for i in range(len(merged)):

            mx, my, mw, mh = merged[i]

            mcx = mx + mw // 2
            mcy = my + mh // 2

            if abs(cx - mcx) < thresh and abs(cy - mcy) < thresh:

                nx = min(x, mx)
                ny = min(y, my)

                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny

                merged[i] = (nx, ny, nw, nh)

                merged_flag = True
                break

        if not merged_flag:
            merged.append((x, y, w, h))

    return merged


cap = cv2.VideoCapture(VIDEO_SOURCE)

bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

tracker = CentroidTracker()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = bg.apply(gray)

    fgmask = cv2.medianBlur(fgmask, 5)

    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) >= MIN_AREA]

    merged_boxes = merge_boxes(raw_boxes, MERGE_DISTANCE)

    detections = [((x + w // 2, y + h // 2), (x, y, w, h)) for (x, y, w, h) in merged_boxes]

    objects = tracker.update(detections)

    cv2.line(frame, (0, LINE_Y), (FRAME_WIDTH, LINE_Y), (255, 0, 0), 2)
    cv2.line(frame, (0, LINE1_Y), (FRAME_WIDTH, LINE1_Y), (0, 255, 255), 2)
    cv2.line(frame, (0, LINE2_Y), (FRAME_WIDTH, LINE2_Y), (255, 255, 0), 2)

    for oid, (cx, cy) in objects.items():

        x, y, w, h = tracker.boxes[oid]


        prev_y = last_positions.get(oid, cy)
        now = time.time()

        if oid not in classified_ids:
            # CLASSIFICAÇÃO PARA QUEM ENTRA
            if prev_y < LINE1_Y <= cy:
                classified_ids[oid] = classify_object(frame, (x, y, w, h))
            # CLASSIFICAÇÃO PARA QUEM SAI
            elif prev_y > LINE2_Y >= cy:
                classified_ids[oid] = classify_object(frame, (x, y, w, h))

        classes = classified_ids.get(oid, [("...", 0.0)])

        labels = [l for l, _ in classes]
        is_vehicle = any(label in vehicle_keywords for label in labels)

        texto_classes = "veiculo" if is_vehicle else "nao veiculo"

        if oid not in counted_ids:

            if prev_y < LINE_Y <= cy:
                count_in += 1
                counted_ids.add(oid)

            elif prev_y > LINE_Y >= cy:
                count_out += 1
                counted_ids.add(oid)

        crossed_line = None

        if prev_y < LINE1_Y <= cy or prev_y > LINE1_Y >= cy:
            crossed_line = LINE1_Y

        elif prev_y < LINE2_Y <= cy or prev_y > LINE2_Y >= cy:
            crossed_line = LINE2_Y

        if crossed_line is not None:

            if oid not in measurements:

                direction = "ENTRANDO" if cy > prev_y else "SAINDO"

                measurements[oid] = {
                    "time": now,
                    "line": crossed_line,
                    "direction": direction
                }

            else:

                first = measurements[oid]

                if crossed_line != first["line"]:

                    dt = now - first["time"]

                    if dt > 0:

                        velocidade = (DISTANCIA_METROS / dt) * 3.6

                        last_radar_text = (
                            f"[RADAR] ID {oid} | {first['direction']} | "
                            f"{velocidade:.2f} km/h | {texto_classes}"
                        )

                        print(last_radar_text)

                    del measurements[oid]

        last_positions[oid] = cy

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame,
                    f"ID {oid} | {texto_classes}",
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
cv2.destroyAllWindows()