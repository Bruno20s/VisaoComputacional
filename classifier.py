from ultralytics import YOLO

cls_model = YOLO("yolo11m-cls.pt")

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