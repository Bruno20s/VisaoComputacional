import numpy as np
from scipy.spatial import distance as dist
from config import MAX_DISTANCE, MAX_DISAPPEARED

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