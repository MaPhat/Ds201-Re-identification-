import numpy as np
from .track import Track
from typing import List
from .kalman_filter import Kalman_Filter
from.linear_assignment import min_cost_matching
class Tracker:
    def __init__(self, max_age=30, n_init=3):
        self.max_age = max_age
        self.n_init = n_init

        self.kf = Kalman_Filter()
        self.tracks = []
        self.track_id = 1
    def predict(self):
        for track in self.tracks:
            track = self.kf.predict(track)

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for match in matches:
            track_idx = match[0]
            detection_id = match[1]
            self.tracks[track_idx].update(self.kf, detections[detection_id])
        
        for unmatched in unmatched_tracks:
            self.tracks[unmatched].mark_missed()

        for unmatched in unmatched_detections:
            self._inititate(detection=detections[unmatched])

    def _match(self, detections):
        matches, unmatched_tracks, unmatched_detections = min_cost_matching(self.tracks, detections=detections)

        return matches, unmatched_tracks, unmatched_detections

    def _inititate(self, detection):
        mean, covarriance = self.kf.init(detection)

        self.tracks.append(Track(mean=mean, covariance=covarriance, track_id=self.track_id))
        self.track_id += 1