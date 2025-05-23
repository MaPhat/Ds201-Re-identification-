import numpy as np
from .kalman_filter import Kalman_Filter

class TrackState:
    """
    Trạng thái	    Mô tả
    Tentative	    Đối tượng mới được phát hiện, chưa chắc chắn
    Confirmed	    Đã được phát hiện liên tiếp ≥ n_init lần
    Deleted	        Đã bị loại bỏ do mất dấu hoặc không đủ tin cậy
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    def __init__(self, mean, covariance, track_id, n_init=3, max_age=30, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.n_init = n_init
        self.max_age = max_age
        self.state = TrackState.Tentative

        if feature is not None:
            self.feature = feature

        self.hits = 1
        self.age = 1
        self.TimeSinceUpdate = 0

    def predict(self, kf : Kalman_Filter):
        self.mean, self.covariance = kf.predict(self.mean,self.covariance)

        self.age += 1
        self.TimeSinceUpdate += 1

    def update(self, kf: Kalman_Filter, measurement):
        """
        measurement: (xcenter, ycenter, ratio, height)
        """
        self.mean, self.covariance = kf.update(measurement=measurement, mean=self.mean, covariance=self.covariance)

        self.hits += 1
        self.TimeSinceUpdate = 0

        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.age > self.max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted 
