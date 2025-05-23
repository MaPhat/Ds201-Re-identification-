import numpy as np
from scipy.optimize import linear_sum_assignment
from .IOU_matching import IOU

def min_cost_matching(tracks, detections, track_indices=None, detection_indices=None):

    """
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    """
    if track_indices == None:
        track_indices = np.arange(len(tracks))
    if detection_indices == None:
        detection_indices = np.arange(len(detections))

    if len(track_indices) == 0 & len(detection_indices) == 0:
        return [], track_indices, detection_indices
    
    cost_matrix = []

    for _, row_t in enumerate(tracks):
        cost = []
        for _, row_d in enumerate(detections):
            cost_value = IOU(row_t.mean, row_d)

            cost.append(cost_value)

        cost_matrix.append(cost)
    
    cost_matrix = 1 - np.array(cost_matrix)
    indices = np.asarray(linear_sum_assignment(cost_matrix)).T

    matches, unmatched_detections, unmatched_tracks = [], [], []
    for idx, detection_id in enumerate(detection_indices):
        if idx not in indices[:,1]:
            unmatched_detections.append(detection_id)

    for idx, track_id in enumerate(track_indices):
        if idx not in indices[:,0]:
            unmatched_tracks.append(track_id)

    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > 0.7:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections
