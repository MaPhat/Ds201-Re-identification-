import numpy as np
import math

def IOU(bbox1, bbox2):
    """
    bbox1, bbox2: ndarray
    (x_center, y_center, aspect_ratio, height)

    -> because of the form of deepsort input, we have to convert it to type
    (x_center, y_center, width, height)
    """
    b1 = bbox1.copy()
    b2 = bbox2.copy()

    # Convert aspect ratio to width
    b1[2] = b1[2] * b1[3]
    b2[2] = b2[2] * b2[3]

    # Convert to [x1, y1, x2, y2]
    box1 = [b1[0] - b1[2]/2, b1[1] - b1[3]/2, b1[0] + b1[2]/2, b1[1] + b1[3]/2]
    box2 = [b2[0] - b2[2]/2, b2[1] - b2[3]/2, b2[0] + b2[2]/2, b2[1] + b2[3]/2]

    # Calculate intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

