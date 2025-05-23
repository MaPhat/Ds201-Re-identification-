from Deep_Sort import Kalman_Filter, Tracker
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO

model = YOLO(r"Ds201-Re-identification-\model\vehicle_front.pt")
list_img = [
    r'Ds201-Re-identification-\test1.png',
    r'Ds201-Re-identification-\test2.png'
]
tracker = Tracker()

for img_path in list_img:
    frame = cv2.imread(img_path)
    result = model(frame)
    boxes = result[0].boxes

    _class = (boxes.cls).cpu().numpy()
    _conf = (boxes.conf).cpu().numpy()
    _data = (boxes.xywh).cpu().numpy()
    df = pd.DataFrame(_data, columns=['x', 'y', 'w', 'h'])
    df['conf'] = _conf
    df['class'] = _class

    idx2class = {
        0 : 'xe_may',
        1 : 'o_to'
    }
    detection = []

    for _, row in df.iterrows():
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        conf = row['conf']
        clss = row['class']
        detection.append(np.array([x, y, w, h]))

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(frame, (x, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {clss}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        
    tracker.update(detections=_data)

    if (img_path == r'Ds201-Re-identification-\test2.png'):
        for track in tracker.tracks:
            # if track.is_confirmed() and track.time_since_update == 0:  # chỉ vẽ track còn sống
            x, y, w, h = track.mean[:4]
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # màu xanh lá
            cv2.putText(frame, f"ID {track.track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    print(tracker.tracks)

cv2.imshow("Tracked Vehicles", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
        

