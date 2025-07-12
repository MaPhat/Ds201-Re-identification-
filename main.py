import cv2
from ultralytics import YOLO
from deepsort_tracker import DeepSort
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import time
import torch
import random
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # nếu dùng multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

def cosine_distance(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def reid_query(track, feature_bank, top_k=5):
    query_feat = track.features
    track_class = track.det_class
    time_recorded = track.time_recorded
    
    time_min_motor = 140
    time_max_motor = 180
    time_min_car = 850
    time_max_car = 900

    dists = []
    
    for track_id, info in feature_bank.items():
        if info["used"]:
            continue
        if info["class"] != track_class:
            continue
        # if track_class == 0 and not (time_min_motor < time_recorded - info["time"] < time_max_motor):
        #     continue
        # if track_class == 1 and not (time_min_car < time_recorded - info["time"] < time_max_car):
        #     continue
        for f in info["feature"]:
            dist = cosine_distance(query_feat, f)
            dists.append((track_id, dist))
    dists.sort(key=lambda x: x[1], reverse=True)

    #Nếu cosine distance lớn nhất lớn hơn ___ thì không có id nào trong cam 2 giống cam 1
    if len(dists) == 0:
        return None
    if dists[0][1] < 0.7:
        return None
    for id, dist in dists[:top_k]:
        print(f"Id: {id}, dist: {dist}")
    list_id_match = [int(id.split("_")[1]) for id, _ in dists[:top_k]]
    id_count = {}
    for id in list_id_match:
        if id not in id_count:
            id_count[id] = 1
        else:
            id_count[id] += 1
    print(id_count)
    return 'cam1_' + str(sorted(id_count, reverse=True)[0])

def main():
    timer = time.time()
    model = YOLO(r'yolo_weight.pt')

    list_cam = [
        r'Ga_Anphu_crop.MOV',
        r'Ga_Thaodien_crop - Trim.mp4'
    ]

    tracker = {}
    for id, cam in enumerate(list_cam):
        tracker[cam] = DeepSort(max_age=12, camera_id=f"cam{id}")

    cap1 = cv2.VideoCapture(list_cam[0])
    cap2 = cv2.VideoCapture(list_cam[1])

    frame_count = 0
    while cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret2:
            break
        if not ret1:
            frame1 = np.zeros_like(frame2)
            results1 = None
        else:
            results1 = model(frame1, verbose=False)[0]
        frame_count += 1

        detections1 = []
        if results1 is not None:
            for det in results1.boxes:
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                conf = float(det.conf)
                class_id = int(det.cls) 

                if conf > 0.4:
                    detections1.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))
            tracks1 = tracker[list_cam[0]].update_tracks(detections1, frame=frame1, camera_id='cam1', time_recorded=frame_count)
            for track in tracks1:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()  # [left, top, right, bottom]
                x1, y1, x2, y2 = map(int, ltrb)

                cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame1, f'ID {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # track_db[track_id] = (track.get_feature(), track.get_det_class(), time.time())

        results2 = model(frame2, verbose=False)[0]
        detections2 = []
        for det in results2.boxes:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            conf = float(det.conf)
            class_id = int(det.cls)

            if conf > 0.4:
                detections2.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))

        tracks2 = tracker[list_cam[1]].update_tracks(detections2, frame=frame2, camera_id='cam2', time_recorded=frame_count)
        for track in tracks2:
            if not track.is_confirmed():
                continue
            _class_track = track.det_class
            print("="*10)
            print(track.track_id)
            feature_bank = tracker[list_cam[0]].tracker.metric4deltrack.samples
            max_age4_reid = 2
               
            matched_id = reid_query(track=track,feature_bank=feature_bank)
            if matched_id is not None:
                if matched_id == track.suggested_reid:
                    track.reid_consecutive += 1 
                else:
                    track.suggested_reid = matched_id
                    track.reid_consecutive = 1 
                
                if track.reid_consecutive > max_age4_reid and track.track_reid_cam1 == -1:
                    track.track_reid_cam1 = matched_id
                    feature_bank[track.track_reid_cam1]['used'] = True
            else:
                track.reid_consecutive = 0
                track.suggested_reid = None
            
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            x1, y1, x2, y2 = map(int, ltrb)

            if track.track_reid_cam1 != -1:
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame2, f'ID {track.track_reid_cam1}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame2, f'ID {track.track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame1_resized = cv2.resize(frame1, (960, 540))
        frame2_resized = cv2.resize(frame2, (960, 540))

        # Ghép frame theo chiều ngang
        combined_frame = cv2.hconcat([frame1_resized, frame2_resized])

        # Hiển thị
        cv2.imshow("Tracked Vehicles", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    print("✅ Script đã chạy đến cuối")
