from kafka import KafkaConsumer
import cv2
import base64
import json
import numpy as np

# Tạo Kafka Consumer
consumer = KafkaConsumer(
    'visualization',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='visualizer-2window',
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    key_deserializer=lambda x: x.decode('utf-8') if x else None
)

print('listening...')
# Hàm xử lý ảnh + vẽ bbox
def process_and_display(key, frame_data):
    try:
        frame_id = frame_data['frame_id']
        image_bytes = frame_data['image_bytes']
        tracks = json.loads(frame_data['track_info'])

        # Giải mã ảnh
        if isinstance(image_bytes, str):
            image_bytes = base64.b64decode(image_bytes)

        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Lỗi giải mã ảnh từ {key}")
            return

        # Vẽ bounding boxes
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["track_id"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Hiển thị theo camera
        cv2.imshow(f"{key}", frame)

    except Exception as e:
        print(f"[Lỗi xử lý {key}]: {e}")

# Vòng lặp consumer
for message in consumer:
    key = message.key  # cam1 hoặc cam2
    value = message.value  # dict chứa frame_id, image_bytes, track_info

    process_and_display(key, value)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()