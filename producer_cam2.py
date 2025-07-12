import cv2
from kafka import KafkaProducer
import time
import base64
import json

path_to_vd = r'Ga_Thaodien_crop - Trim.mp4'
cap = cv2.VideoCapture(path_to_vd)

topic_name = 'project'
kafka_server = 'localhost:9092'

producer = KafkaProducer(bootstrap_servers=kafka_server)

print('sending video...')
i = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    _, frame = cv2.imencode('.jpg', frame)
    frame_bytes = frame.tobytes()
    frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')

    value = {
        "value1": frame_b64,
        "value2": i
    }

    # Chuyển dict thành bytes
    producer.send(topic_name, key=b'cam2', value=json.dumps(value).encode('utf-8'))
    i += 1
    time.sleep(0.5)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()