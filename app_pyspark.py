import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, from_json, udf, struct, to_json
from pyspark.sql.types import StringType, BinaryType, StructType, IntegerType
import os
import base64
import json
from typing import Iterator 
import requests
from deepsort_tracker import DeepSort
import time

os.environ["PYSPARK_PYTHON"] = r"python"

tracker = {
    "cam1": DeepSort(max_age=12, camera_id="cam1"),
    "cam2": DeepSort(max_age=12, camera_id="cam2")
}


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
        if track_class == 0 and not (time_min_motor < time_recorded - info["time"] < time_max_motor):
            continue
        if track_class == 1 and not (time_min_car < time_recorded - info["time"] < time_max_car):
            continue
        for f in info["feature"]:
            dist = cosine_distance(query_feat, f)
            dists.append((track_id, dist))
    dists.sort(key=lambda x: x[1], reverse=True)

    #Nếu cosine distance lớn nhất lớn hơn ___ thì không có id nào trong cam 2 giống cam 1
    if len(dists) == 0:
        return None
    if dists[0][1] < 0.8:
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
    # print(id_count)
    return 'cam1_' + str(sorted(id_count, reverse=True)[0])

spark = SparkSession.builder \
    .appName("VideoFrameStream") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "3") \
    .config("spark.default.parallelism", "24") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.instances", "3") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

class SingleModel:
    _instance = None
    _model = None
    _model_path = 'yolov8.pt' 

    @classmethod
    def get_model(cls):
        if cls._model is None:
            print(f"Loading YOLO model from {cls._model_path} on worker process...")
            cls._model = YOLO(cls._model_path) 
        return cls._model

def decode_single_frame(image_bytes):
    if not isinstance(image_bytes, bytes):
        print(f"Lỗi: decode_single_frame nhận kiểu dữ liệu không phải bytes: {type(image_bytes)}")
        return None
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Lỗi khi giải mã frame: {e}")
        return None


@pandas_udf(StringType())
def object_detection(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    start = time.time()
    current_model = SingleModel.get_model()


    for batch_frame_bytes_series in iterator:
        frames = [decode_single_frame(fb) for fb in batch_frame_bytes_series if fb is not None]

        valid_frames = [f for f in frames if f is not None]

        if not valid_frames:
            yield pd.Series(["[]"] * len(batch_frame_bytes_series), dtype=str)
            continue


        results_batch = current_model(valid_frames) 

        final_batch_detections = ["[]"] * len(batch_frame_bytes_series)
        original_indices_of_valid_frames = [i for i, fb in enumerate(batch_frame_bytes_series) if fb is not None]

        for result_idx, result in enumerate(results_batch):
            detections = []
            if result is not None and hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = current_model.names[cls] if cls in current_model.names else f"Unknown_{cls}" # <-- Dùng current_model

                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": class_name
                    })
            
            if result_idx < len(original_indices_of_valid_frames):
                original_index = original_indices_of_valid_frames[result_idx]
                final_batch_detections[original_index] = json.dumps(detections)
            else:
                print(f"Cảnh báo: Chỉ số kết quả ({result_idx}) vượt quá số lượng frame hợp lệ ban đầu.")
                

        yield pd.Series(final_batch_detections, dtype=str)
    end = time.time()

    print(f"=================Dectection for 1 mini-batch: {end-start}=================")


def convert_detection(json_string):
    parsed = json.loads(json_string)
    results = []
    for det in parsed:
        x1, y1, x2, y2 = det["box"]
        w = x2 - x1
        h = y2 - y1
        results.append(([x1, y1, w, h], det["confidence"], det["class_name"]))
    return results

@pandas_udf(StringType())
def process_batch(information_series: pd.Series)-> pd.Series:
    start = time.time()
    out = []
    track_ids = []
    for row in information_series:
        info = json.loads(row)
        camera_id = info['cam_id']
        frame_bytes = info['frame_bytes']
        frame_id = info['frame_id']
        detections = info['detections']
        
        # print(detections)

        detections = convert_detection(detections)
        np_arr = np.frombuffer(base64.b64decode(frame_bytes), np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        tracks = tracker[camera_id].update_tracks(
            detections, frame=frame,
            camera_id=camera_id, time_recorded=frame_id
        )
        for track in tracks:
            if not track.is_confirmed():
                continue
            if camera_id == 'cam1':
                ltrb = track.to_ltrb()  # [left, top, right, bottom]
                x1, y1, x2, y2 = map(int, ltrb)

                track_ids.append({
                    'bbox' : [x1,y1,x2,y2],
                    'track_id' : track.track_id
                })

            elif camera_id == 'cam2':
                feature_bank = tracker['cam1'].tracker.metric4deltrack.samples
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
                        track_ids.append({
                            'bbox' : [x1,y1,x2,y2],
                            'track_id' : track.track_reid_cam1
                        })
                else:
                    track_ids.append({
                            'bbox' : [x1,y1,x2,y2],
                            'track_id' : track.track_id
                        })

        out.append(json.dumps(track_ids))
        
    end = time.time()
    print(f"=================RE-ID for 1 sub-batch: {end - start}=================")
    return pd.Series(out)


df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "project") \
    .option("startingOffsets", "latest") \
    .load() \
    .selectExpr("CAST(key AS STRING) as key", "CAST(value AS STRING) as value")

value_schema = StructType() \
    .add("value1", StringType()) \
    .add("value2", IntegerType())

df = df\
    .withColumn("jsonData", from_json(col("value"), value_schema)) \
    .select(
        col("key"),
        col("jsonData.value1").alias("value"),
        col("jsonData.value2").alias("frame_id")
    )

decode_base64_udf = udf(lambda x: base64.b64decode(x) if x else None, BinaryType())

df = df.withColumn("value", decode_base64_udf(col("value")))

df_cam1 = df.filter(col("key") == "cam1")
df_cam2 = df.filter(col("key") == "cam2")

processed_df_cam1 = df_cam1 \
                    .withColumn("detections", object_detection(col("value"))) \
                    .withColumn("information", to_json(
                        struct(
                            col('key').alias('cam_id'),
                            col('value').alias('frame_bytes'),
                            col('frame_id'),
                            col('detections')
                        )
                    ))

stream_cam1 = processed_df_cam1 \
        .withColumn("tracks", process_batch(
            col('information')
        ))

processed_df_cam2 = df_cam2 \
                    .withColumn("detections", object_detection(col("value"))) \
                    .withColumn("information", to_json(
                        struct(
                            col('key').alias('cam_id'),
                            col('value').alias('frame_bytes'),
                            col('frame_id'),
                            col('detections')
                        )
                    ))

stream_cam2 = processed_df_cam2 \
        .withColumn("tracks", process_batch(
            col('information')
        ))

# def send_to_flask(batch_df, batch_id):
#     selected_data = batch_df.select('detections', 'value_base64', 'key').collect()
    
#     records_to_send = [row.asDict() for row in selected_data]

#     if not records_to_send:
#         print(f"Batch {batch_id}: No records to send to Flask.")
#         return

#     flask_server_url = "http://localhost:5000/receive_frame"

#     try:
#         response = requests.post(flask_server_url, json=records_to_send)
#         response.raise_for_status()
#         print(f"Batch {batch_id}: Successfully sent {len(records_to_send)} records to Flask. Response: {response.json()}")
#     except requests.exceptions.RequestException as e:
#         print(f"Batch {batch_id}: Error sending data to Flask: {e}")
#     except json.JSONDecodeError as e:
#         print(f"Batch {batch_id}: Error decoding Flask response JSON: {e}")

query1 = stream_cam1.select('tracks') \
    .writeStream \
    .outputMode("append") \
    .trigger(processingTime='3 seconds') \
    .format('console') \
    .option("truncate", False) \
    .start() \


query2 = stream_cam2.select('tracks') \
    .writeStream \
    .outputMode("append") \
    .trigger(processingTime='3 seconds') \
    .format('console') \
    .option("truncate", False) \
    .start() \

query1.awaitTermination()
query2.awaitTermination()