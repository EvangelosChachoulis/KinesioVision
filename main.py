import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import sqlite3 as sql
import pandas as pd
from datetime import date

MODEL_PATH = "pose_landmarker_heavy.task"
VIDEO_FILE_PATH = "BodyWeightSquats/v_BodyWeightSquats_g20_c06.avi"
UP_THRESHOLD = 160
DOWN_THRESHOLD = 90

def landmarks_detection(frame, timestamp : int, landmarker):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
    pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
    return pose_landmarker_result

def draw_landmarks_on_image(image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image = image,
        landmark_list = pose_landmarks,
        connections = vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec = pose_landmark_style,
        connection_drawing_spec = pose_connection_style)

def calculate_angle(A, B, C):
    ba = np.array(A) - np.array(B)
    bc = np.array(C) - np.array(B)
    dot_product = np.dot(ba, bc)
    ba_length = np.linalg.norm(ba)
    bc_length = np.linalg.norm(bc)
    cosine = dot_product / (ba_length * bc_length)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def extract_cords(result, w : float, h : float):
    if result.pose_landmarks:
        person = result.pose_landmarks[0]
        right_visibility = person[24].visibility + person[26].visibility + person[28].visibility
        left_visibility = person[23].visibility + person[25].visibility + person[27].visibility
        if right_visibility > left_visibility:
            hip_x = person[24].x * w
            knee_x = person[26].x * w
            ankle_x = person[28].x * w
            hip_y = person[24].y * h
            knee_y = person[26].y * h
            ankle_y = person[28].y * h
        else:
            hip_x = person[23].x * w
            knee_x = person[25].x * w
            ankle_x = person[27].x * w
            hip_y = person[23].y * h
            knee_y = person[25].y * h
            ankle_y = person[27].y * h
        return [[hip_x, hip_y],[knee_x, knee_y],[ankle_x, ankle_y]]
    else:
        print("No models detected.")
        return None

def calculate_reps_and_min(angle : float, state : str, reps : int, cur_min : float, min_list : list[float]):
    if angle <= DOWN_THRESHOLD and state == "up":
        state = "down"
        cur_min = angle
    elif state == "down":
        if angle < cur_min:
            cur_min = angle

        if angle >= UP_THRESHOLD:
            reps += 1
            state = "up"
            min_list.append(cur_min)

    return reps, state, cur_min

def db_setup():
    sql_connection = sql.connect("squats_result_database.db")
    cursor = sql_connection.cursor()
    cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS squats_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            reps_completed INTEGER,
            maximin_angle REAL,
            mean_of_minimums_angle REAL
        )
    """)
    sql_connection.commit()
    sql_connection.close()
    print("Database is ready.")

def db_insert(name : str, date : str, reps : int, min : float, avg : float):
    sql_connection = sql.connect("squats_result_database.db")
    cursor = sql_connection.cursor()
    cursor.execute("INSERT INTO squats_results (name, date, reps_completed, maximin_angle, mean_of_minimums_angle) VALUES (?, ?, ?, ?, ?)", (name, date, reps, min, avg))
    sql_connection.commit()
    sql_connection.close()
    print("Database has been updated.")

def export_db2excel():
    sql_connection = sql.connect("squats_result_database.db")
    dataframe = pd.read_sql_query("SELECT * FROM squats_results", sql_connection)
    dataframe.to_excel("squats_result.xlsx", index = False)
    sql_connection.close()

def export_db2csv():
    sql_connection = sql.connect("squats_result_database.db")
    dataframe = pd.read_sql_query("SELECT * FROM squats_results", sql_connection)
    dataframe.to_csv("squats_result.csv", index = False)
    sql_connection.close()

def calculate_db_mean_of_mean_mins():
    sql_connection = sql.connect("squats_result_database.db")
    dataframe = pd.read_sql_query("SELECT mean_of_minimums_angle FROM squats_results", sql_connection)
    sql_connection.close()
    mean_value = dataframe["mean_of_minimums_angle"].mean()
    if pd.isna(mean_value):
        return 0.0
    else:
        return float(mean_value)

def export_file():
    while True:
        file_export = str(input("Would you like to export the database into an excel/csv file (e: excel, c: csv, b: both, n: no)? ")).lower()
        if file_export == 'e':
            export_db2excel()
            print("\nExport to excel completed.")
            break
        elif file_export == 'c':
            export_db2csv()
            print("\nExport to csv completed.")
            break
        elif file_export == 'b':
            export_db2excel()
            export_db2csv()
            print("\nExport to excel/csv completed.")
            break
        elif file_export == 'n':
            pass
            break
        else:
            print("\nIncorrect input. Please try again.\n")

if __name__ == "__main__":
    db_setup()
    capture = cv2.VideoCapture(VIDEO_FILE_PATH)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH), running_mode=VisionRunningMode.VIDEO)
    name = input("Enter athlete's name: ")
    reps = 0
    state = "up"
    min_angle = DOWN_THRESHOLD
    min_angle_each_rep = []
    maximin = 0
    mean_min = 0
    date = str(date.today().strftime("%Y-%m-%d"))
    mean_meanmins = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            flag, frame = capture.read()
            timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
            if not flag:
                break
            landmarks_detection_result = landmarks_detection(frame, timestamp_ms, landmarker)
            draw_landmarks_on_image(frame, landmarks_detection_result)
            cords = extract_cords(landmarks_detection_result, width, height)
            if cords is not None:
                angle = float(calculate_angle(cords[0], cords[1], cords[2]))
                reps, state, min_angle = calculate_reps_and_min(angle, state, reps, min_angle, min_angle_each_rep)
                frame = cv2.putText(frame, str(int(angle)), (int(cords[1][0])+10, int(cords[1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                frame = cv2.putText(frame, f"Reps: {reps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("frame", frame)
            exit_key = cv2.waitKey(1) & 0xFF
            if exit_key == ord('q') or exit_key == ord(';'):
                print("Process terminated by user.")
                break
        if len(min_angle_each_rep) > 0:
            maximin = min(min_angle_each_rep)
            mean_min = float(np.mean(min_angle_each_rep))
        else:
            maximin = 0.0
            mean_min = 0.0
            print("No reps completed.")
        db_insert(name, date, reps, maximin, mean_min)
        mean_meanmins = calculate_db_mean_of_mean_mins()

        if mean_min <= mean_meanmins:
            print(f"{name}'s squat performance is in the normal range.")
        else:
            print(f"{name}'s squat performance is below the average.")
    capture.release()
    cv2.destroyAllWindows()
    export_file()