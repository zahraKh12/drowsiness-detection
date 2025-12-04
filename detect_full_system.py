import cv2
import numpy as np
import mediapipe as mp
import time
import os
from tensorflow.keras.models import load_model
from playsound import playsound
import threading

# =========================
# تحميل النموذج
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "drowsiness_model.h5")
model = load_model(model_path)
print("✔ النموذج تم تحميله بنجاح!")

classes = ["awake", "drowsy", "yawn", "other"]
IMG_HEIGHT, IMG_WIDTH = model.input_shape[1], model.input_shape[2]

# =========================
# دالة تجهيز الصورة للتنبؤ
# =========================
def preprocess_eye(img):
    try:
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except:
        return None

# =========================
# دالة إطلاق الإنذار
# =========================
alarm_sound_path = os.path.join(current_dir, "alarm.wav")
def play_alarm():
    playsound(alarm_sound_path)

# =========================
# إعداد Mediapipe
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# =========================
# فتح الكاميرا
# =========================
cap = cv2.VideoCapture(0)

sleep_start_time = None
DROWSY_THRESHOLD = 2.0  # ثواني
EAR_THRESHOLD = 0.2     # أقل من هذا يعتبر العين مغلقة

# =========================
# دالة لحساب EAR (Eye Aspect Ratio)
# =========================
def eye_aspect_ratio(eye_landmarks, w, h):
    # eye_landmarks: قائمة إحداثيات 6 نقاط العين
    p = [(int(lm.x * w), int(lm.y * h)) for lm in eye_landmarks]
    # المسافة الرأسية بين النقاط (y)
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    # المسافة الأفقية (x)
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# مؤشرات Mediapipe للعين
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]  # 6 نقاط للعين اليسرى
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # 6 نقاط للعين اليمنى

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape

    status = "Unknown"
    color = (255, 255, 255)
    eye_closed = False

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]

        # حساب EAR لكل عين
        left_ear = eye_aspect_ratio([face_landmarks.landmark[i] for i in LEFT_EYE_IDX], w, h)
        right_ear = eye_aspect_ratio([face_landmarks.landmark[i] for i in RIGHT_EYE_IDX], w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            eye_closed = True
        else:
            eye_closed = False
            sleep_start_time = None

        # استخراج العينين للـ CNN
        def crop_eye(eye_points):
            x1 = min([int(face_landmarks.landmark[i].x * w) for i in eye_points])
            y1 = min([int(face_landmarks.landmark[i].y * h) for i in eye_points])
            x2 = max([int(face_landmarks.landmark[i].x * w) for i in eye_points])
            y2 = max([int(face_landmarks.landmark[i].y * h) for i in eye_points])
            return frame[y1:y2, x1:x2]

        left_eye_img = crop_eye(LEFT_EYE_IDX)
        right_eye_img = crop_eye(RIGHT_EYE_IDX)
        eye_states = []

        for eye_img in [left_eye_img, right_eye_img]:
            processed = preprocess_eye(eye_img)
            if processed is not None:
                pred = model.predict(processed, verbose=0)
                label_index = np.argmax(pred)
                label_name = classes[label_index]
                eye_states.append(label_name)

        # دمج نتائج CNN مع EAR
        if eye_closed or ("drowsy" in eye_states):
            status = "Drowsy"
            color = (0, 0, 255)
            if sleep_start_time is None:
                sleep_start_time = time.time()
        else:
            status = "Awake"
            color = (0, 255, 0)
            sleep_start_time = None

    # مدة الغلق
    sleep_duration = 0
    if sleep_start_time:
        sleep_duration = time.time() - sleep_start_time

    cv2.putText(frame, f"Status: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    if sleep_duration > 0:
        cv2.putText(frame, f"Closed: {sleep_duration:.1f}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if sleep_duration >= DROWSY_THRESHOLD:
        cv2.putText(frame, "⚠ DROWSY! WAKE UP!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        threading.Thread(target=play_alarm).start()

    cv2.imshow("Drowsiness Detection with EAR + CNN", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
