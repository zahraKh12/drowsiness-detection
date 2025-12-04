import cv2
import mediapipe as mp
import time
import math

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

# نسبة فتح الفم MAR threshold
MAR_THRESHOLD = 0.6
# نسبة Eye Aspect Ratio (EAR) threshold
EAR_THRESHOLD = 0.25

# نقاط العين والفم في Mediapipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 311, 78, 308, 13, 14]  # بعض نقاط المبدأ فقط

def euclidean_distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def eye_aspect_ratio(landmarks, eye_points):
    # 6 نقاط للعين
    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]
    ear = (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / (2 * euclidean_distance(p1, p4))
    return ear

def mouth_aspect_ratio(landmarks, mouth_points):
    # 8 نقاط للمثال
    A = euclidean_distance(landmarks[mouth_points[0]], landmarks[mouth_points[7]])
    B = euclidean_distance(landmarks[mouth_points[1]], landmarks[mouth_points[5]])
    C = euclidean_distance(landmarks[mouth_points[2]], landmarks[mouth_points[4]])
    D = euclidean_distance(landmarks[mouth_points[0]], landmarks[mouth_points[3]])
    mar = (A + B + C) / (2 * D)
    return mar

# فتح الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))
            
            # حساب EAR لكل عين
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2
            
            # حساب MAR
            mar = mouth_aspect_ratio(landmarks, MOUTH)
            
            # كشف النعاس
            if ear < EAR_THRESHOLD:
                cv2.putText(frame, "Sleepy!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # كشف التثاؤب
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "Yawning!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            # رسم النقاط
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
    
    cv2.imshow("Drowsiness + Yawning Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # اضغط ESC للخروج
        break

cap.release()
cv2.destroyAllWindows()
