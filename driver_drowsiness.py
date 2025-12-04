import cv2
import mediapipe as mp
from math import dist

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

LEFT_EYE_IDX = [33, 159, 145, 133, 153, 144]
RIGHT_EYE_IDX = [362, 386, 374, 263, 380, 373]
EAR_THRESHOLD = 0.2

def eye_aspect_ratio(landmarks, left_indices, right_indices):
    left_eye = [landmarks[i] for i in left_indices]
    right_eye = [landmarks[i] for i in right_indices]
    
    def ratio(eye):
        vert = dist((eye[1].x, eye[1].y), (eye[5].x, eye[5].y))
        hor = dist((eye[0].x, eye[0].y), (eye[3].x, eye[3].y))
        return vert / hor
    
    return (ratio(left_eye) + ratio(right_eye)) / 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )
            
            ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, RIGHT_EYE_IDX)
            
            if ear < EAR_THRESHOLD:
                cv2.putText(frame, "SLEEP ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "AWAKE", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # اضغطي 'q' للخروج
        break

cap.release()
cv2.destroyAllWindows()
