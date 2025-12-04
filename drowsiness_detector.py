import cv2
from tensorflow.keras.models import load_model
import numpy as np
from playsound import playsound
import threading

# ================================
# تحميل النموذج
# ================================
model = load_model("drowsiness_model.h5")

# ================================
# تشغيل الإنذار بدون تعطيل البرنامج
# ================================
alarm_path = "alarm.wav"  # ضع ملف alarm.wav هنا في نفس الصفحة

def play_alarm():
    try:
        playsound(alarm_path)
    except:
        print("⚠️ لم أستطع تشغيل صوت الإنذار!")

# ================================
# تشغيل الكاميرا
# ================================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تجهيز الصورة
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (80, 80))
    gray_resized = gray_resized.reshape(1, 80, 80, 1) / 255.0

    # التنبؤ
    prediction = model.predict(gray_resized)
    label = "Open" if prediction[0][0] > 0.5 else "Closed"

    # إذا مغلقة → نعاس
    if label == "Closed":
        cv2.putText(frame, "Drowsy !!!", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

        # تشغيل الإنذار مرة كل مرة فقط
        threading.Thread(target=play_alarm).start()

    else:
        cv2.putText(frame, "Awake", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow("Drowsiness Detector", frame)

    # زر الخروج (ESC)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
