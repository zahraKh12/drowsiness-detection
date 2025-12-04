import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# 1) مسار البيانات
# ----------------------------------------------------
DATASET_PATH = r"C:\Users\Asus\tzst vscode\dataset\train"

CLASSES = ["Open", "Closed", "yawn", "no_yawn"]
IMG_SIZE = 64

# ----------------------------------------------------
# 2) تحميل الصور
# ----------------------------------------------------
X = []
y = []

print("➡️ Loading images...")

for label, folder in enumerate(CLASSES):
    folder_path = os.path.join(DATASET_PATH, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img)
        y.append(label)

X = np.array(X) / 255.0
y = np.array(y)

print("✔️ Images loaded:", X.shape)
print("✔️ Labels loaded:", y.shape)

# ----------------------------------------------------
# 3) تقسيم البيانات
# ----------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, num_classes=len(CLASSES))
y_val   = to_categorical(y_val,   num_classes=len(CLASSES))

# ----------------------------------------------------
# 4) بناء نموذج CNN
# ----------------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------------------------------------------
# 5) التدريب
# ----------------------------------------------------
print("➡️ Training model...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# ----------------------------------------------------
# 6) الحفظ
# ----------------------------------------------------
model.save("drowsiness_model.h5")
print("✔️ Model saved as drowsiness_model.h5")
