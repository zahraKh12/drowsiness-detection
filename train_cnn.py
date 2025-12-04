import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ================================
# 1) مسار البيانات
# ================================
DATASET_DIR = r"C:\Users\Asus\tzst vscode\dataset\train"

# حجم الصورة
IMG_SIZE = 80
BATCH = 32

# ================================
# 2) تجهيز البيانات DataGenerator
# ================================
datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,   # 20% validation من مجلد train نفسه
    horizontal_flip=True,
    zoom_range=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="training"
)

valid_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation"
)

# معرفة الفئات
print("Classes found:", train_data.class_indices)

# ================================
# 3) بناء نموذج CNN
# ================================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.4),

    Dense(4, activation="softmax")   # 4 classes
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================================
# 4) التدريب
# ================================
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=15
)

# ================================
# 5) حفظ النموذج
# ================================
model.save("drowsiness_model.h5")
print("✔ Model Saved as drowsiness_model.h5")
