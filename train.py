import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Define paths
data_dir = "data"  # Folder where your images are stored
model_path = "model/keras_model.h5"  # Path to save the model
labels_path = "model/labels.txt"  # Path to save the labels

# 2. Prepare the dataset using ImageDataGenerator
image_size = (300, 300)  # Resize all images to 300x300
batch_size = 32

datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0/255.0)  # Normalize pixel values

# Load training data
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 3. Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
model.fit(train_data, validation_data=val_data, epochs=5)

# 5. Save the trained model
model.save(model_path)
print(f"Model saved at {model_path}")

# 6. Save the class labels to labels.txt
class_indices = train_data.class_indices  # {'a': 0, 'b': 1, ...}
labels = {v: k for k, v in class_indices.items()}  # Reverse dict {0: 'a', 1: 'b', ...}

with open(labels_path, 'w') as f:
    for idx in range(len(labels)):
        f.write(f"{labels[idx]}\n")

print(f"Labels saved at {labels_path}")
