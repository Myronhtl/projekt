import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

# Daten laden
def load_data(input_folder, size=(28, 28)):
    images = []
    labels = []

    for label, letter in enumerate(sorted(os.listdir(input_folder))): 
        letter_path = os.path.join(input_folder, letter)
        if os.path.isdir(letter_path): 
            for filename in os.listdir(letter_path):
                if filename.lower().endswith(".png"):
                    img_path = os.path.join(letter_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    img_array = img_resized / 255.0
                    images.append(img_array)
                    labels.append(label)

    x_data = np.array(images).reshape(-1, 28, 28, 1).astype('float32')
    y_data = np.array(labels).astype('int')
    return x_data, y_data

x, y = load_data("..//BigDataSet")

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)
datagen.fit(x_train)

# CNN-Modell
# CNN-Modell
model = models.Sequential([

    # Eingabeschicht: Bildgröße 28x28x1 (Graustufen)
    layers.Input(shape=(28, 28, 1)),

    # Erste Convolutional-Schicht: 64 Filter, extrahiert Merkmale
    layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),  # Stabilisiert das Training
    layers.LeakyReLU(negative_slope=0.1),  # Aktivierungsfunktion
    layers.MaxPooling2D((2, 2)),  # Reduziert Bildgröße (28x28 → 14x14)

    # Zweite Convolutional-Schicht: 128 Filter, tiefere Merkmale
    layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.MaxPooling2D((2, 2)),  # Bildgröße weiter reduzieren (14x14 → 7x7)

    # Dritte Convolutional-Schicht: 256 Filter, noch tiefere Merkmale
    layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.MaxPooling2D((2, 2)),  # Bildgröße (7x7 → 3x3)
    layers.Dropout(0.3),  # Verhindert Überanpassung

    # Voll verbundene Schichten: Klassifikation
    layers.Flatten(),  # Wandelt 3D-Daten in 1D um
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # 128 Neuronen
    layers.Dropout(0.5),  # Weitere Regularisierung
    layers.Dense(64, activation='relu'),  # 64 Neuronen
    layers.Dense(26, activation='softmax')  # 26 Klassen (A-Z), Softmax für Wahrscheinlichkeiten
])


# Kompilieren
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Trainieren
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=28 ,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping])

# Evaluieren
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Speichern im neuen Format
model.save("train_model.keras")
print("Modell gespeichert als train_model.keras")