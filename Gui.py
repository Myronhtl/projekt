import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Modell laden
model = load_model("cnn_model.keras")

# Funktion zur Vorhersage eines Buchstabens basierend auf den Modellwahrscheinlichkeiten
def predict_letter(probabilities):
    index = np.argmax(probabilities)
    return chr(index + ord('A'))

# Fenster erstellen
root = tk.Tk()
root.title("Buchstabenerkennung")
root.geometry("800x800")  # Fenstergröße ändern
root.config(bg="black")

# Größere Zeichenfläche
canvas_width = 300
canvas_height = 300

# Canvas erstellen
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white', relief="solid", borderwidth=2)
canvas.pack(pady=20)

# Button für Vorhersage
button_predict = tk.Button(root, text="Erkennen", font=("Arial", 14), command=lambda: predict_drawing())
button_predict.pack(pady=10)

# Button für Löschen
button_clear = tk.Button(root, text="Löschen", font=("Arial", 14), command=lambda: clear_canvas())
button_clear.pack(pady=10)

# Label für das Ergebnis
label_result = tk.Label(root, text="Erkannt: ", font=("Arial", 16))
label_result.pack(pady=10)

# Bild erstellen, in das gezeichnet wird
image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)

# Mousebewegung auf dem Canvas binden
def paint(event):
    # Zeichenfunktion, wenn der Benutzer mit der Maus zieht
    x, y = event.x, event.y
    r = 12  # Größerer Radius des Kreises
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
    draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

# Vorhersagefunktion
def predict_drawing():
    # Binarisieren des Bildes: Alles unter einem bestimmten Schwellenwert wird auf Schwarz gesetzt, der Rest bleibt Weiß
    thresholded_image = image.point(lambda p: p < 200 and 255)  # Setzt alles unter 200 auf schwarz, alles darüber auf weiß

    # Bild auf 28x28 Pixel skalieren
    resized_image = thresholded_image.resize((28, 28), Image.Resampling.LANCZOS)

    # Konvertiere das Bild in Graustufen
    resized_image = resized_image.convert("L")

    # Bild invertieren (Weiß wird zu Schwarz und umgekehrt)
    image_array = 1 - np.array(resized_image).astype(np.float32) / 255.0  # Bild invertieren
    image_array = image_array.reshape(1, 28, 28, 1)  # Umformung auf (1, 28, 28, 1)

    # Modellvorhersage
    prediction = model.predict(image_array)
    letter = predict_letter(prediction[0])  # Vorhergesagter Buchstabe
    confidence = np.max(prediction) * 100  # Wahrscheinlichkeit der Vorhersage

    # Ergebnis anzeigen
    label_result.config(text=f"Erkannt: {letter}\nWahrscheinlichkeit: {confidence:.2f}%")

# Canvas löschen
def clear_canvas():
    # Canvas löschen
    canvas.delete("all")
    global image, draw
    image = Image.new("L", (canvas_width, canvas_height), color=255)
    draw = ImageDraw.Draw(image)

# Binde die Paint-Funktion an die Mausbewegung
canvas.bind("<B1-Motion>", paint)

# Tkinter-GUI starten
root.mainloop()

