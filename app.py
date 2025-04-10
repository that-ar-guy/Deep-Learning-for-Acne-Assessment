import cv2
import numpy as np
import threading
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import ttkbootstrap as tb
from ttkbootstrap.constants import *
# Model & Class Setup
SEVERITY_COLORS = {
    "Mild Acne": (0, 255, 0),
    "Moderate Acne": (0, 255, 255),
    "Severe Acne": (0, 0, 255)
}

CLASSES = ['No Acne', 'Mild Acne', 'Moderate Acne', 'Severe Acne']
IMG_SIZE = (224, 224)

# Load face detector
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load acne classifier
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(len(CLASSES), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_acne(image):
    preprocessed = preprocess_image(image)
    predictions = model.predict(preprocessed)
    return CLASSES[np.argmax(predictions)]

def get_recommendations(age, acne_type):
    recommendations = {
        "No Acne": "Maintain a healthy skincare routine.",
        "Mild Acne": "Use gentle cleansers and moisturizers. Consider salicylic acid.",
        "Moderate Acne": "Use benzoyl peroxide or consult a dermatologist.",
        "Severe Acne": "Consult a dermatologist for prescription treatment."
    }
    if age < 18:
        age_specific = "Avoid harsh treatments. Be gentle."
    elif 18 <= age <= 25:
        age_specific = "Balance oil control and hydration."
    else:
        age_specific = "Combine acne and anti-aging care."
    return f"{recommendations[acne_type]} {age_specific}"

def get_relative_regions(face_width, face_height):
    return {
        "forehead": (int(0.2 * face_width), int(0.05 * face_height), int(0.6 * face_width), int(0.2 * face_height)),
        "chin": (int(0.2 * face_width), int(0.75 * face_height), int(0.6 * face_width), int(0.2 * face_height)),
        "cheek_left": (int(0.05 * face_width), int(0.35 * face_height), int(0.3 * face_width), int(0.3 * face_height)),
        "cheek_right": (int(0.65 * face_width), int(0.35 * face_height), int(0.3 * face_width), int(0.3 * face_height)),
        "nose": (int(0.4 * face_width), int(0.35 * face_height), int(0.2 * face_width), int(0.3 * face_height)),
    }

# GUI Setup
class AcneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧴 Smart Acne Detector")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        self.cap = None
        self.age = 0
        self.running = False
        self.root.attributes('-fullscreen', True)
        # Theming
        style = tb.Style("cosmo")  # try 'darkly', 'flatly', 'superhero', etc.

        # Title Section
        title_frame = Frame(root, bg="#f0f0f0")
        title_frame.pack(fill=X, pady=10)

        Label(title_frame, text="🤖 Acne Detection System", font=("Helvetica", 22, "bold"), bg="#f0f0f0").pack()

        # Input Section
        input_frame = tb.Frame(root)
        input_frame.pack(pady=10)

        tb.Label(input_frame, text="Enter your age:", font=("Helvetica", 12)).pack(side=LEFT, padx=5)
        self.age_entry = tb.Entry(input_frame, width=10, bootstyle=PRIMARY)
        self.age_entry.pack(side=LEFT, padx=5)

        self.start_button = tb.Button(input_frame, text="▶ Start Detection", bootstyle=SUCCESS, command=self.start_detection)
        self.start_button.pack(side=LEFT, padx=10)

        self.quit_button = tb.Button(input_frame, text="⏹ Quit", bootstyle=DANGER, command=self.close_app)
        self.quit_button.pack(side=LEFT, padx=5)

        # Webcam Frame
        self.canvas = Label(root, bg="#000")
        self.canvas.pack(pady=10)

        # Recommendation Box
        rec_frame = tb.LabelFrame(root, text="📝 Recommendation", bootstyle=INFO)
        rec_frame.pack(fill=X, padx=20, pady=10)

        self.recommend_label = tb.Label(rec_frame, text="No recommendation yet.", wraplength=700, justify=LEFT, font=("Helvetica", 11))
        self.recommend_label.pack(padx=10, pady=10)

    def start_detection(self):
        try:
            self.age = int(self.age_entry.get())
        except ValueError:
            tb.messagebox.showerror("Invalid Age", "Please enter a valid number.")
            return

        self.cap = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self.process_video).start()

    def process_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                face_h, face_w = face_roi.shape[:2]
                REGIONS = get_relative_regions(face_w, face_h)

                for region, (rx, ry, rw, rh) in REGIONS.items():
                    if ry + rh > face_h or rx + rw > face_w:
                        continue

                    roi = face_roi[ry:ry + rh, rx:rx + rw]
                    if roi.size == 0:
                        continue

                    prediction = predict_acne(roi)
                    color = SEVERITY_COLORS.get(prediction, (255, 255, 255))

                    cv2.rectangle(frame, (x+rx, y+ry), (x+rx+rw, y+ry+rh), color, 2)
                    cv2.putText(frame, f"{region}: {prediction}", (x+rx, y+ry-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if region == "forehead":
                        recommendation = get_recommendations(self.age, prediction)
                        self.recommend_label.config(text=recommendation)

            # Convert to Tkinter image
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img = img.resize((720, 480))  # Resize for better fit
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

        self.cap.release()

    def close_app(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tb.Window(themename="cosmo")  # try superhero, darkly, minty, etc.
    app = AcneApp(root)
    root.mainloop()