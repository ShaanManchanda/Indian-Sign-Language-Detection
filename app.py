import tkinter as tk
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Load model and actions
model = load_model('action.h5')
actions = np.array(['Accident', 'Call', 'Doctor', 'Help', 'Hot', 'Lose', 'Pain', 'Thief'])

# MediaPipe utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to process image and extract results using MediaPipe
def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw styled landmarks
def draw_styled_landmarks(image, results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

# Function to extract keypoints for prediction
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])

# Tkinter App Class
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Sign Language Detection")
        self.root.geometry("800x600")

        self.label = tk.Label(self.root, text="Real-Time Indian Sign Language Detection", font=("Arial", 14))
        self.label.pack()
        
        self.canvas = tk.Label(self.root)
        self.canvas.pack()
        
        self.cap = cv2.VideoCapture(0)
        self.sequence = []
        self.threshold = 0.5
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.update_frame()
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame, results = mediapipe_detection(frame, self.holistic)
        draw_styled_landmarks(frame, results)
        
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]
        
        if len(self.sequence) == 30:
            res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            if np.max(res) > self.threshold:
                action = actions[np.argmax(res)]
                cv2.putText(frame, action, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)
        
        self.root.after(10, self.update_frame)
    
    def close_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()
