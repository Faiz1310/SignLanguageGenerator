import cv2
import mediapipe as mp
import pickle
import numpy as np
import winsound  # <-- 1. IMPORT WINSOUND (no pip install needed!)

# --- 1. LOAD THE TRAINED MODEL ---
model_filename = 'sign_language_model.pkl'
print(f"Loading model from {model_filename}...")
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

# --- 2. SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp.solutions.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- 3. State Tracker Variable ---
last_known_prediction = "No Hand"

print("Starting webcam... Show your signs! Press 'q' to quit.")

# --- 4. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.flip(rgb_frame, 1)

    results = hands.process(rgb_frame)

    prediction_text = "No Hand" # Default text

    # --- 5. PREDICTION LOGIC ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # --- 6. Format Data for Prediction ---
        landmarks_row = []
        for landmark in hand_landmarks.landmark:
            landmarks_row.append(landmark.x)
            landmarks_row.append(landmark.y)
            landmarks_row.append(landmark.z)
        
        data_for_prediction = np.array([landmarks_row])

        # --- 7. Make Prediction ---
        prediction = model.predict(data_for_prediction)
        prediction_text = prediction[0]

    # --- 8. BEEP LOGIC (NOW WITH WINSOUND) ---
    if prediction_text != last_known_prediction:
        if prediction_text != "No Hand":
            # Play a 1000 Hz beep for 200 milliseconds
            winsound.Beep(1000, 200)  # <-- 2. REPLACED BEEPY
        
        last_known_prediction = prediction_text

    # --- 9. DISPLAY PREDICTION ---
    cv2.putText(
        frame, 
        f"Prediction: {prediction_text}", 
        (10, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )

    # --- 10. DISPLAY FRAME ---
    cv2.imshow("Sign Language Predictor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 11. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Program stopped.")