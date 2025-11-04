import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- 1. LOAD THE TRAINED MODEL ---
model_filename = 'sign_language_model.pkl'
print(f"Loading model from {model_filename}...")
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

# --- 2. SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting webcam... Show your signs! Press 'q' to quit.")

# --- 3. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.flip(rgb_frame, 1)

    results = hands.process(rgb_frame)

    prediction_text = "No Hand" # Default text

    # --- 4. PREDICTION LOGIC ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw the landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # --- 5. NEW: Format Data for Prediction ---
        # Extract landmarks into a flat list, just like our CSV
        landmarks_row = []
        for landmark in hand_landmarks.landmark:
            landmarks_row.append(landmark.x)
            landmarks_row.append(landmark.y)
            landmarks_row.append(landmark.z)
        
        # We need to reshape this list into a 2D array
        # because the model expects a "batch" of samples
        data_for_prediction = np.array([landmarks_row])

        # --- 6. NEW: Make Prediction ---
        # Use the loaded model to predict the class
        prediction = model.predict(data_for_prediction)
        
        # Get the predicted letter (it's the first item in the list)
        prediction_text = prediction[0]

    # --- 7. DISPLAY PREDICTION ---
    cv2.putText(
        frame, 
        f"Prediction: {prediction_text}", 
        (10, 50), # Position
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, # Font scale
        (0, 255, 0), # Color (Green)
        2 # Thickness
    )

    # --- 8. DISPLAY FRAME ---
    cv2.imshow("Sign Language Predictor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 9. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Program stopped.")