import cv2
import mediapipe as mp
import pickle
import numpy as np
import time  # <-- 1. IMPORT TIME LIBRARY

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

# --- 3. NEW: LOGIC VARIABLES ---
current_word = ""
current_sentence = ""

# Debounce variables
last_prediction = None
prediction_debounce_counter = 0
DEBOUNCE_THRESHOLD = 10 

# Cooldown variables
COOLDOWN_SECONDS = 3  # 3-second cooldown
cooldown_end_time = 0 # Time when cooldown ends

print("Starting webcam... Spell words! Press 'q' to quit.")

# --- 4. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.flip(rgb_frame, 1)

    results = hands.process(rgb_frame)

    prediction_text = "No Hand"
    current_time = time.time() # Get current time

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
        prediction = model.predict(data_for_prediction)
        prediction_text = prediction[0]

        # --- 7. NEW: COOLDOWN & CLEAR LOGIC ---

        # Check if we are in a cooldown period
        if current_time < cooldown_end_time:
            # If so, just display "Cooldown" and skip all logic
            prediction_text = "Cooldown..."
        else:
            # We are NOT in cooldown, so process normally
            
            # Debounce: Check if the prediction is stable
            if prediction_text == last_prediction:
                prediction_debounce_counter += 1
            else:
                last_prediction = prediction_text
                prediction_debounce_counter = 0

            # If prediction is stable (past the threshold)
            if prediction_debounce_counter > DEBOUNCE_THRESHOLD:
                
                # A) Check for "Clear" (Thumbs Up)
                if prediction_text == "Clear":
                    current_word = "" # Clear the current word
                
                # B) Check for "Space" (Open Palm)
                elif prediction_text == "Space":
                    if current_word: # Make sure word is not empty
                        current_sentence += current_word + " "
                        current_word = "" # Reset the current word
                    
                    # !! START COOLDOWN !!
                    cooldown_end_time = current_time + COOLDOWN_SECONDS
                
                # C) Check for a letter
                elif prediction_text != "No Hand":
                    # Add letter to word, but only if it's different
                    if not current_word or prediction_text != current_word[-1]:
                        current_word += prediction_text

                # Reset counter so we don't add letters 100x
                prediction_debounce_counter = 0

    # --- 8. DISPLAY PREDICTION & SENTENCE ---
    # Display the current prediction
    cv2.putText(
        frame, 
        f"Prediction: {prediction_text}", 
        (10, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (0, 255, 0), 2
    )
    
    # Display the current word being built
    cv2.putText(
        frame, 
        f"Word: {current_word}", 
        (10, 100), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (255, 0, 0), 2
    )

    # Display the full sentence
    cv2.putText(
        frame, 
        f"Sentence: {current_sentence}", 
        (10, 150), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (255, 0, 0), 2
    )

    # --- 9. DISPLAY FRAME ---
    cv2.imshow("Sign Language Predictor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 10. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Program stopped.")