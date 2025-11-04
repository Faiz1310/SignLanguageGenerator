import cv2
import mediapipe as mp
import keyboard  # You may need to run: pip install keyboard
import csv
import os

# --- 1. SETUP ---

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- 2. NEW: CSV File Setup ---
# Define the name of our output file
csv_file_name = 'hand_landmarks.csv'

# Check if the file already exists to decide on writing headers
file_exists = os.path.isfile(csv_file_name)

# Open the CSV file in 'append' mode
csv_file = open(csv_file_name, 'a', newline='')
csv_writer = csv.writer(csv_file)

# If the file is NEW, write the header row
if not file_exists:
    header = ['label']
    for i in range(21):
        header += [f'x{i}', f'y{i}', f'z{i}']
    csv_writer.writerow(header)

print("Data Collector running...")
print("Hold a sign (e.g., 'A') and press the 'a' key to save.")
print("Press 'q' to quit.")

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# --- 3. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.flip(rgb_frame, 1)

    results = hands.process(rgb_frame)

    # --- 4. DRAWING & DATA EXTRACTION ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw the landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # --- 5. Data Saving Logic ---
        # Add all the letters you want to collect here
        label = None
        if keyboard.is_pressed('a'):
            label = 'A'
        elif keyboard.is_pressed('b'):
            label = 'B'
        elif keyboard.is_pressed('c'):
            label = 'C'
        elif keyboard.is_pressed('d'):
            label = 'D'
        # ...you can add 'e', 'f', etc. here

        # If a key was pressed:
        if label:
            # 1. Extract the landmark data
            landmarks_row = [label] # Start the row with the label
            for landmark in hand_landmarks.landmark:
                landmarks_row.append(landmark.x)
                landmarks_row.append(landmark.y)
                landmarks_row.append(landmark.z)
            
            # 2. Write the data to the CSV file
            csv_writer.writerow(landmarks_row)
            
            print(f"Saved data for: {label}")
            
            # A small delay to prevent saving 100s of rows
            cv2.waitKey(500) 


    # --- 6. DISPLAY FRAME ---
    cv2.imshow("Data Collector", frame)

    # Check for 'q' to quit the main loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. CLEANUP ---
cap.release()
csv_file.close() # <-- IMPORTANT: Close the file
cv2.destroyAllWindows()
print(f"Data saved to {csv_file_name}")