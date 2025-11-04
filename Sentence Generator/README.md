# Real-Time Sign Language Recognition

This is a Python project that uses OpenCV and MediaPipe to detect hand gestures in real-time from a webcam. It uses a custom-trained Machine Learning model (a Random Forest) to classify signs and build sentences.



---

## Features

* **Real-Time Hand Tracking:** Uses `mediapipe` to get 21 landmarks.
* **Custom ML Model:** Trained on `scikit-learn` to recognize custom signs.
* **Sentence Building:** Can build words from letters (A, B, C, D) and add them to a sentence.
* **Special Gestures:**
    * **"Space" (Open Palm):** Adds the current word to the sentence.
    * **"Clear" (Thumbs Up):** Clears the current word.
* **Cooldown Timer:** A 3-second cooldown after "Space" to prevent ghost predictions.

---

## How It Works

The project is built in three phases:
1.  **`data_collector.py`:** A script to capture hand landmarks for new signs and save them to `hand_landmarks.csv`.
2.  **`model_trainer.py`:** This script reads the CSV, trains a `RandomForestClassifier`, and saves the final AI "brain" as `sign_language_model.pkl`.
3.  **`live_predictor.py`:** The main application. It loads the `.pkl` model and uses the webcam to classify signs and build sentences live.

---

## How to Run

1.  **Clone the repository:**
    ```
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    ```
2.  **Install dependencies:**
    ```
    pip install opencv-python mediapipe scikit-learn pandas numpy keyboard
    ```
3.  **Train Your Model:**
    You must train your own model, as the `.pkl` file is not included.

    * Run `data_collector.py` and follow the on-screen prompts to create `hand_landmarks.csv`.
    ```
    python data_collector.py
    ```
    * Run `model_trainer.py` to create the `sign_language_model.pkl` file.
    ```
    python model_trainer.py
    ```
4.  **Run the application!**
    ```
    python live_predictor.py
    ```