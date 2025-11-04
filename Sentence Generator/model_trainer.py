import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- 1. LOAD THE DATASET ---
print("Loading dataset...")
# Load the CSV file into a pandas DataFrame
df = pd.read_csv('hand_landmarks.csv')

# Separate the 'label' (our 'y') from the features (our 'X')
X = df.drop('label', axis=1).values # All columns EXCEPT 'label'
y = df['label']              # Only the 'label' column

print(f"Data loaded. Found {len(y)} samples.")
print(f"Found labels: {y.unique()}") # Show what letters we found

# --- 2. SPLIT DATA FOR TRAINING AND TESTING ---
# We split the data so we can train on one part and test on another
# test_size=0.2 means 20% of data is for testing, 80% is for training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. TRAIN THE MACHINE LEARNING MODEL ---
print("Training the model...")
# We will use a "Random Forest Classifier"
# This is a great, reliable "all-around" model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. TEST THE MODEL ---
print("Testing the model...")
# Use the trained model to make predictions on the *test* data
y_pred = model.predict(X_test)

# Check how many predictions were correct
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- 5. SAVE THE TRAINED MODEL ---
# We save the model so our webcam script can use it
model_filename = 'sign_language_model.pkl'

with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_filename}")