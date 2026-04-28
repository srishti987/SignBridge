import cv2
import os
import numpy as np
import mediapipe as mp
from datetime import datetime

# words to sign
WORDS = [
    "yes",
    "no",
    "wait",
    "go",
    "stop",
    "help",
    "want",
    "need",
    "like",
    "good",
    "bad",
    "eat",
    "drink",
    "book",
    "please",
    "hello",
    "talk"
]

SEQ_LEN = 32
DATA_PATH = "dataset" # for directory or path issue refer this variable.

mp_hands = mp.solutions.hands # pyright: ignore [reportAttributeAccessIssue]
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils # pyright: ignore [reportAttributeAccessIssue]

cap = cv2.VideoCapture(0)

# Create folders automatically
for word in WORDS:
    os.makedirs(os.path.join(DATA_PATH, word), exist_ok=True)

current_word_idx = 0
recording = False
sequence = []

# helpers
def extract_keypoints(results):
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*63)  # 21 points * 3 coords

    # Ensure 2 hands (pad if 1 hand)
    while len(keypoints) < 126:
        keypoints.extend([0]*3)

    return np.array(keypoints)

def save_sequence(word, seq):
    word_path = os.path.join(DATA_PATH, word)
    count = len(os.listdir(word_path))
    filename = f"sample_{count+1:03d}.npy"
    np.save(os.path.join(word_path, filename), np.array(seq))
    print(f"[SAVED] {word}/{filename}")

# MAIN LOOP 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    word = WORDS[current_word_idx]

    # UI overlay
    cv2.putText(image, f"WORD: {word}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    status = "RECORDING" if recording else "PAUSED"
    cv2.putText(image, f"STATUS: {status}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Recording logic
    if recording:
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == SEQ_LEN:
            save_sequence(word, sequence)
            sequence = []
            recording = False  # auto pause

    cv2.imshow("Data Collector", image)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('r'):
        recording = True
        sequence = []
        print(f"[RECORDING STARTED] {word}")

    elif key == ord('p'):
        recording = False
        sequence = []
        print("[PAUSED]")

    elif key == ord('n'):
        current_word_idx = (current_word_idx + 1) % len(WORDS)

    elif key == ord('b'):
        current_word_idx = (current_word_idx - 1) % len(WORDS)

cap.release()
cv2.destroyAllWindows()
