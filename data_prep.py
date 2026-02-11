import os
import cv2
import numpy as np

IMG_SIZE = 48
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(folder):
    X, y = [], []
    count = 0

    for emotion in os.listdir(folder):
        emotion_path = os.path.join(folder, emotion)
        if not os.path.isdir(emotion_path):
            continue

        label = 1 if emotion == "happy" else 0

        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (48, 48))
            X.append(img)
            y.append(label)

            count += 1
            if count % 1000 == 0:
                print(f"Loaded {count} images...")

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y)

    return X, y


# Load train and test data
X_train, y_train = load_data(os.path.join(BASE_DIR, "train"))
X_test, y_test = load_data(os.path.join(BASE_DIR, "test"))

# Reshape for CNN
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# Save
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Data preparation completed")
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))
