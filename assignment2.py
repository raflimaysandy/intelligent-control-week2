import numpy as np
import os
import cv2
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Path ke dataset
DATASET_PATH = 'training_dataset'

data = []
labels = []

# Membaca dataset dari folder
for color_name in os.listdir(DATASET_PATH):
    color_path = os.path.join(DATASET_PATH, color_name)
    if os.path.isdir(color_path):
        for img_name in os.listdir(color_path):
            img_path = os.path.join(color_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            avg_color = img.mean(axis=(0, 1))  # Ambil rata-rata warna RGB
            data.append(avg_color)
            labels.append(color_name)

# Konversi ke array numpy
X = np.array(data)
y = np.array(labels)

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Melatih model Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"Akurasi pada data latih: {train_acc*100:.2f}%")
print(f"Akurasi pada data uji: {accuracy*100:.2f}%")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definisi rentang warna untuk deteksi
    colors_ranges = {
        'Red': [(0, 120, 70), (10, 255, 255)],
        'Green': [(36, 100, 100), (86, 255, 255)],
        'Blue': [(94, 80, 2), (126, 255, 255)],
        'Yellow': [(15, 100, 100), (35, 255, 255)],
        'Purple': [(129, 50, 70), (158, 255, 255)],
        'White': [(0, 0, 200), (180, 30, 255)],
        'Black': [(0, 0, 0), (180, 255, 50)]
    }
    
    for color_name, (lower, upper) in colors_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                color_bgr = {
                    'Red': (0, 0, 255), 'Green': (0, 255, 0), 'Blue': (255, 0, 0),
                    'Yellow': (0, 255, 255), 'Purple': (128, 0, 128), 'White': (255, 255, 255), 'Black': (0, 0, 0)
                }[color_name]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()