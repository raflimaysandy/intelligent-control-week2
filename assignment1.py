import numpy as np
import joblib
import pandas as pd
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
color_data = pd.read_csv('colors.csv')
X = color_data[['R', 'G', 'B']].values
y = color_data['color_name'].values

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Simpan model dan scaler
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluasi model
y_pred = svm_model.predict(X_test)
train_preds = svm_model.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)
accuracy = accuracy_score(y_test, y_pred)

print(f"Akurasi pada data latih: {train_acc * 100:.2f} %")
print(f"Akurasi pada data uji: {accuracy * 100:.2f} %")

# Load model dan scaler
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Pilih dua area untuk deteksi warna
    region1 = frame[height // 3, width // 3]
    region2 = frame[2 * height // 3, 2 * width // 3]

    # Normalisasi sebelum prediksi
    region1_scaled = scaler.transform([region1])
    region2_scaled = scaler.transform([region2])

    # Prediksi warna
    color_pred1 = svm_model.predict(region1_scaled)[0]
    color_pred2 = svm_model.predict(region2_scaled)[0]

    # Konversi warna dari RGB ke BGR untuk bounding box
    color1_rgb = tuple(map(int, region1[::-1]))
    color2_rgb = tuple(map(int, region2[::-1]))

    # Tambahkan teks keterangan warna utama
    cv2.putText(frame, f'Color 1: {color_pred1}', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f'Color 2: {color_pred2}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Tampilkan bounding box dengan warna yang sesuai
    cv2.rectangle(frame, (width // 3 - 50, height // 3 - 50), (width // 3 + 50, height // 3 + 50), color1_rgb, 2)
    cv2.putText(frame, color_pred1, (width // 3 - 50, height // 3 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1_rgb, 2)

    cv2.rectangle(frame, (2 * width // 3 - 50, 2 * height // 3 - 50), (2 * width // 3 + 50, 2 * height // 3 + 50), color2_rgb, 2)
    cv2.putText(frame, color_pred2, (2 * width // 3 - 50, 2 * height // 3 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2_rgb, 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
