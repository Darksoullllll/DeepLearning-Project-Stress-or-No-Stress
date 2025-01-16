import cv2
import numpy as np
from tensorflow.keras.models import load_model #type:ignore
from PIL import Image


model = load_model("./Saved Trained Model/MobileNet_Stress.keras")



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_face(face_img):
    lst = []
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)  
    img = img.resize((224, 224))  
    lst.append(np.array(img) / 255.0)
    return lst


video_path = 0
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        
        face = frame[y:y+h, x:x+w]

        preprocessed_face = preprocess_face(face)
        preprocessed_face  = np.array(preprocessed_face)
        
    
        predicted = model.predict(preprocessed_face)
        print(predicted)

        if predicted[0][0] > 0.5:
            label = "Stress"
            color = (0, 0, 255)  # Red color for stress
        else:
            label = "No Stress"
            color = (0, 255, 0)  # Green color for no stress

        # Draw a bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    cv2.imshow('Stress Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()