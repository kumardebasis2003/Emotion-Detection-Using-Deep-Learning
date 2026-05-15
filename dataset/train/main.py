import cv2
import numpy as np

from keras.models import load_model

model = load_model('model.h5')

face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        roi_gray = gray[y:y+h, x:x+w]

        roi_gray = cv2.resize(roi_gray,(48,48))

        roi = roi_gray.astype('float')/255.0

        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi)

        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            2
        )

        cv2.putText(
            frame,
            emotion,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow('Emotion Detection',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()