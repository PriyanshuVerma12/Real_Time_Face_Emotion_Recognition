import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("my_model.h5")  # Your model file in the same directory

# Define emotion labels (make sure this matches your model's output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the region of interest (ROI)
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess the face image for emotion prediction
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized_face = cv2.resize(gray_face, (48, 48))  # Resize to 48x48 (common input size for emotion models)
        normalized_face = resized_face / 255.0  # Normalize pixel values to [0, 1]
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))  # Add batch dimension

        # Predict emotion
        prediction = model.predict(reshaped_face)
        emotion_index = np.argmax(prediction)  # Get the index of the predicted emotion
        predicted_emotion = emotion_labels[emotion_index]  # Get the corresponding emotion label

        # Display the predicted emotion on the frame
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with detected faces and emotion labels
    # Create the OpenCV window in fullscreen mode
    # cv2.namedWindow('Emotion Detection', cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty('Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
