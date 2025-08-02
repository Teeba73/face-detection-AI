import cv2
from deepface import DeepFace

#Start video capture
cap = cv2.VideoCapture(0)

#Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the detected face for emotion analysis
        roi_color = frame[y:y + h, x:x + w]

        try:
            # Use DeepFace to analyze emotions
            analysis = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False)
            # Extract the dominant emotion
            dominant_emotion = analysis['dominant_emotion']

            # Display the emotion on the frame
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error analyzing face: {e}")

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
