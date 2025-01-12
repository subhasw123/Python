import cv2
import os
import datetime
import pandas as pd

if not os.path.exists("Saved_Faces"):
    os.makedirs("Saved_Faces")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

def display_attendance_log():
    if os.path.exists("attendance_log.csv"):
        attendance_data = pd.read_csv("attendance_log.csv", header=None)
        attendance_data.columns = ["Name", "Timestamp", "Image Path"]
        print(attendance_data.to_string(index=False))
    else:
        print("\nNo attendance records found.\n")

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection - Front Camera', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('s'):
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y + h, x:x + w]
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"Saved_Faces/face_{timestamp}_{i}.jpg"
                cv2.imwrite(file_name, face_img)
                name = input("Enter the name of the person: ")
                with open("attendance_log.csv", "a") as log:
                    log.write(f"{name},{timestamp},{file_name}\n")
                print(f"Attendance recorded for {name}.")

    if key == ord('r'):
        display_attendance_log()

video_capture.release()
cv2.destroyAllWindows()
