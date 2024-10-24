import cv2

<<<<<<< HEAD
cap = cv2.VideoCapture("rtsp://99.64.152.69:8554/mystream2")  # Change index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

