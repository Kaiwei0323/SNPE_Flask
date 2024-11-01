import cv2

cap = cv2.VideoCapture('/dev/video0')  # or use '/dev/video0'

if not cap.isOpened():
    print("Cannot open camera")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

