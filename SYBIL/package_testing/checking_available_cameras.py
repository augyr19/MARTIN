import cv2

for i in [0, 1]:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Showing preview for camera index {i}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()
        cap.release()
