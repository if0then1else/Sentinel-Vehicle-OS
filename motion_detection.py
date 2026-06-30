from motion_detection_class import *

camera = cv2.VideoCapture(0)
detector = MotionDetector()

while True:
    ok, frame = camera.read()

    if not ok:
        break

    event = detector.process(frame)

    if event:
        print(f"Motion detected: {event.timestamp} Area={event.area}")

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()


