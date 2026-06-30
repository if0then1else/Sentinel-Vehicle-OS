from dataclasses import dataclass
from datetime import datetime
import cv2


@dataclass
class MotionEvent:
    timestamp: datetime
    area: int


class MotionDetector:
    def __init__(self, min_area: int = 1500):
        self.min_area = min_area
        self.background = None

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.background is None:
            self.background = gray
            return None

        diff = cv2.absdiff(self.background, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= self.min_area:
                return MotionEvent(
                    timestamp=datetime.now(),
                    area=int(area)
                )

        return None
    
