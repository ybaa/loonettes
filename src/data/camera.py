import numpy as np
import cv2
import datetime
from src.constants import CAPTURES_FROM_CAM_PATH

class Camera:
    camNr = 0

    def __init__(self, camNr):
        self.camNr = camNr

    def __caputre__(self):
        capture = cv2.VideoCapture(self.camNr)
        while (True):
            # Capture frame-by-frame
            ret, frame = capture.read()

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if cv2.waitKey(1) & 0xFF == ord('s'):
                self.save_snapshot(CAPTURES_FROM_CAM_PATH, frame)

        # When everything done, release the capture
        capture.release()
        cv2.destroyAllWindows()

    def save_snapshot(self, path, frame):
        ts = datetime.datetime.now().timestamp()
        ts = str(ts).replace(".", "")
        fullpath = path + '/cam' + str(self.camNr) + '_'+ ts + '.jpg'
        cv2.imwrite(fullpath, frame)
        return