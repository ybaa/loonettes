import cv2
import datetime
from src.constants import MONO_CAPTURE_FROM_CAM_PATH, STEREO_CAPTURES_FROM_CAM_PATH, STEREO_CAM


class Camera:

    def __init__(self, left_cam_nr=STEREO_CAM['LEFT'], right_cam_nr=STEREO_CAM['RIGHT']):
        self.mono_or_left_cam_nr = left_cam_nr
        self.right_cam_nr = right_cam_nr

    def show_constant_mono_capture(self):
        """Show window with constant capture from a single camera.

        Click: 'q' to quit, 's' to save capture
        """
        capture = cv2.VideoCapture(self.mono_or_left_cam_nr)
        while (True):
            # Capture frame-by-frame
            ret, frame = capture.read()

            # Display the resulting frame
            cv2.imshow('frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('s'):
                self.save_single_capture(MONO_CAPTURE_FROM_CAM_PATH, frame)
            elif cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        capture.release()
        cv2.destroyAllWindows()

    def save_single_capture(self, path, frame):
        ts = datetime.datetime.now().timestamp()
        ts = str(ts).replace(".", "")
        full_path = path + '/cam' + str(self.mono_or_left_cam_nr) + '_' + ts + '.jpg'
        cv2.imwrite(full_path, frame)
        return

    def save_stereo_captures(self, path, frame_left, frame_right):
        ts = datetime.datetime.now().timestamp()
        ts = str(ts).replace(".", "")
        full_path_left = path + '/cam' + str(self.mono_or_left_cam_nr) + '_' + ts + '_L' '.jpg'
        full_path_right = path + '/cam' + str(self.mono_or_left_cam_nr) + '_' + ts + '_R' '.jpg'
        cv2.imwrite(full_path_left, frame_left)
        cv2.imwrite(full_path_right, frame_right)
        return

    def capture_stereo(self, path=STEREO_CAPTURES_FROM_CAM_PATH, save=False):
        """Captures photos from stereo camera and saves it to given location."""
        capture_left = cv2.VideoCapture(2)
        grabbed_left = capture_left.grab()

        if grabbed_left:
            ret_left, frame_left = capture_left.retrieve()
            capture_left.release()
            capture_right = cv2.VideoCapture(4)
            grabbed_right = capture_right.grab()

            if grabbed_right:
                ret_right, frame_right = capture_right.retrieve()
                capture_right.release()

                if save:
                    self.save_stereo_captures(path, frame_left, frame_right)

                return frame_left, frame_right
