import cv2
import datetime
from src.constants import STEREO_CAM
import numpy as np
import glob


class Camera:

    def __init__(self, left_cam_nr=STEREO_CAM['LEFT'], right_cam_nr=STEREO_CAM['RIGHT']):
        self.mono_or_left_cam_nr = left_cam_nr
        self.right_cam_nr = right_cam_nr

    def show_constant_mono_capture(self, rectify=False):
        """Show window with constant capture from a single camera.

        Click: 'q' to quit, 's' to save capture
        """
        capture = cv2.VideoCapture(self.mono_or_left_cam_nr)
        while True:
            # Capture frame-by-frame
            ret, frame = capture.read()
            frame = self.rectify_img(frame) if rectify else frame

            # Display the resulting frameq
            cv2.imshow('frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('s'):
                self.save_single_capture('../data/raw/captures/mono', frame)
                print('photo taken')
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

    def capture_stereo(self, path='../data/raw/captures/stereo', save=False, rectify=True):
        """Captures photos from stereo camera and saves it to given location."""
        capture_left = cv2.VideoCapture(self.mono_or_left_cam_nr)
        grabbed_left = capture_left.grab()

        if grabbed_left:
            ret_left, frame_left = capture_left.retrieve()
            capture_left.release()
            capture_right = cv2.VideoCapture(self.right_cam_nr)
            grabbed_right = capture_right.grab()

            if grabbed_right:
                ret_right, frame_right = capture_right.retrieve()
                capture_right.release()

                ret_left = self.rectify_img(frame_left) if rectify else frame_left
                ret_right = self.rectify_img(frame_right) if rectify else frame_right

                if save:
                    self.save_stereo_captures(path, ret_left, ret_right)
                    print('photo taken')

                return ret_left, ret_right

    def capture_constant_stereo_simultaneously(self):
        capture_left = cv2.VideoCapture(self.mono_or_left_cam_nr)
        capture_right = cv2.VideoCapture(self.right_cam_nr)

        capture_left.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        capture_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        capture_left.set(cv2.CAP_PROP_FPS, 30)

        capture_right.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        capture_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        capture_right.set(cv2.CAP_PROP_FPS, 30)

        while True:
            # Capture frame-by-frame
            ret_l, frame_l = capture_left.read()
            ret_r, frame_r = capture_right.read()

            # Display the resulting frame
            cv2.imshow('frame_r', frame_l)
            cv2.imshow('frame_l', frame_r)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        capture_left.release()
        capture_right.release()
        cv2.destroyAllWindows()

    @staticmethod
    def calibrate():
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob('../data/raw/cameraCalibration/*.jpg')

        for index, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                # cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey()

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                                   imgpoints,
                                                                   gray.shape[::-1],
                                                                   None,
                                                                   None)

                print(index)
                print(mtx)

                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                if index == len(images)-1:
                    # undistort
                    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
                    # crop the image
                    # x, y, w, h = roi
                    # rectified = dst[y:y + h, x:x + w]
                    # cv2.imshow('img', rectified)
                    # cv2.waitKey()
                    np.savetxt("../cam_mtx", mtx, delimiter=',')
                    np.savetxt("../cam_dist", dist, delimiter=',')
                    np.savetxt("../cam_roi", roi, delimiter=',')
                    np.savetxt("../cam_newcameramtx", newcameramtx, delimiter=',')

        cv2.destroyAllWindows()

    @staticmethod
    def rectify_img(img):
        mtx = np.loadtxt(fname='../cam_mtx', delimiter=',')
        dist = np.loadtxt(fname='../cam_dist', delimiter=',')
        newcameramtx = np.loadtxt(fname='../cam_newcameramtx', delimiter=',')
        roi = np.loadtxt(fname='../cam_roi', delimiter=',', dtype='int32')

        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        rectified = dst[y:y + h, x:x + w]
        # cv2.imshow('img', rectified)
        # cv2.waitKey()

        return rectified