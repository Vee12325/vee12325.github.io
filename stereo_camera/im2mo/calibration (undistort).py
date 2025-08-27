import cv2
import numpy as np
import os

# โหลด calibration
calib_file = "calibration.npz"
if not os.path.exists(calib_file):
    print(f"{calib_file} not found!")
    exit()

data = np.load(calib_file)
mtxL, distL = data["mtxL"], data["distL"]
mtxR, distR = data["mtxR"], data["distR"]

# เปิดกล้อง
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Cannot read from cameras")
        break

    # 🔹 Undistort ภาพ
    frameL_ud = cv2.undistort(frameL, mtxL, distL, None, mtxL)
    frameR_ud = cv2.undistort(frameR, mtxR, distR, None, mtxR)

    # แสดงภาพ
    cv2.imshow("Left Camera - Undistorted", frameL_ud)
    cv2.imshow("Right Camera - Undistorted", frameR_ud)

    key = cv2.waitKey(1)
    if key == 27:  # ESC เพื่อออก
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
