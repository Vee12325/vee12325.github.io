import cv2
import numpy as np
import os

# ---------- CONFIG ----------
CAM_LEFT  = 1
CAM_RIGHT = 0
WIDTH, HEIGHT = 640, 480
PATTERN_SIZE = (10, 7)   # 9x6 chessboard (ในแนว "inner corners")
SQUARE_SIZE = 0.025     # ขนาดช่อง chessboard (เมตร)
CALIB_FILE = "calibration.npz"

# เตรียม object points (0,0,0) ... (8,5,0)
objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points
imgpointsL = [] # 2D points left
imgpointsR = [] # 2D points right

def open_cam(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # ตัด CAP_DSHOW ได้ถ้าเป็น Linux
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

capL = open_cam(CAM_LEFT)
capR = open_cam(CAM_RIGHT)

if not capL.isOpened() or not capR.isOpened():
    raise RuntimeError("ไม่สามารถเปิดกล้องได้ ตรวจสอบ index CAM_LEFT / CAM_RIGHT")

print("""
Controls:
  SPACE : ตรวจ chessboard และบันทึกคู่ภาพ
  c     : คำนวณ calibration
  q/ESC : ออก
""")

while True:
    capL.grab()
    capR.grab()
    _, frameL = capL.retrieve()
    _, frameR = capR.retrieve()

    both = cv2.hconcat([frameL, frameR])
    cv2.imshow("Stereo Capture (L | R)", both)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

    elif key == 32:  # SPACE
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, PATTERN_SIZE, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, PATTERN_SIZE, None)

        if retL and retR:
            objpoints.append(objp)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

            cv2.drawChessboardCorners(frameL, PATTERN_SIZE, cornersL, retL)
            cv2.drawChessboardCorners(frameR, PATTERN_SIZE, cornersR, retR)
            print(f"เก็บ sample แล้ว: {len(objpoints)}")
        else:
            print("หา chessboard ไม่เจอ ลองใหม่")

    elif key == ord('c'):  # Run calibration
        if len(objpoints) < 10:
            print("เก็บ sample อย่างน้อย 10 คู่ ก่อน calibration")
            continue

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        image_size = grayL.shape[::-1]

        print("กำลังคำนวณ calibration ...")
        retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, image_size, None, None)
        retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, image_size, None, None)

        flags = cv2.CALIB_FIX_INTRINSIC
        ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpointsL, imgpointsR,
            mtxL, distL, mtxR, distR,
            image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=flags
        )
        output_filename = "calibration.npz"
        print("Calibration เสร็จ ✅")
        np.savez(CALIB_FILE,
                 mtxL=mtxL, distL=distL,
                 mtxR=mtxR, distR=distR,
                 R=R, T=T, E=E, F=F)
        print(f"บันทึก calibration ที่ {CALIB_FILE}")

        full_path = os.path.abspath(output_filename)
        print(f"File location:")
        print(f"--> {full_path}")

capL.release()
capR.release()
cv2.destroyAllWindows()
