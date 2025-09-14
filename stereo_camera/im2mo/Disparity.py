import cv2
import numpy as np
import math

# --- 1. โหลดค่าการ Calibrate ---
CALIBRATION_FILE = 'calibration.npz'

try:
    with np.load(CALIBRATION_FILE) as data:
        mtxL, distL = data['mtxL'], data['distL']
        mtxR, distR = data['mtxR'], data['distR']
        R, T = data['R'], data['T']
        image_size = tuple(data['image_size'])
        print(f"โหลดไฟล์ '{CALIBRATION_FILE}' สำเร็จ")
except Exception as e:
    print(f"โหลด calibration ไม่ได้: {e}")
    exit()

# --- 2. เปิดกล้อง ---
LEFT_CAMERA_INDEX = 1
RIGHT_CAMERA_INDEX = 0
capL = cv2.VideoCapture(LEFT_CAMERA_INDEX)
capR = cv2.VideoCapture(RIGHT_CAMERA_INDEX)
if not capL.isOpened() or not capR.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

# --- 3. Rectification map ---
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, image_size, R, T, alpha=0.9
)
mapL1, mapL2 = cv2.initUndistortRectifyMap(
    mtxL, distL, R1, P1, image_size, cv2.CV_32FC1
)
mapR1, mapR2 = cv2.initUndistortRectifyMap(
    mtxR, distR, R2, P2, image_size, cv2.CV_32FC1
)

# --- 4. Stereo matcher ---
window_size = 7
min_disp = 0
num_disp = 64  # ต้องหารด้วย 16 ลงตัว

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# --- 5. Callback สำหรับการคลิก ---
clicked_points = []  # เก็บพิกัด 2 จุด

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    disparity, points_3D = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if disparity[y, x] > disparity.min():
            X, Y, Z = points_3D[y, x]
            print(f"คลิกที่ ({x},{y}) → พิกัด 3D = ({X:.2f}, {Y:.2f}, {Z:.2f})")

            clicked_points.append((X, Y, Z))

            if len(clicked_points) == 2:
                # คำนวณระยะ Euclidean distance
                p1, p2 = clicked_points
                dist = math.dist(p1, p2)
                print(f"📏 ระยะห่างระหว่างจุดทั้งสอง = {dist:.2f} หน่วย")
                clicked_points = []  # reset เพื่อวัดใหม่
        else:
            print("ไม่มี disparity valid ที่ตำแหน่งนี้")

cv2.namedWindow('Disparity Map')

print("\nเริ่มต้นแสดงภาพ... คลิก 2 จุดเพื่อวัดระยะ, กด 'q' เพื่อออก")

# --- 6. Loop หลัก ---
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    # Rectify
    rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

    # Grayscale
    grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)

    # Disparity
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Normalize เพื่อแสดงผล
    disparity_visual = cv2.normalize(
        disparity, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Reproject to 3D
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # ส่งข้อมูลให้ callback เวลา click
    cv2.setMouseCallback('Disparity Map', mouse_callback, param=(disparity, points_3D))

    # แสดงผล
    cv2.imshow('Rectified Left', rectifiedL)
    cv2.imshow('Disparity Map', disparity_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Cleanup ---
capL.release()
capR.release()
cv2.destroyAllWindows()
print("ปิดโปรแกรมเรียบร้อย")
