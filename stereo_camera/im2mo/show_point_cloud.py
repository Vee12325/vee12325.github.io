import cv2
import numpy as np
import open3d as o3d

# --- 1-4. ส่วนการตั้งค่าและคำนวณ (เหมือนเดิมทั้งหมด) ---

LEFT_CAMERA_INDEX = 1
RIGHT_CAMERA_INDEX = 0
CALIBRATION_FILE = 'calibration.npz'

try:
    with np.load(CALIBRATION_FILE) as data:
        mtxL, distL = data['mtxL'], data['distL']
        mtxR, distR = data['mtxR'], data['distR']
        R, T = data['R'], data['T']
        image_size = tuple(data['image_size'])
        # ⭐ โหลดเมทริกซ์ Q ที่จำเป็นสำหรับการแปลงเป็น 3D
        # ถ้าไม่มี Q ในไฟล์, เราจะคำนวณใหม่
        if 'Q' in data:
            Q = data['Q']
        else:
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR, image_size, R, T, alpha=1)
        print(f"โหลดไฟล์ '{CALIBRATION_FILE}' สำเร็จ")
except FileNotFoundError:
    print(f"ไม่พบไฟล์ '{CALIBRATION_FILE}'!")
    exit()

capL = cv2.VideoCapture(LEFT_CAMERA_INDEX)
capR = cv2.VideoCapture(RIGHT_CAMERA_INDEX)
if not capL.isOpened() or not capR.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

# คำนวณ rectification maps
R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR, image_size, R, T, alpha=1)
mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv2.CV_32FC1)

# --- 5. ตั้งค่า Stereo SGBM (เหมือนเดิม) ---
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,        # รองรับวัตถุใกล้ (~0.4–0.5 m) ถึงไกล (~4–5 m)
    blockSize=7,               # กำลังดีสำหรับ detail ปานกลาง

    P1=8 * 3 * 7**2,
    P2=32 * 3 * 7**2,

    disp12MaxDiff=1,
    uniquenessRatio=10,        # match ต้องชัดขึ้น
    speckleWindowSize=50,      # กำจัด noise
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

print("\nจัดวางวัตถุที่ต้องการ แล้วกด 's' เพื่อสร้าง Point Cloud | กด 'q' เพื่อออกจากโปรแกรม")

# --- 6. วนลูปเพื่อแสดงวิดีโอและรอคำสั่ง ---
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    # --- Rectify ภาพ ---
    rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

    # --- คำนวณ Disparity Map แบบ real-time ---
    grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Normalize ให้อยู่ในช่วง 0-255 เพื่อแสดงผลได้
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # --- แสดงหน้าต่าง ---
    cv2.imshow('Rectified View', np.hstack((rectifiedL, rectifiedR)))
    cv2.imshow("Disparity Map", disp_vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print("กำลังคำนวณ Point Cloud... กรุณารอสักครู่")

        points_3D = cv2.reprojectImageTo3D(disparity, Q)
        colors = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2RGB)

        # --- กรองจุด ---
        min_disparity = 4
        mask_disp = disparity > min_disparity
        points_3D_filtered = points_3D[mask_disp]
        mask_inf = np.isfinite(points_3D_filtered).all(axis=1)

        out_points = points_3D_filtered[mask_inf]
        out_colors = colors[mask_disp][mask_inf]

        if len(out_points) == 0:
            print("ไม่พบจุด 3D ที่จะแสดงผล! ลองปรับปรุง Disparity Map หรือเปลี่ยนมุมมอง")
            continue

        # --- แสดง Point Cloud ด้วย Open3D ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(out_points)
        pcd.colors = o3d.utility.Vector3dVector(out_colors / 255.0)

        transform = [[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]]
        pcd.transform(transform)

        print("กำลังเปิดหน้าต่าง 3D...")
        o3d.visualization.draw_geometries([pcd])
        print("ปิดหน้าต่าง 3D แล้ว กลับสู่โหมดวิดีโอ")


# --- 9. คืนทรัพยากร (เหมือนเดิม) ---
capL.release()
capR.release()
cv2.destroyAllWindows()
print("ปิดโปรแกรมเรียบร้อย")