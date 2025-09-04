import cv2
import numpy as np

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
        print(f"โหลดไฟล์ '{CALIBRATION_FILE}' สำเร็จ")
except FileNotFoundError:
    print(f"ไม่พบไฟล์ '{CALIBRATION_FILE}'!")
    exit()
except KeyError:
    print(f"ไฟล์ '{CALIBRATION_FILE}' ไม่มีข้อมูล 'image_size' กรุณารันสคริปต์ Calibration ใหม่")
    exit()

capL = cv2.VideoCapture(LEFT_CAMERA_INDEX)
capR = cv2.VideoCapture(RIGHT_CAMERA_INDEX)
if not capL.isOpened() or not capR.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, image_size, R, T, alpha=0.9)
mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv2.CV_32FC1)

# --- 5. ตั้งค่า Stereo SGBM (เหมือนเดิม) ---
#stereo = cv2.StereoSGBM_create(
#    minDisparity=16,
#    numDisparities=16,        # รองรับวัตถุใกล้ (~0.4–0.5 m) ถึงไกล (~4–5 m)
#    blockSize=7,               # กำลังดีสำหรับ detail ปานกลาง

#    P1=8 * 3 * 7**2,
#    P2=32 * 3 * 7**2,

#    disp12MaxDiff=1,
#    uniquenessRatio=10,        # match ต้องชัดขึ้น
#    speckleWindowSize=50,      # กำจัด noise
#    speckleRange=2,
#    preFilterCap=63,
#    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#)

window_size = 7
min_disp = 16
num_disp = 16   # ต้องหารด้วย 16 ลงตัว เช่น 16, 32, 64, ...

stereo = cv2.StereoSGBM(
    minDisparity=min_disp,
    numDisparities=num_disp,
    SADWindowSize=window_size,

    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,

    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=2,
    fullDP=False
)

print("\nเริ่มต้นแสดงภาพ... กด 'q' เพื่อออกจากโปรแกรม")
disp = stereo.compute(capL, capR).astype(np.float32) / 16.0
disp_img = (disp - min_disp) / num_disp


#print("\nเริ่มต้นแสดงภาพ... กด 'q' เพื่อออกจากโปรแกรม")

# --- 6. วนลูปเพื่อแสดงผล ---
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    # 6.1 ทำการ Rectify ภาพ
    rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

    # 6.2 คำนวณ Disparity Map
    grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    
    # 6.3 ทำให้ Disparity Map มองเห็นได้ชัดเจน
    disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # --- ✨ 6.4 ส่วนของการแสดงผลที่เปลี่ยนแปลง ---
    
    # วาดเส้นแนวนอนบนภาพ Rectified เพื่อช่วยในการตรวจสอบ
    for i in range(20, image_size[1], 50):
        cv2.line(rectifiedL, (0, i), (image_size[0], i), (0, 255, 0), 1)
        cv2.line(rectifiedR, (0, i), (image_size[0], i), (0, 255, 0), 1)

    # สร้างหน้าต่างสำหรับแสดงภาพ Rectified ซ้าย-ขวา เทียบกัน
    rectified_pair = np.hstack((rectifiedL, rectifiedR))
    cv2.imshow('Rectified Stereo Pair', rectified_pair)
    
    # สร้างหน้าต่างสำหรับแสดง Disparity Map แยกต่างหาก
    cv2.imshow('Disparity Map', disparity_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. คืนทรัพยากร ---
capL.release()
capR.release()
cv2.destroyAllWindows()
print("ปิดโปรแกรมเรียบร้อย")