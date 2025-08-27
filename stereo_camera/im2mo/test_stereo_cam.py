import cv2
import numpy as np

# --- 1. ตั้งค่าพื้นฐาน ---

# แก้ไข index ของกล้องให้ตรงกับของคุณ
# อาจจะเป็น (0, 1), (1, 2) หรืออื่นๆ ลองเช็คดูครับ
LEFT_CAMERA_INDEX = 1
RIGHT_CAMERA_INDEX = 0

# ระบุตำแหน่งไฟล์ calibration.npz ของคุณ
CALIBRATION_FILE = 'calibration.npz'

# --- 2. โหลดค่า Calibration ---

try:
    with np.load(CALIBRATION_FILE) as data:
        mtxL = data['mtxL']
        distL = data['distL']
        mtxR = data['mtxR']
        distR = data['distR']
        R = data['R']
        T = data['T']
        print(f"โหลดไฟล์ '{CALIBRATION_FILE}' สำเร็จ")
except FileNotFoundError:
    print(f"ไม่พบไฟล์ '{CALIBRATION_FILE}'! กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกัน")
    exit()

# --- 3. เริ่มต้นการทำงานของกล้อง ---

capL = cv2.VideoCapture(LEFT_CAMERA_INDEX)
capR = cv2.VideoCapture(RIGHT_CAMERA_INDEX)

if not capL.isOpened() or not capR.isOpened():
    print("ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบ Camera Index")
    exit()

# ตั้งค่าความละเอียดของกล้อง (ถ้าต้องการ)
# capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# capR.set(cv2.CAP_P RO P_FRAME_HEIGHT, 480)

# --- 4. คำนวณค่าสำหรับ Rectification ---

# ดึงขนาดภาพจากเฟรมแรก
retL, frameL = capL.read()
if not retL:
    print("ไม่สามารถอ่านภาพจากกล้องซ้ายได้")
    exit()
image_size = (frameL.shape[1], frameL.shape[0])

# คำนวณ Rotation matrix, Projection matrix และอื่นๆ สำหรับการ Rectify
# alpha=0.0 หมายถึงให้ภาพที่ rectify แล้วไม่มีพิกเซลสีดำเลย (อาจถูกครอปบางส่วน)
# alpha=1.0 หมายถึงให้เก็บพิกเซลทั้งหมดไว้ (อาจเห็นขอบสีดำ)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, image_size, R, T, alpha=0.9
)

# สร้าง map สำหรับการแปลงภาพ (remap)
mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv2.CV_32FC1)

print("\nเริ่มต้นแสดงภาพ... กด 'q' เพื่อออกจากโปรแกรม")

# --- 5. วนลูปเพื่อแสดงผล ---

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("ไม่สามารถรับภาพจากกล้องได้")
        break

    # 5.1 แสดงภาพดิบที่ยังไม่แก้ไข (Original)
    original_images = np.hstack((frameL, frameR)) # ต่อภาพซ้าย-ขวาในแนวนอน
    cv2.imshow('Original Uncalibrated Images', original_images)

    # 5.2 ทำการ Rectify ภาพ
    rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)
    
    # 5.3 วาดเส้นแนวนอนเพื่อช่วยให้เห็นภาพชัดขึ้น
    for i in range(20, image_size[1], 50):
        cv2.line(rectifiedL, (0, i), (image_size[0], i), (0, 255, 0), 1)
        cv2.line(rectifiedR, (0, i), (image_size[0], i), (0, 255, 0), 1)

    # 5.4 แสดงภาพที่แก้ไขแล้ว (Rectified)
    rectified_images = np.hstack((rectifiedL, rectifiedR))
    cv2.imshow('Rectified Images', rectified_images)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. คืนทรัพยากร ---

capL.release()
capR.release()
cv2.destroyAllWindows()
print("ปิดโปรแกรมเรียบร้อย")