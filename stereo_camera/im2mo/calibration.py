import cv2
import numpy as np
import time
import os

# --- 1. การตั้งค่า (Configuration) ---
# ขนาด chessboard (จำนวนมุมด้านใน ไม่ใช่จำนวนช่อง)
pattern_size = (10, 7)
square_size = 0.025  # เมตร (สำคัญมาก! ต้องวัดขนาดจริงให้แม่นยำ)

# เตรียมจุด 3D ของ chessboard ในโลกจริง (Object Points)
# สร้าง grid ที่มีขนาดเท่า pattern_size และคูณด้วยขนาดช่อง
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# List เก็บจุด
objpoints = []  # 3D points in real world space
imgpointsL = [] # 2D points in left image plane
imgpointsR = [] # 2D points in right image plane

# --- 2. การเก็บข้อมูลภาพ (Image Acquisition) ---
# ปรับเลข device ตามจริง (อาจจะเป็น 0, 1 หรือ 1, 2)
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(0)

# ⭐ ปรับปรุง: เพิ่มตัวแปรสำหรับ UI feedback
capture_message_display_time = 0
message = ""

print("มองหา Chessboard ในกล้องทั้งสองตัว...")
print("กด 'c' เพื่อถ่ายภาพเมื่อตรวจพบ | กด 'ESC' เพื่อจบการทำงานและเริ่มคำนวณ")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("ไม่สามารถอ่านค่าจากกล้องได้")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # หา chessboard corners
    retL_corners, cornersL = cv2.findChessboardCorners(grayL, pattern_size, None)
    retR_corners, cornersR = cv2.findChessboardCorners(grayR, pattern_size, None)

    # สร้างภาพสำหรับแสดงผล
    visL, visR = frameL.copy(), frameR.copy()

    # ⭐ ปรับปรุง: แสดงจำนวนเฟรมที่ถ่ายไปแล้วบนหน้าจอ
    cv2.putText(visL, f"Frames: {len(objpoints)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(visR, f"Frames: {len(objpoints)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if retL_corners: cv2.drawChessboardCorners(visL, pattern_size, cornersL, retL_corners)
    if retR_corners: cv2.drawChessboardCorners(visR, pattern_size, cornersR, retR_corners)
    
    # ⭐ ปรับปรุง: แสดงข้อความ feedback ชั่วคราวหลังถ่ายภาพ
    if time.time() < capture_message_display_time:
        cv2.putText(visL, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(visR, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow('Left Camera', visL)
    cv2.imshow('Right Camera', visR)

    key = cv2.waitKey(1)
    if key == ord('c') and retL_corners and retR_corners:
        # ⭐ ปรับปรุง: เพิ่มความแม่นยำด้วย cornerSubPix
        # เป็นการหาตำแหน่งมุมที่ละเอียดกว่าระดับพิกเซล
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersL_subpix = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR_subpix = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        
        # เก็บข้อมูลที่ refine แล้ว
        objpoints.append(objp)
        imgpointsL.append(cornersL_subpix)
        imgpointsR.append(cornersR_subpix)
        
        message = f"Captured frame {len(objpoints)}!"
        print(message)
        capture_message_display_time = time.time() + 1.5 # แสดงข้อความ 1.5 วินาที

    elif key == 27:  # ESC เพื่อหยุด
        if len(objpoints) < 10:
            print("Warning: จำนวนภาพน้อยเกินไป (แนะนำ 15-20 ภาพ) อาจทำให้ผลลัพธ์ไม่แม่นยำ")
        break

capL.release()
capR.release()
cv2.destroyAllWindows()

if len(objpoints) == 0:
    print("ไม่มีการถ่ายภาพ โปรแกรมจะปิดตัวลง")
    exit()

print("\nกำลังประมวลผล Calibration... กรุณารอสักครู่...")

# --- 3. การคำนวณ Calibration ---
image_size = grayL.shape[::-1]

# 🔹 Calibration แยกกล้อง
# ⭐ ปรับปรุง: รับค่า re-projection error กลับมาด้วย (retL, retR)
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, image_size, None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, image_size, None, None)

# 🔹 Stereo Calibration
# ⭐ ปรับปรุง: รับค่า re-projection error ของ stereo กลับมาด้วย (ret)
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    image_size,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC
)

# --- 4. แสดงผลลัพธ์และบันทึก ---
print("\n--- Calibration Results ---")
print(f"Left Camera RMS re-projection error: {retL:.4f} pixels")
print(f"Right Camera RMS re-projection error: {retR:.4f} pixels")
print(f"Stereo RMS re-projection error: {ret:.4f} pixels")
print("--> (ค่า error ควรจะน้อยกว่า 1.0, ยิ่งใกล้ยิ่ง 0 ดี)")

# 🔹 แสดง Baseline
baseline = np.linalg.norm(T)
print(f"\nBaseline distance: {baseline * 100:.2f} cm") # แปลงเป็น cm เพื่อให้ดูง่าย

# 🔹 บันทึก calibration.npz
output_filename = "calibration.npz"
np.savez(output_filename,
         mtxL=mtxL, distL=distL,
         mtxR=mtxR, distR=distR,
         R=R, T=T,
         E=E, F=F,
         image_size=image_size)
print(f"\nCalibration saved successfully to '{output_filename}'")

# ✨ เพิ่มโค้ด 3 บรรทัดนี้เพื่อแสดงที่ตั้งไฟล์
full_path = os.path.abspath(output_filename)
print(f"File location:")
print(f"--> {full_path}")