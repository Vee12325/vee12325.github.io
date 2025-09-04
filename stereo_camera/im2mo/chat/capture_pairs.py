import cv2
import os

# ---------- CONFIG ----------
CAM_LEFT  = 0   # index กล้องซ้าย
CAM_RIGHT = 1   # index กล้องขวา
WIDTH, HEIGHT = 640, 480
SAVE_DIR_LEFT = "images/left"
SAVE_DIR_RIGHT = "images/right"

os.makedirs(SAVE_DIR_LEFT, exist_ok=True)
os.makedirs(SAVE_DIR_RIGHT, exist_ok=True)

def open_cam(idx, width, height, fps=30):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows ใช้ CAP_DSHOW, Linux/Ubuntu ตัดออกได้
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

capL = open_cam(CAM_LEFT, WIDTH, HEIGHT, 30)
capR = open_cam(CAM_RIGHT, WIDTH, HEIGHT, 30)

if not capL.isOpened() or not capR.isOpened():
    raise RuntimeError("ไม่สามารถเปิดกล้องได้ ตรวจสอบ index CAM_LEFT / CAM_RIGHT")

counter = 0
print("""
Controls:
  SPACE : บันทึกคู่ภาพ (left/right)
  q     : ออก
""")

while True:
    # sync frame
    capL.grab()
    capR.grab()
    _, frameL = capL.retrieve()
    _, frameR = capR.retrieve()

    both = cv2.hconcat([frameL, frameR])
    cv2.imshow("Stereo Capture (L | R)", both)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q หรือ ESC
        break
    elif key == 32:  # SPACE
        fnameL = os.path.join(SAVE_DIR_LEFT,  f"left_{counter:03d}.png")
        fnameR = os.path.join(SAVE_DIR_RIGHT, f"right_{counter:03d}.png")
        cv2.imwrite(fnameL, frameL)
        cv2.imwrite(fnameR, frameR)
        print(f"Saved {fnameL}, {fnameR}")
        counter += 1

capL.release()
capR.release()
cv2.destroyAllWindows()
