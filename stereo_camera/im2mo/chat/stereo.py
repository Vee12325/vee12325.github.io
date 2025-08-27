import cv2
import numpy as np
import time

# --------- ตั้งค่าอุปกรณ์กล้อง ---------
CAM_LEFT  = 0   # เปลี่ยนตามเครื่อง
CAM_RIGHT = 1   # เปลี่ยนตามเครื่อง
WIDTH, HEIGHT = 640, 480  # ใช้ความละเอียดเดียวกับตอน calibrate
FPS_TARGET = 30

def open_cam(idx, width, height, fps=30):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # บน Linux/Mac ตัด CAP_DSHOW ออกได้
    if not cap.isOpened():
        raise RuntimeError(f"เปิดกล้องไม่สำเร็จ index={idx}")
    # พยายามตั้ง MJPEG เพื่อลด latency/เพิ่ม FPS (บางเครื่องอาจไม่รองรับ)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

# --------- โหลด calibration ----------
calib = np.load("C:/Users/Pc/Desktop/im2mo/calibration.npz")
mtxL, distL = calib["mtxL"], calib["distL"]
mtxR, distR = calib["mtxR"], calib["distR"]
R, T       = calib["R"], calib["T"]
E, F       = calib["E"], calib["F"]
baseline = float(np.linalg.norm(T))  # เมตร

# --------- เปิดกล้อง ---------
capL = open_cam(CAM_LEFT,  WIDTH, HEIGHT, FPS_TARGET)
capR = open_cam(CAM_RIGHT, WIDTH, HEIGHT, FPS_TARGET)

# --------- เตรียม rectify map (คำนวณครั้งเดียว) ---------
image_size = (WIDTH, HEIGHT)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL, distL,
    mtxR, distR,
    image_size, R, T, alpha=0
)

map1x, map1y = cv2.initUndistortRectifyMap(
    mtxL, distL, R1, P1, image_size, cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    mtxR, distR, R2, P2, image_size, cv2.CV_32FC1
)


# focal length (pixel) จาก P1 หลัง rectify
focal_length = float(P1[0, 0])

# --------- ตัวช่วยคำนวณ StereoSGBM ---------
def make_matcher(params):
    blockSize = params["blockSize"]
    if blockSize % 2 == 0:
        blockSize += 1
    blockSize = max(3, min(blockSize, 21))

    numDisp = params["numDisp"]
    numDisp = max(16, (numDisp // 16) * 16)  # ต้องหาร 16 ลงตัว

    stereo = cv2.StereoSGBM_create(
        minDisparity=params["minDisp"],
        numDisparities=numDisp,
        blockSize=blockSize,
        P1=8 * 3 * (blockSize ** 2),
        P2=32 * 3 * (blockSize ** 2),
        disp12MaxDiff=params["d12Max"],
        uniquenessRatio=params["uniq"],
        speckleWindowSize=params["spw"],
        speckleRange=params["spr"],
        preFilterCap=params["pfc"],
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return stereo

# --------- UI: Trackbars ---------
cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Disparity", 800, 600)

def nothing(x): pass
# numDisp: เริ่มที่ 128, สูงสุด 256 (ต้องหาร 16 ลงตัว)
cv2.createTrackbar("numDisp (x16)", "Disparity", 8, 16, nothing)  # 8*16=128

# blockSize: patch size, odd number 5-11
cv2.createTrackbar("blockSize", "Disparity", 7, 15, nothing)

# minDisparity: เริ่ม 0
cv2.createTrackbar("minDisp", "Disparity", 0, 32, nothing)

# uniquenessRatio: 5-15
cv2.createTrackbar("uniq", "Disparity", 5, 15, nothing)

# disp12MaxDiff: 1-2
cv2.createTrackbar("d12Max", "Disparity", 1, 5, nothing)

# speckleWindowSize: 50-150
cv2.createTrackbar("spw", "Disparity", 100, 150, nothing)

# speckleRange: 16-32
cv2.createTrackbar("spr", "Disparity", 32, 32, nothing)

# preFilterCap: 31-63
cv2.createTrackbar("preFilterCap", "Disparity", 63, 63, nothing)

prev_params = None
stereo = None

print("""
Controls:
  s  : save snapshots (rectified L/R, disparity, depth)
  g  : toggle horizontal lines guide
  q/ESC : quit
""")

show_guides = True
frame_id = 0
t0 = time.time()
save_id = 0

while True:
    # ใช้ grab()/retrieve() เพื่อ sync เวลารับภาพ
    okL = capL.grab()
    okR = capR.grab()
    if not (okL and okR):
        print("กรอบภาพไม่มา บางกล้องอาจไม่พร้อม")
        break

    _, frameL = capL.retrieve()
    _, frameR = capR.retrieve()

    # แปลงเป็น gray
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Rectify (remap) ให้ epipolar line ขนานกัน
    rectL = cv2.remap(grayL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(grayR, map2x, map2y, cv2.INTER_LINEAR)

    # อ่านค่าจาก trackbars
    params = dict(
        numDisp = max(16, cv2.getTrackbarPos("numDisp (x16)", "Disparity") * 16),
        blockSize = max(3, cv2.getTrackbarPos("blockSize", "Disparity")),
        minDisp = cv2.getTrackbarPos("minDisp", "Disparity"),
        uniq = cv2.getTrackbarPos("uniq", "Disparity"),
        d12Max = cv2.getTrackbarPos("d12Max", "Disparity"),
        spw = cv2.getTrackbarPos("spw", "Disparity"),
        spr = cv2.getTrackbarPos("spr", "Disparity"),
        pfc = max(1, cv2.getTrackbarPos("preFilterCap", "Disparity")),
    )

    # สร้าง/อัพเดต matcher เมื่อ param เปลี่ยน
    if params != prev_params:
        stereo = make_matcher(params)
        prev_params = params

    # คำนวณ disparity (ค่าที่ OpenCV คืนมา scale ด้วย 16)
    disparity_raw = stereo.compute(rectL, rectR).astype(np.float32) / 16.0

    # depth (เมตร): depth = f * B / d (เฉพาะ d > 0)
    depth = np.zeros_like(disparity_raw, dtype=np.float32)
    valid = disparity_raw > 0.0
    depth[valid] = (focal_length * baseline) / disparity_raw[valid]

    # Visualization
    disp_vis = cv2.normalize(disparity_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # รวมภาพซ้าย+ขวา เพื่อดูเส้นไกด์เช็คการ rectify
    both = np.vstack([rectL, rectR])
    both_color = cv2.cvtColor(both, cv2.COLOR_GRAY2BGR)
    if show_guides:
        h = both_color.shape[0] // 2
        for y in range(0, both_color.shape[0], 40):
            cv2.line(both_color, (0, y), (both_color.shape[1]-1, y), (0, 255, 0), 1)

    cv2.imshow("Rectified L/R (stacked)", both_color)
    cv2.imshow("Disparity", disp_vis)
    cv2.imshow("Depth (normalized)", depth_vis)

    # FPS แสดงใน title ทุก ๆ วินาที
    frame_id += 1
    if frame_id % 10 == 0:
        t1 = time.time()
        fps = 10.0 / (t1 - t0 + 1e-6)
        t0 = t1
        cv2.setWindowTitle("Disparity", f"Disparity  |  FPS ~ {fps:.1f}")

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):  # ESC หรือ q
        break
    elif key == ord('g'):
        show_guides = not show_guides
    elif key == ord('s'):
        cv2.imwrite(f"rectL_{save_id}.png", rectL)
        cv2.imwrite(f"rectR_{save_id}.png", rectR)
        cv2.imwrite(f"disp_{save_id}.png", disp_vis)
        cv2.imwrite(f"depth_{save_id}.png", depth_vis)
        print(f"Saved: rectL_{save_id}.png, rectR_{save_id}.png, disp_{save_id}.png, depth_{save_id}.png")
        save_id += 1

capL.release()
capR.release()
cv2.destroyAllWindows()
