'''
import cv2
import subprocess
from ultralytics import YOLO

# =========================
# 1. 参数
# =========================

INPUT_RTSP = "rtsp://admin:hk000000@192.168.1.47"

OUTPUT_RTSP = "rtsp://127.0.0.1:8554/yolo"

MODEL_PATH = "model/yolo26x.pt"

WIDTH = 660
HEIGHT = 660
FPS = 20


# =========================
# 2. 加载 YOLO
# =========================

model = YOLO(MODEL_PATH)


# =========================
# 3. 打开摄像头 RTSP
# =========================

cap = cv2.VideoCapture(INPUT_RTSP)

if not cap.isOpened():
    raise RuntimeError("无法打开输入 RTSP")


# =========================
# 4. FFmpeg
# =========================

ffmpeg_cmd = [
    "ffmpeg",

    # 从 stdin 读取原始视频
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{WIDTH}x{HEIGHT}",
    "-r", str(FPS),
    "-i", "-",

    # H.264 编码
    "-c:v", "libx264",

    # 编码速度
    "-preset", "veryfast",

    # 低延迟
    "-tune", "zerolatency",

    # 像素格式
    "-pix_fmt", "yuv420p",

    # RTSP
    "-f", "rtsp",
    "-rtsp_transport", "tcp",

    OUTPUT_RTSP
]

ffmpeg_process = subprocess.Popen(
    ffmpeg_cmd,
    stdin=subprocess.PIPE
)


# =========================
# 5. 主循环
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("读取 RTSP 帧失败")
        break

    # YOLO26
    results = model(
        frame,
        conf=0.4,
        verbose=False
    )

    # 绘制检测框
    annotated_frame = results[0].plot()

    # 显示本地画面
    cv2.imshow(
        "YOLO26",
        annotated_frame
    )

    # 发送给 FFmpeg
    ffmpeg_process.stdin.write(
        annotated_frame.tobytes()
    )

    # q退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# 6. 清理
# =========================

cap.release()

ffmpeg_process.stdin.close()
ffmpeg_process.wait()

cv2.destroyAllWindows()
'''

import cv2
import time
import torch
from ultralytics import YOLO



# =========================
# 2. 加载模型
# =========================

model = YOLO("model/yolo26x.pt")


# =========================
# 3. RTSP
# =========================

RTSP_URL = "rtsp://admin:hk000000@192.168.1.47"

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("无法打开 RTSP")
    exit()

print("RTSP连接成功")


# =========================
# 4. 实时检测
# =========================

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("读取帧失败")
        break

    # YOLO
    results = model(
        frame,
        imgsz=960,
        device=0,
        conf=0.25,
        verbose=False
    )

    # 绘制检测框
    annotated_frame = results[0].plot()

    # FPS
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # 显示
    cv2.imshow(
        "YOLO26 RTSP",
        annotated_frame
    )

    # Q退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()