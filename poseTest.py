import cv2
from ultralytics import YOLO

# 加载 Pose 模型
model = YOLO("model/yolo26m-pose.pt")

# RTSP 地址
RTSP_URL = "rtsp://192.168.0.171/720p-stream"

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("无法打开 RTSP")
    exit()

print("RTSP连接成功")

#记录举手帧数
raise_hand_count = 0
action = "None"

while True:

    ret, frame = cap.read()

    if not ret:
        print("读取视频帧失败")
        break

    # 姿态估计
    results = model(
        frame,
        imgsz=640,
        device=0,
        conf=0.25,
        verbose=True
    )

    #打印关键点数据到terminal
    '''
    if results[0].keypoints is not None:

        keypoints = results[0].keypoints.xy

        for person_id, person in enumerate(keypoints):

            print(f"Person {person_id}")

            for point_id, point in enumerate(person):
                x, y = point

                print(
                    f"  Point {point_id}: "
                    f"x={x.item():.1f}, "
                    f"y={y.item():.1f}"
                )
    '''
    '''
    进行关键点记录
    以及动作判别
    '''

    keypoints = results[0].keypoints

    if keypoints is not None and len(keypoints.xy) > 0:

        xy = keypoints.xy[0]

        # 左肩、左手腕
        left_shoulder = xy[5]
        left_wrist = xy[9]

        # 右肩、右手腕
        right_shoulder = xy[6]
        right_wrist = xy[10]

        # Y 坐标
        left_shoulder_y = left_shoulder[1]
        left_wrist_y = left_wrist[1]

        right_shoulder_y = right_shoulder[1]
        right_wrist_y = right_wrist[1]

        # 左手是否举起
        left_hand_raised = (
                left_wrist_y < left_shoulder_y - 20
        )

        # 右手是否举起
        right_hand_raised = (
                right_wrist_y < right_shoulder_y - 20
        )

        # 左手或者右手举起
        hand_raised = (
                left_hand_raised or right_hand_raised
        )

        if hand_raised:
            raise_hand_count += 1
        else:
            raise_hand_count = 0

        if raise_hand_count >= 3:
            action = "Raise Hand"
        else:
            action = "No Raise"

        print(action)

    '''
    最后进行绘制与opencv打开视频
    '''

    # 绘制人体框和骨架
    annotated_frame = results[0].plot()

    #将动作识别结果打印到视频画面上
    cv2.putText(
        annotated_frame,
        action,
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow(
        "YOLO26 Pose",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
