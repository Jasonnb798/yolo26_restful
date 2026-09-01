import cv2
from ultralytics import YOLO
import numpy as np

#膝盖夹角计算函数
def calculate_angle(a,b,c):
    """
    计算三个点 A-B-C 的夹角。
    B 是角的顶点。
    """

    a = a.cpu().numpy()
    b = b.cpu().numpy()
    c = c.cpu().numpy()

    # BA 和 BC 两个向量
    ba = a - b
    bc = c - b

    # 计算余弦值
    cosine_angle = np.dot(ba, bc) / (
            np.linalg.norm(ba) *
            np.linalg.norm(bc)
    )

    # 防止浮点误差导致 arccos 出错
    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    # 弧度转角度
    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return angle



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
#raise_hand_count = 0
raise_hand_count = {}
#action = "None"
action_states = {}
# 下蹲连续帧计数
squat_count = {}

# 膝关节角度小于这个值时，认为腿弯曲比较明显
SQUAT_ANGLE_THRESHOLD = 120

posture_states = {}

while True:

    ret, frame = cap.read()

    if not ret:
        print("读取视频帧失败")
        break

    # 姿态估计
    '''
    results = model(
        frame,
        imgsz=640,
        device=0,
        conf=0.25,
        verbose=True
    )
    '''

    #进行person追踪
    results = model.track(
        frame,
        imgsz=960,
        device=0,
        conf=0.25,
        persist=True,
        verbose=False
    )

    # 绘制人体框和骨架
    annotated_frame = results[0].plot()

    #打印关键点数据到terminal
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
    进行关键点记录
    以及动作判别
    '''

    keypoints = results[0].keypoints
    #获取track Id
    boxes = results[0].boxes
    if boxes.id is not None:
        track_ids = boxes.id

    if (
            keypoints is not None
            and len(keypoints.xy) > 0
            and boxes.id is not None
    ):

        for person_index, person in enumerate(keypoints.xy):
            track_id = int(track_ids[person_index])
            xy = person

            # 左肩、左手腕
            left_shoulder = xy[5]
            left_wrist = xy[9]

            # 右肩、右手腕
            right_shoulder = xy[6]
            right_wrist = xy[10]

            # 左腿关键点
            left_hip = xy[11]
            left_knee = xy[13]
            left_ankle = xy[15]

            # 右腿关键点
            right_hip = xy[12]
            right_knee = xy[14]
            right_ankle = xy[16]

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

            if track_id not in raise_hand_count:
                raise_hand_count[track_id] = 0

            if hand_raised:
                raise_hand_count[track_id] += 1
            else:
                raise_hand_count[track_id] = 0

            if raise_hand_count[track_id] >= 5:
                action_states[track_id] = "Raise Hand"
            else:
                action_states[track_id] = "No Raise"
            action = action_states[track_id]

            #计算角度
            left_knee_angle = None
            right_knee_angle = None
            left_knee_angle = calculate_angle(
                left_hip,
                left_knee,
                left_ankle
            )
            right_knee_angle = calculate_angle(
                right_hip,
                right_knee,
                right_ankle
            )

            #判断是否下蹲
            squat_detected = False
            if (
                    left_knee_angle is not None
                    and right_knee_angle is not None
            ):
                if (
                        left_knee_angle < SQUAT_ANGLE_THRESHOLD
                        and right_knee_angle < SQUAT_ANGLE_THRESHOLD
                ):
                    squat_detected = True

            #下蹲计数
            if track_id not in squat_count:
                squat_count[track_id] = 0

            if squat_detected:
                squat_count[track_id] += 1
            else:
                squat_count[track_id] = 0

            if squat_count[track_id] >= 5:
                posture_states[track_id] = "Squat"
            else:
                posture_states[track_id] = "Stand"

            #print(action)

            box = boxes.xyxy[person_index]
            x1, y1, x2, y2 = map(int, box)

            # 将动作识别结果打印到视频画面上
            cv2.putText(
                annotated_frame,
                f"ID: {track_id} {action}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"Posture: {posture_states[track_id]}",
                (x1, y1 + 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    '''
    最后opencv打开视频
    '''
    cv2.imshow(
        "YOLO26 Pose",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
