import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import collections

def detect_orange_red_rectangles_kalman():
    """
    使用 OpenCV 检测并框选摄像头视频帧中的橘红色矩形，使用卡尔曼滤波减少抖动。
    """
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 初始化卡尔曼滤波器
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 状态向量维度 4 (x, y, vx, vy)，观测向量维度 2 (x, y)

    # 状态转移矩阵 (假设匀速运动模型)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # 观测矩阵 (观测位置)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    # 过程噪声协方差矩阵 (假设位置和速度噪声较小)
    kf.Q = np.eye(4) * 1e-5

    # 观测噪声协方差矩阵 (假设位置观测噪声较大)
    kf.R = np.eye(2) * 0.5

    # 初始状态协方差矩阵 (初始状态不确定性较大)
    kf.P *= 1000.  # 或 np.eye(4) * 1000.

    kf.x = np.array([0., 0., 0., 0.])  # 初始状态：位置和速度都为 0

    last_center = None # 上一帧中心点

    # 中值滤波相关队列
    center_points_buffer = collections.deque(maxlen=20)
    width_buffer = collections.deque(maxlen=20)
    height_buffer = collections.deque(maxlen=20)
    
    # 存储上一状态
    last_w = 0
    last_h = 0
    lost_frames_count = 0
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        # 转换为 HSV 色彩空间
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred_hsv_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0) # 5x5 高斯核，标准差由函数自动计算

        # 橘红色范围 (保持不变)
        lower_orange_red = np.array([0, 100, 100])
        upper_orange_red = np.array([25, 255, 255])
        orange_red_mask = cv2.inRange(blurred_hsv_frame, lower_orange_red, upper_orange_red)

        # 合并颜色掩膜
        lower_nothing = np.array([0, 0, 0])
        upper_nothing = np.array([255,255,255])
        nothing_mask = cv2.inRange(blurred_hsv_frame, lower_nothing, upper_nothing)
        multi_color_mask = orange_red_mask
        
        
        # 形态学操作去除噪点
        kernel = np.ones((5,5), np.uint8)
        # orange_red_mask = cv2.erode(multi_color_mask, kernel, iterations=2)
        # orange_red_mask = cv2.dilate(multi_color_mask, kernel, iterations=2)
        multi_color_mask = cv2.morphologyEx(multi_color_mask, cv2.MORPH_OPEN, kernel)
        multi_color_mask = cv2.morphologyEx(multi_color_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(multi_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours_frame = frame.copy()
        cv2.drawContours(contours_frame, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', contours_frame)

        detected_rectangles = []

        for contour in contours:
            # 轮廓近似，判断是否为矩形
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: # 避免除零错误
                continue
            solidity = float(area) / hull_area
            if solidity < 0.1: # 凸度阈值，可以根据实际情况调整
                continue # 凸度过低，跳过该轮廓
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                area = cv2.contourArea(contour)
                
                # 0.85 <= aspect_ratio <= 1.15 and
                if area > 2000:
                    detected_rectangles.append(((x, y, w, h), contour))

        best_rectangle = None
        max_area = 0
        for rect_info in detected_rectangles:
            (x, y, w, h), contour = rect_info
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                best_rectangle = rect_info

        if best_rectangle:
            if lost_frames_count != 0: # 重新检测到目标，清空滤波相关队列
                width_buffer.clear()
                height_buffer.clear()
                center_points_buffer.clear()
                lost_frames_count = 0 # 重置丢帧计数器
                
            (x_bbox, y_bbox, w_bbox, h_bbox), contour = best_rectangle
            center_x_measured = x_bbox + w_bbox // 2
            center_y_measured = y_bbox + h_bbox // 2
            measurement = np.array([center_x_measured, center_y_measured])


            # 卡尔曼滤波预测和更新
            kf.predict()
            kf.update(measurement)

            # 获取滤波后的中心点位置
            center_x_filtered = int(kf.x[0])
            center_y_filtered = int(kf.x[1])
            
            # 增加中值滤波
            # **中值滤波平滑中心点坐标**
            center_points_buffer.append((center_x_filtered, center_y_filtered)) # 将卡尔曼滤波后的中心点加入队列
            if center_points_buffer: # 队列不为空时才进行中值滤波
                x_coords = [p[0] for p in center_points_buffer]
                y_coords = [p[1] for p in center_points_buffer]
                center_x_median = int(np.median(x_coords))
                center_y_median = int(np.median(y_coords))
                center_x_filtered = center_x_median # 使用中值滤波后的 x 坐标
                center_y_filtered = center_y_median # 使用中值滤波后的 y 坐标
            
            # **中值滤波平滑矩形框宽高**
            width_buffer.append(w_bbox)
            height_buffer.append(h_bbox)
            w_filtered = w_bbox # 默认使用当前帧宽度
            h_filtered = h_bbox # 默认使用当前帧高度

            if width_buffer and height_buffer: # 队列不为空时才进行中值滤波
                w_median = int(np.median(list(width_buffer)))
                h_median = int(np.median(list(height_buffer)))
                w_filtered = w_median # 使用宽度中值滤波结果
                h_filtered = h_median # 使用高度中值滤波结果

            # 使用滤波后的中心点重新计算矩形框 (保持原始宽高)
            x_filtered = center_x_filtered - w_filtered // 2
            y_filtered = center_y_filtered - h_filtered // 2
            
            last_w = w_filtered
            last_h = h_filtered
            
            cv2.rectangle(frame, (x_filtered, y_filtered), (x_filtered + w_filtered, y_filtered + h_filtered), (0, 255, 0), 2)
            
            last_center = (center_x_filtered, center_y_filtered)


        else:
            kf.predict() # 即使没有检测到，也进行预测，保持状态更新
            last_center = None
            lost_frames_count += 1
            if lost_frames_count < 10:
                center_x_filtered = int(kf.x[0])
                center_y_filtered = int(kf.x[1])
                x_filtered = center_x_filtered - last_w // 2
                y_filtered = center_y_filtered - last_h // 2
                cv2.rectangle(frame, (x_filtered, y_filtered), (x_filtered + last_w, y_filtered + last_h), (0, 255, 0), 2)
            else:
                # 清空中值滤波相关队列
                width_buffer.clear()
                height_buffer.clear()
                center_points_buffer.clear()
                
        # 显示结果帧
        cv2.imshow('Orange-Red Rectangle Detection with Kalman Filter', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头和窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_orange_red_rectangles_kalman()