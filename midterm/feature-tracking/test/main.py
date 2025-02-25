import cv2
import numpy as np

def detect_orange_red_rectangles():
    """
    使用 OpenCV 检测并框选摄像头视频帧中的橘红色矩形。
    """
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        # 转换为 HSV 色彩空间
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义橘红色范围的 HSV 值
        lower_orange_red = np.array([0, 100, 100])  #  接近红色的橘红色下限
        upper_orange_red = np.array([255, 255, 255]) # 橘红色上限， 可以根据需要调整 H 值上限

        orange_red_mask = cv2.inRange(hsv_frame, lower_orange_red, upper_orange_red)


        # 形态学操作去除噪点
        kernel = np.ones((5, 5), np.uint8)
        orange_red_mask = cv2.erode(orange_red_mask, kernel, iterations=1)
        orange_red_mask = cv2.dilate(orange_red_mask, kernel, iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(orange_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 轮廓近似，判断是否为矩形
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) == 4: # 矩形有四个顶点
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                # if 0.8 <= aspect_ratio <= 1.2: # 宽高比率限制，更像正方形或者接近正方形的矩形
                if True: # 不限制宽高比率
                    area = cv2.contourArea(contour)
                    if area > 1000: # 面积过滤，去除过小的噪点轮廓
                        # 框选矩形
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # 显示结果帧
        cv2.imshow('Orange-Red Rectangle Detection', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头和窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_orange_red_rectangles()