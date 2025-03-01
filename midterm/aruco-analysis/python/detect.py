# encoding: utf-8
import cv2
import numpy as np

def main():
    # 定义ArUco字典和检测参数
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 假设的相机参数（需要根据实际校准数据替换）
    camera_matrix = np.array([
        [1000, 0, 320],  # 焦距和光心（假设图像尺寸为640x480）
        [0, 1000, 240],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((5, 1))  # 假设无镜头畸变

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测ArUco标记
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            # 绘制检测到的标记
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # 遍历每个检测到的标记
            for i in range(len(ids)):
                # 估计姿态（标记实际边长以米为单位）
                marker_length = 0.05  # 根据实际打印的标记尺寸修改
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], marker_length, camera_matrix, dist_coeffs
                )

                # 绘制坐标轴（长度为3cm）
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                # 提取相对位置（相机坐标系）
                x = tvec[0][0][0]
                y = tvec[0][0][1]
                z = tvec[0][0][2]

                # 显示位置信息
                text = f"ID {ids[i][0]}: X:{x:.2f}m, Y:{y:.2f}m, Z:{z:.2f}m"
                cv2.putText(frame, text, (10, 30 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('ArUco Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()