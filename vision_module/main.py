import numpy as np
import cv2 as cv
import gradio as gr

import logging
from datetime import datetime
from queue import Queue
import threading
import time

log_queue = Queue()

# 自定义日志处理器，将日志写入队列
class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

# 获取摄像头画面
def get_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("无法从摄像头读取画面")
            break
        _, img_encoded = cv.imencode('.jpg', frame)
        yield img_encoded.tobytes()
        time.sleep(0.03)  # 限制帧率，减少CPU负载

# 读取日志队列中的日志
def get_logs():
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return "\n".join(logs[-100:])  # 仅保留最近100条日志

# Gradio 界面
def live_camera():
    while True:
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return frame

def update_logs():
    return get_logs()

# 日志初始化
# 获得logger，日志存储在log目录下的“日期-时间.log”文件中
logging.basicConfig(filename=f'log/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(QueueHandler())


# 摄像头初始化
cap = cv.VideoCapture(0)
if not cap.isOpened():
    logger.error("无法打开摄像头")

# 相机校准函数
def calibrate_camera():
    # 设置棋盘格的规格
    chessboardSize = (9, 6)
    # 设置棋盘格的物理尺寸
    chessboardSquareSize = 0.0245
    # 设置棋盘格的物理坐标
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2) * chessboardSquareSize
    # 设置图像中的物理坐标
    objPoints = []
    # 设置图像中的像素坐标
    imgPoints = []
    # 设置图像的宽高
    imageSize = None
    # 设置相机的内参
    cameraMatrix = None
    # 设置相机的畸变参数
    distCoeffs = None
    # 设置图像路径
    imagePath = 'chessboard.jpg'
    # 读取图像
    image = cv.imread(imagePath)
    # 设置图像的宽高
    imageSize = (image.shape[1], image.shape[0])
    # 转换为灰度图
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 查找棋盘格的角点
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    # 如果找到了角点
    if ret:
        # 添加物理坐标
        objPoints.append(objp)
        # 添加像素坐标
        imgPoints.append(corners)
        # 绘制角点
        cv.drawChessboardCorners(image, chessboardSize, corners, ret)
        # 查找相机的内参和畸变参数
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, imageSize, None, None)
        # 保存相机的内参和畸变参数
        np.savez('camera.npz', cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
        # 显示图像
        cv.imshow('image', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    return cameraMatrix, distCoeffs

if __name__ == '__main__':
    # cameraMatrix, distCoeffs = calibrate_camera()
    # logger.info(f'cameraMatrix: {cameraMatrix}')
    # logger.info(f'distCoeffs: {distCoeffs}')
    
    
    # 启动gradio服务，分左右两边，左边显示当前摄像头接受的画面，右边显示启动到现在的所有日志信息
    with gr.Blocks() as app:
        gr.Markdown("## 实时摄像头 & 日志监控")
        with gr.Row():
            cam_feed = gr.Image(label="摄像头画面")
            log_display = gr.Textbox(label="日志信息", interactive=False, lines=20)

        # 参数含义：函数名，参数，返回值，更新频率
        app.load(live_camera, None, cam_feed)
        app.load(update_logs, None, log_display)
    
    app.launch(server_name="0.0.0.0", server_port=8080)
    cap.release()
    cv.destroyAllWindows()