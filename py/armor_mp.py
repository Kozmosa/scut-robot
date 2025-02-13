import cv2 as cv
import numpy as np
import logging
import datetime
from tqdm import tqdm
import queue
import logging.handlers
import threading
import concurrent.futures
import os # 导入os模块，用于获取CPU核心数

DRAW_RED_RECT = False

def getCoutourCenter(contour):
    M = cv.moments(contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centerPoint = np.array([cx, cy])
    return centerPoint

def drawRedRect(cnt, result):
    if DRAW_RED_RECT:
        log.debug('Draw red rectangle')
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

def drawArmorRect(image, frame_id=0):
    log.debug(f'---Frame{str(frame_id)} Start---')
    # 读取图像
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # 定义红色的HSV范围（示例值，需根据实际情况调整）
    lower_red1 = np.array([0, 110, 60])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 110, 60])
    upper_red2 = np.array([180, 255, 255])

    # 生成掩膜
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask = cv.bitwise_or(mask1, mask2)

    # 形态学处理：闭运算填充孔洞，开运算去除噪声
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 筛选面积较大的轮廓
    min_area = 50
    result = image.copy()

    armors = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        log.debug(f'---Coutour Start---')
        # 计算轮廓的外接矩形的面积
        x, y, w, h = cv.boundingRect(cnt)
        rect_area = w * h
        wh_ratio = w / h
        log.debug(f'Width: {w}, Height: {h}, Ratio: {wh_ratio}')
        # 计算轮廓的面积与外接矩形面积的比值
        extent = float(area) / rect_area
        log.debug(f'Area: {area}, Rect Area: {rect_area}, Extent: {extent}')

        # 计算凸度
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        log.debug(f'Solidity: {solidity}')

        # 计算圆形度
        perimeter = cv.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        log.debug(f'Circularity: {circularity}')

        # 轮廓近似
        epsilon = 0.02 * perimeter
        approx_contour = cv.approxPolyDP(cnt, epsilon, True)
        vertices_count = len(approx_contour)
        log.debug(f'Vertices Count: {vertices_count}')


        if extent < 0.7 and wh_ratio > 0.75:
            log.debug('Extent < 0.7, skip')
            log.debug('Width/Height > 0.75, skip')
            drawRedRect(cnt, result)
            continue
        elif solidity < 0.8:
            log.debug('Solidity < 0.8, skip')
            drawRedRect(cnt, result)
            continue
        elif circularity < 0.3:
            log.debug('Circularity < 0.3, skip')
            drawRedRect(cnt, result)
            continue
        elif vertices_count > 12:
            log.debug('Vertices Count > 12, skip')
            drawRedRect(cnt, result)
            continue
        else:
            log.debug('Extent >= 0.7, keep')
            log.debug('Solidity >= 0.8, keep')
            log.debug('Circularity >= 0.3, keep')
            log.debug('Vertices Count <= 12, keep')


        if area > min_area:
            armors.append(cnt)
            count = len(armors)
            # 绘制轮廓
            # cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)

            # 绘制外接矩形
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # 在外接矩形右上角标注灯柱序号
            # 参数含义：图像，文本，坐标，字体，字号，颜色，字体厚度
            cv.putText(result, str(count), (x+w, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            log.debug(f'---Coutour{count} End---')


    for armor in armors:
        center = getCoutourCenter(armor)
        # 参数含义：图像，圆心坐标，半径，颜色，填充
        cv.circle(result, (center[0], center[1]), 2, (0, 0, 255), -1)


    # 打印帧序号
    # 参数含义：图像，文本，坐标，字体，字号，颜色，字体厚度
    # 坐标格式：(x, y)
    cv.putText(result, str(frame_id), (960, 200), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    log.debug(f'---Frame End---')

    return mask, result, armors


# Configure Logging
def setup_logging():
    """配置异步日志处理并返回 logger."""
    log_queue = queue.Queue(-1)  # 创建一个队列用于日志记录

    # 创建一个 QueueHandler 并将其添加到 root logger
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root_logger = logging.getLogger() # 获取 root logger
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.DEBUG) # 设置 root logger 的日志级别

    # 创建一个 FileHandler 来处理队列中的日志记录
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H:%M")
    log_file_handler = logging.FileHandler(f'./log/app_{formatted_time}.log', 'w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') # 日志格式
    log_file_handler.setFormatter(formatter)

    # 创建并启动 QueueListener，将日志从队列发送到 FileHandler
    listener = logging.handlers.QueueListener(log_queue, log_file_handler)
    listener.start()

    logger = logging.getLogger(__name__) # 获取当前模块的 logger
    return logger, listener

log, listener = setup_logging()

def process_frame(frame_id, frame, output_queue):
    """处理单帧图像的函数，用于多线程处理，并将结果放入输出队列"""
    mask, result, armors = drawArmorRect(frame, frame_id=frame_id)
    # 在处理线程中进行 resize
    mask_resized = cv.resize(mask, frame_size)
    result_resized = cv.resize(result, frame_size)
    output_queue.put((frame_id, mask_resized, result_resized)) # 放入 resize 后的帧

def writer_thread_func(mask_writer, result_writer, output_queue, total_frames, frame_size):
    """写入线程函数，从队列中读取帧并按帧ID顺序写入视频"""
    frame_id_to_write = 0
    frames_written = 0
    processed_frames = {} # 存储已处理帧的字典，key为frame_id

    pbar_write = tqdm(total=total_frames, desc="写入视频帧", position=1, leave=True) # 创建写入进度条

    while frames_written < total_frames:
        frame_id, mask, result = output_queue.get() # 从队列中获取帧

        processed_frames[frame_id] = (mask, result) # 存储处理后的帧

        while frame_id_to_write in processed_frames:
            mask_to_write, result_to_write = processed_frames.pop(frame_id_to_write)

            
            mask_writer.write(mask_to_write)
            result_writer.write(result_to_write)

            frame_id_to_write += 1
            frames_written += 1
            pbar_write.update(1)
            output_queue.task_done() # 通知队列任务完成

    pbar_write.close()
    mask_writer.release()
    result_writer.release()
    log.info('Video writing complete.')


if __name__ == '__main__':
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H:%M")

    # 读取视频: test.mp4
    cap = cv.VideoCapture('test.mp4')
    if not cap.isOpened():
        log.error('Error: Cannot open video file.')
        exit(1)

    # 将遮罩和结果分别存储于mask.mp4和result.mp4
    fourcc = cv.VideoWriter_fourcc(*'mp4v') # 使用 avc 编解码器
    fps = 60.0 # 帧率 60 FPS
    frame_width = 1920 # 帧宽度
    frame_height = 1080 # 帧高度
    frame_size = (frame_width, frame_height) # 帧尺寸 (宽度, 高度)

    mask_writer = cv.VideoWriter(f'mask_{formatted_time}.mp4', fourcc, fps, frame_size, isColor=False)
    result_writer = cv.VideoWriter(f'result_{formatted_time}.mp4', fourcc, fps, frame_size, isColor=True)

    if not mask_writer.isOpened():
        log.error("Error: Could not open mask_xxx.mp4 video file for writing!") # 更明确的错误信息
        exit(1)
    if not result_writer.isOpened():
        log.error("Error: Could not open result_xxx.mp4 video file for writing!") # 更明确的错误信息
        exit(1)

    # 获取视频总帧数
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames < 0:
        log.error("无法获取视频总帧数，进度条可能无法准确显示。")
        total_frames = 0
    else:
        log.info(f"视频总帧数: {total_frames}")

    # 初始化 tqdm 进度条
    pbar_read = tqdm(total=total_frames, desc="读取视频帧", position=0, leave=True) # 创建读取进度条

    output_queue = queue.Queue(maxsize=150) # 创建帧队列
    writer_thread = threading.Thread(target=writer_thread_func, args=(mask_writer, result_writer, output_queue, total_frames, frame_size))
    writer_thread.daemon = True # 设置为守护线程，主线程退出时自动退出
    writer_thread.start() # 启动写入线程

    # 线程池设置
    num_threads = os.cpu_count() if os.cpu_count() else 4 # 根据CPU核心数自动调整线程数
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)


    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            log.info('End of video.')
            break

        # 提交帧处理任务到线程池, 并将output_queue传递给处理函数
        executor.submit(process_frame, frame_id, frame, output_queue)

        pbar_read.update(1) # 更新读取进度条
        frame_id += 1

    executor.shutdown(wait=False) # 关闭线程池, 设置wait=False，不等待所有任务完成立即返回
    output_queue.join() # 阻塞主线程，直到output_queue队列为空，即所有帧都被写入线程处理完毕
    pbar_read.close() # 关闭读取进度条

    writer_thread.join() # 等待写入线程结束 (虽然设置为daemon线程，但为了更安全地关闭，还是显式join一下)
    cap.release()
    cv.destroyAllWindows()
    log.info('Video processing complete.')
    listener.stop() # 停止日志监听器