import cv2 as cv
import numpy as np
import logging as log
import datetime
from tqdm import tqdm

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

def drawArmorRect_cpu(image):
    # 读取图像
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # 定义红色的HSV范围
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    # 生成掩膜
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask = cv.bitwise_or(mask1, mask2)

    # 形态学处理
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 筛选轮廓
    min_area = 50
    result = image.copy()
    armors = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        rect_area = w * h
        wh_ratio = w / h
        log.debug(f'Width: {w}, Height: {h}, Ratio: {wh_ratio}')
        extent = float(area) / rect_area
        log.debug(f'Area: {area}, Rect Area: {rect_area}, Extent: {extent}')
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        log.debug(f'Solidity: {solidity}')
        perimeter = cv.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        log.debug(f'Circularity: {circularity}')
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
            cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv.putText(result, str(count), (x+w, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for armor in armors:
        center = getCoutourCenter(armor)
        cv.circle(result, (center[0], center[1]), 5, (0, 0, 255), -1)

    return mask, result, armors


def drawArmorRect_cuda(image):
    # 将图像上传到GPU
    gpu_image = cv.cuda_GpuMat()
    gpu_image.upload(image)

    # 颜色转换在GPU上进行
    gpu_hsv = cv.cuda.cvtColor(gpu_image, cv.COLOR_BGR2HSV)

    # 定义红色的HSV范围
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    # 生成掩膜在GPU上进行
    gpu_mask1 = cv.cuda.inRange(gpu_hsv, lower_red1, upper_red1)
    gpu_mask2 = cv.cuda.inRange(gpu_hsv, lower_red2, upper_red2)
    gpu_mask = cv.cuda.bitwise_or(gpu_mask1, gpu_mask2)

    # 形态学处理在GPU上进行
    kernel = np.ones((5,5), np.uint8)
    gpu_kernel = cv.cuda_GpuMat()
    gpu_kernel.upload(kernel)
    gpu_mask = cv.cuda.morphologyEx(gpu_mask, cv.MORPH_CLOSE, gpu_kernel)
    gpu_mask = cv.cuda.morphologyEx(gpu_mask, cv.MORPH_OPEN, gpu_kernel)

    # 轮廓查找和筛选仍然在CPU上进行，因为cv.cuda模块的轮廓查找功能不如CPU版本完善，且轮廓数据量相对较小
    mask = gpu_mask.download() # 将掩膜下载到CPU
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 筛选轮廓
    min_area = 50
    result = image.copy() # result 仍然是CPU上的图像
    armors = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        rect_area = w * h
        wh_ratio = w / h
        log.debug(f'Width: {w}, Height: {h}, Ratio: {wh_ratio}')
        extent = float(area) / rect_area
        log.debug(f'Area: {area}, Rect Area: {rect_area}, Extent: {extent}')
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        log.debug(f'Solidity: {solidity}')
        perimeter = cv.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        log.debug(f'Circularity: {circularity}')
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
            cv.drawContours(result, [cnt], -1, (0, 255, 0), 2) # 绘图操作在CPU上
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2) # 绘图操作在CPU上
            cv.putText(result, str(count), (x+w, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # 绘图操作在CPU上

    for armor in armors:
        center = getCoutourCenter(armor)
        cv.circle(result, (center[0], center[1]), 5, (0, 0, 255), -1) # 绘图操作在CPU上

    return mask, result, armors # mask 是CPU numpy 数组, result 是 CPU numpy 数组, armors 是 contours 列表


# Configure Logging
now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H:%M")

log.basicConfig(level=log.INFO,
                    filename=f'./log/app_{formatted_time}.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    # 读取视频: test.mp4
    cap = cv.VideoCapture('test.mp4')
    if not cap.isOpened():
        log.error('Error: Cannot open video file.')
        exit(1)

    # 将遮罩和结果分别存储于mask.mp4和result.mp4
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    frame_width = 1920
    frame_height = 1080
    frame_size = (frame_width, frame_height)

    mask_writer = cv.VideoWriter(f'mask_cuda_{formatted_time}.mp4', fourcc, fps, frame_size, isColor=False) # 修改了文件名以区分CPU和CUDA版本
    result_writer = cv.VideoWriter(f'result_cuda_{formatted_time}.mp4', fourcc, fps, frame_size, isColor=True) # 修改了文件名以区分CPU和CUDA版本

    if not mask_writer.isOpened():
        log.error("Error: Could not open mask_cuda_xxx.mp4 video file for writing!")
        exit(1)
    if not result_writer.isOpened():
        log.error("Error: Could not open result_cuda_xxx.mp4 video file for writing!")
        exit(1)

    # 获取视频总帧数
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames < 0:
        log.error("无法获取视频总帧数，进度条可能无法准确显示。")
        total_frames = 0
    else:
        log.info(f"视频总帧数: {total_frames}")

    # 初始化 tqdm 进度条
    pbar = tqdm(total=total_frames, desc="处理视频帧 (CUDA)") # 修改进度条描述以区分CPU和CUDA版本

    # 从视频中读取帧
    once = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            log.info('End of video.')
            break
        # 记录帧形状
        log.info(f'Frame shape: {frame.shape}')
        mask, result, armors = drawArmorRect_cuda(frame) # 使用 CUDA 加速版本
        if not once:
            cv.imwrite('mask_cuda.jpg', mask) # 修改了文件名以区分CPU和CUDA版本
            cv.imwrite('result_cuda.jpg', result) # 修改了文件名以区分CPU和CUDA版本
            once = 1

        # 缩放 mask 和 result 帧到 frame_size
        mask_resized = cv.resize(mask, frame_size)
        result_resized = cv.resize(result, frame_size)

        mask_writer.write(mask_resized)
        result_writer.write(result_resized)

        pbar.update(1)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    mask_writer.release()
    result_writer.release()
    cv.destroyAllWindows()
    log.info('Video processing complete with CUDA.') # 修改日志信息以区分CPU和CUDA版本