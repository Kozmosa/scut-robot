import email
import cv2 as cv
import numpy as np
import logging as log
import datetime
from tqdm import tqdm
import queue
import logging.handlers


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

    # 定义偏青色的蓝色HSV范围 (根据RGB(78,247,236)转换结果调整)
    lower_cyan_blue = np.array([50, 80, 80])  # 调整了S和V的下限，排除暗色
    upper_cyan_blue = np.array([255, 255, 255]) # 调整H的上限，更偏向青色

    # 生成青蓝色掩膜
    mask = cv.inRange(hsv, lower_cyan_blue, upper_cyan_blue)
    
    # 形态学处理：闭运算填充孔洞，开运算去除噪声
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 筛选面积较大的轮廓
    min_area = 30
    result = image.copy()

    armors = []

    for coutourCount, cnt in enumerate(contours):
        if coutourCount > 0:
            log.debug(f'---Coutour{coutourCount-1} End---')
        area = cv.contourArea(cnt)
        log.debug(f'---Coutour{coutourCount} Start---')
        # 计算轮廓的外接矩形的面积
        x, y, w, h = cv.boundingRect(cnt)
        rect_area = w * h
        wh_ratio = w / h
        std_wh_ratio = float(7) / 23
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
        
            
        if extent < 0.65 or abs(wh_ratio - std_wh_ratio) > 0.1:
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
    
    # 过滤不成对光柱
    paired_armors = []
    pairs = []
    paired_index = set()
    armor_centers = [getCoutourCenter(armor) for armor in armors]
    armor_bounding_rects = [cv.boundingRect(armor) for armor in armors]
    armor_radii = [np.sqrt(w**2 + h**2) / 2 for _, _, w, h in armor_bounding_rects]

    for i in range(len(armors)):
        center1 = armor_centers[i]
        radius1 = armor_radii[i]
        for j in range(len(armors)):
            if i == j or j in paired_index:
                continue
            center2 = armor_centers[j]
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            if distance < 5.5 * radius1:
                log.debug(f'Pair {i} and {j}. distance: {distance}, close to {distance / float(radius1)}x radius.')
                paired_index.add(i)
                paired_index.add(j)
                pairs.append((armors[i], armors[j]))
                break
    armors = paired_armors
        
    for count, pair in enumerate(pairs, start=1):
        for index, armor in enumerate(pair):
            center = getCoutourCenter(armor)
            # 参数含义：图像，圆心坐标，半径，颜色，填充
            cv.circle(result, (center[0], center[1]), 2, (0, 0, 255), -1)
            
            # 绘制外接矩形
            x, y, w, h = cv.boundingRect(armor)
            cv.rectangle(result, (x, y), (x+w, y+h), (255,255,255), 2)
            
            # 在外接矩形右上角标注灯柱序号
            # 参数含义：图像，文本，坐标，字体，字号，颜色，字体厚度
            cv.putText(result, str(count), (x+w, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            log.debug(f'---Draw {str(count)},{str(index)}---')
        
    
    # 打印帧序号
    # 参数含义：图像，文本，坐标，字体，字号，颜色，字体厚度
    # 坐标格式：(x, y)
    cv.putText(result, str(frame_id), (960, 200), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
    log.debug(f'---Frame End---')
        
    return mask, result, armors


# Configure Logging
# def setup_logging():
#     """配置异步日志处理并返回 logger."""
#     log_queue = queue.Queue(-1)  # 创建一个队列用于日志记录

#     # 创建一个 QueueHandler 并将其添加到 root logger
#     queue_handler = logging.handlers.QueueHandler(log_queue)
#     root_logger = logging.getLogger() # 获取 root logger
#     root_logger.addHandler(queue_handler)
#     root_logger.setLevel(logging.DEBUG) # 设置 root logger 的日志级别

#     # 创建一个 FileHandler 来处理队列中的日志记录
#     now = datetime.datetime.now()
#     formatted_time = now.strftime("%Y-%m-%d_%H:%M")
#     log_file_handler = logging.FileHandler(f'./log/app_{formatted_time}.log', 'w')
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S') # 日志格式
#     log_file_handler.setFormatter(formatter)

#     # 创建并启动 QueueListener，将日志从队列发送到 FileHandler
#     listener = logging.handlers.QueueListener(log_queue, log_file_handler)
#     listener.start()

#     logger = logging.getLogger(__name__) # 获取当前模块的 logger
#     return logger, listener

now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H:%M")

log.basicConfig(level=log.DEBUG,  # 设置最低日志级别为 DEBUG
                    filename=f'./log/app_{formatted_time}.log',    # 日志输出到文件 app.log
                    filemode='w',          # 覆盖写入日志文件
                    format='%(asctime)s - %(levelname)s - %(message)s', # 日志格式
                    datefmt='%Y-%m-%d %H:%M:%S') # 日期时间格式

# log, listener = setup_logging()

if __name__ == '__main__':
    # 读取图片: test.jpg
    image = cv.imread('test.png')
    if image is None:
        log.error('Error: Cannot open image file.')
        exit(1)
    
    # 将遮罩和结果分别存储于mask.jpg和result.jpg
    mask, result, armors = drawArmorRect(image)
    cv.imwrite('mask.jpg', mask)
    cv.imwrite('result.jpg', result)
    log.info('Image processing complete.')