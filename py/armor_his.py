"""
# 读取图像
image = cv.imread('image.png')
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# 定义红色的HSV范围（示例值，需根据实际情况调整）
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
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
    # 计算轮廓的外接矩形的面积
    x, y, w, h = cv.boundingRect(cnt)
    rect_area = w * h
    wh_ratio = w / h
    log.debug(f'Width: {w}, Height: {h}, Ratio: {wh_ratio}')
    # 计算轮廓的面积与外接矩形面积的比值
    extent = float(area) / rect_area
    log.debug(f'Area: {area}, Rect Area: {rect_area}, Extent: {extent}')
    if extent < 0.7 and wh_ratio > 0.75:
        log.debug('Extent < 0.7, skip')
        log.debug('Width/Height > 0.75, skip')
        log.debug('Draw red rectangle')
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        continue
    else:
        log.debug('Extent >= 0.7, keep')
        
    
    if area > min_area:
        armors.append(cnt)
        count = len(armors)
        # 绘制轮廓
        cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)
        # 可选：绘制外接矩形
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # 在外接矩形右上角标注灯柱序号
        # 参数含义：图像，文本，坐标，字体，字号，颜色，字体厚度
        cv.putText(result, str(count), (x+w, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        
for armor in armors:
    center = getCoutourCenter(armor)
    cv.circle(result, (center[0], center[1]), 5, (0, 0, 255), -1)

# 显示结果
# cv.imshow('Mask', mask)
# cv.imshow('Result', result)
"""