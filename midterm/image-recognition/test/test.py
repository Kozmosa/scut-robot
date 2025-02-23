import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull
import logging

# init logging
logging.basicConfig(level=logging.DEBUG)

def get_contour_center(contour):
    M = cv.moments(contour)
    if M['m00'] == 0:
        return None
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    center_point = np.array([x, y])
    # convert to opencv point
    center_point = center_point.reshape(-1, 1, 2)
    return center_point

def computer_center_vertices_distance(center, vertices):
    distances = []
    for p in vertices:
        distance = np.linalg.norm(center - p)
        distances.append(distance)
    return distances

def compute_morphology_parameters(contour):
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    x, y, w, h = cv.boundingRect(contour)
    
    # compute solidity
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    solidity = area / hull_area
    
    # compute circularity
    minEnclosingCircle = cv.minEnclosingCircle(contour)
    center, radius = minEnclosingCircle
    circularity = area/(np.pi * radius * radius)
    
    # compute vertices count
    vertices = cv.approxPolyDP(contour, 0.02 * perimeter, True)
    vertices_count = len(vertices)
    hull = ConvexHull(vertices.reshape(-1, 2))
    vertices = vertices[hull.vertices]
    
    # compute extent
    extent = area / (w * h)
    
    return area, perimeter, solidity, circularity, vertices_count, extent, vertices

def compute_interior_angle(point1, vertex, point2):
    """
    计算由三个点 point1, vertex, point2 形成的内角 (以 vertex 为顶点).
    使用向量方法和点积公式计算角度.
    返回角度值 (角度制).
    """
    vector1 = point1 - vertex
    vector2 = point2 - vertex
    vector2_t = np.transpose(vector2)

    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0  # 如果向量长度为0，则角度为0 (避免除以0)

    dot_product = np.dot(vector1, vector2_t)
    cos_angle = dot_product / (norm_vector1 * norm_vector2)

    # 为了避免arccos的定义域问题，限制cos_angle在[-1, 1]范围内
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def check_contour_type(contour):
    x, y, w, h = cv.boundingRect(contour)
    area, perimeter, solidity, circularity, vertices_count, extent, vertices = compute_morphology_parameters(contour)
    
    if circularity > 0.95:
        return 'Circle'
    elif vertices_count == 3:
        return 'Triangle'
    elif vertices_count == 4:
        # rectangle or square
        aspect_ratio = w / h
        # 计算内角平均值
        angles = []
        for i in range(vertices_count):
            current_vertex = vertices[i]
            next_vertex = vertices[(i + 1) % vertices_count]
            previous_vertex = vertices[(i - 1) % vertices_count]
            angle = compute_interior_angle(previous_vertex, current_vertex, next_vertex)
            angles.append(angle)
        angles = np.array(angles)
        angles_variance = np.var(angles)
        
        if abs(aspect_ratio - 1) < 0.05:
            return 'Square'
        elif angles_variance < 1:
            return 'Rectangle'
        else:
            return 'Rhombus'
    elif vertices_count == 5:
        # pentagon or diamond shape
        center = get_contour_center(contour)
        distances = computer_center_vertices_distance(center, vertices)
        npdis = np.array(distances)
        dis_range = np.ptp(npdis)
        dis_variance = np.var(npdis)
        dis_mean = np.mean(npdis)
        if dis_variance < 1 and dis_range < 0.1 * dis_mean:
            return 'Pentagon'
        elif solidity > 0.99:
            return 'Half Circle'
        else:
            return 'Diamond'
    elif vertices_count == 6:
        center = get_contour_center(contour)
        distances = computer_center_vertices_distance(center, vertices)
        npdis = np.array(distances)
        dis_variance = np.var(npdis)
        if dis_variance < 1:
            return 'Hexagon'
        else:
            return 'Unknown'
    elif vertices_count == 7:
        return 'Arrow'
    elif vertices_count == 9:
        if solidity > 0.9:
            return 'Heart'
        else:
            return 'Lunar'
    elif vertices_count == 10:
        return 'Star'
    elif vertices_count == 12:
        return 'Cross'
    elif circularity > 0.7 and solidity > 0.98:
        return 'Ecplise'
    else:
        return 'Unknown'

def main():
    # main function
    # load image
    img = cv.imread('test.jpg')
    
    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite('gray.jpg', gray)
    
    # convert to binary
    _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
    cv.imwrite('binary.jpg', binary)
    
    # convert to binary inverse
    _, binary_inv = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)
    cv.imwrite('binary_inv.jpg', binary_inv)
    
    # 对binary_inv进行开运算，和闭运算
    kernel = np.ones((1,1), np.uint8)
    binary_inv = cv.morphologyEx(binary_inv, cv.MORPH_CLOSE, kernel)
    binary_inv = cv.morphologyEx(binary_inv, cv.MORPH_OPEN, kernel)
    cv.imwrite('binary_inv_morph.jpg', binary_inv)
    
    # detect contours
    contours, _ = cv.findContours(binary_inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for index, contour in enumerate(contours):
        logging.debug(f'Contour{index} has {len(contour)} points')
        # 计算形态学参数
        area, perimeter, solidity, circularity, vertices_count, extent, vertices = compute_morphology_parameters(contour)
        logging.debug(f'Area: {area}, Perimeter: {perimeter}, Solidity: {solidity}, Circularity: {circularity}, Vertices count: {vertices_count}, Extent: {extent}')
        
        # 检测形状
        contour_type = check_contour_type(contour)
        logging.debug(f'Contour{index} is {contour_type}')
        
        # 绘图
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # putText参数含义：图片，文本，坐标，字体，大小，颜色，粗细
        cv.putText(img, f'Contour{index}', (x, y-2), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # cv.putText(img, f'Circularity: {"%.2f" % circularity}', (x, y-12), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # cv.putText(img, f'Vertices count: {vertices_count}', (x, y-12), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv.putText(img, f'Type: {contour_type}', (x, y-12), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv.imwrite('bounding_rect.jpg', img)
    

# main
if __name__ == '__main__':
    main()