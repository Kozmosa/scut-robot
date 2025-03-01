import cv2
import numpy as np

def main():
    # 设置生成参数
    # 与detect.py中参数保持一致
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_id = 1  # 设置标记ID（0-49）
    marker_size = 400  # 生成的图片像素尺寸
    border_bits = 1  # 边框宽度

    # 生成标记图像
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size,
                                               borderBits=border_bits)

    # 保存生成的标记
    cv2.imwrite(f"aruco_marker_{marker_id}.png", marker_img)
    # 显示生成的标记
    cv2.imshow("Generated Marker", marker_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()