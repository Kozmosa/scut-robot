#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

bool DRAW_RED_RECT = false;

Point getCoutourCenter(const vector<Point>& contour) {
    Moments M = moments(contour);
    if (M.m00 == 0) {
        return Point(-1, -1); // Indicate no center found
    }
    int cx = static_cast<int>(M.m10 / M.m00);
    int cy = static_cast<int>(M.m01 / M.m00);
    return Point(cx, cy);
}

void drawRedRect(const vector<Point>& cnt, Mat& result) {
    if (DRAW_RED_RECT) {
        Rect rect = boundingRect(cnt);
        rectangle(result, rect, Scalar(0, 0, 255), 2);
    }
}

tuple<Mat, Mat, vector<vector<Point>>> drawArmorRect(Mat& image, int frame_id = 0) {
    // 读取图像 (Image reading is already done in main function)

    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // 定义偏青色的蓝色HSV范围 (根据RGB(78,247,236)转换结果调整)
    Scalar lower_cyan_blue = Scalar(50, 80, 80);  // 调整了S和V的下限，排除暗色
    Scalar upper_cyan_blue = Scalar(255, 255, 255); // 调整H的上限，更偏向青色

    // 生成青蓝色掩膜
    Mat mask;
    inRange(hsv, lower_cyan_blue, upper_cyan_blue, mask);

    // 形态学处理：闭运算填充孔洞，开运算去除噪声
    Mat kernel = Mat::ones(5, 5, CV_8U);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    // 查找轮廓
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 筛选面积较大的轮廓
    double min_area = 30;
    Mat result = image.clone();
    vector<vector<Point>> armors;

    for (size_t coutourCount = 0; coutourCount < contours.size(); ++coutourCount) {
        vector<Point>& cnt = contours[coutourCount];
        double area = contourArea(cnt);
        // 计算轮廓的外接矩形的面积
        Rect rect = boundingRect(cnt);
        double rect_area = rect.width * rect.height;
        double wh_ratio = (double)rect.width / rect.height;
        double std_wh_ratio = 7.0 / 23.0;
        // 计算轮廓的面积与外接矩形面积的比值
        double extent = area / rect_area;

        // 计算凸度
        vector<Point> hull;
        convexHull(cnt, hull);
        double hull_area = contourArea(hull);
        double solidity = (hull_area > 0) ? (area / hull_area) : 0;

        // 计算圆形度
        double perimeter = arcLength(cnt, true);
        double circularity = (perimeter > 0) ? ((4 * M_PI * area) / (perimeter * perimeter)) : 0;

        // 轮廓近似
        double epsilon = 0.02 * perimeter;
        vector<Point> approx_contour;
        approxPolyDP(cnt, approx_contour, epsilon, true);
        size_t vertices_count = approx_contour.size();


        if (extent < 0.65 || abs(wh_ratio - std_wh_ratio) > 0.1) {
            drawRedRect(cnt, result);
            continue;
        } else if (solidity < 0.8) {
            drawRedRect(cnt, result);
            continue;
        } else if (circularity < 0.3) {
            drawRedRect(cnt, result);
            continue;
        } else if (vertices_count > 12) {
            drawRedRect(cnt, result);
            continue;
        } else {
        }


        if (area > min_area) {
            armors.push_back(cnt);
            // count = armors.size(); // In Python, count is updated in loop, but not used later. Kept for logic clarity if needed.
            // 绘制轮廓
            // drawContours(result, vector<vector<Point>>{cnt}, -1, Scalar(0, 255, 0), 2);
        }
    }

    // 过滤不成对光柱
    vector<pair<vector<Point>, vector<Point>>> pairs;
    vector<Point> armor_centers;
    vector<Rect> armor_bounding_rects;
    vector<double> armor_radii;

    for (const auto& armor : armors) {
        armor_centers.push_back(getCoutourCenter(armor));
        armor_bounding_rects.push_back(boundingRect(armor));
    }
    for (const auto& rect : armor_bounding_rects) {
        armor_radii.push_back(sqrt(pow(rect.width, 2) + pow(rect.height, 2)) / 2.0);
    }

    vector<bool> paired_index(armors.size(), false);

    for (size_t i = 0; i < armors.size(); ++i) {
        Point center1 = armor_centers[i];
        double radius1 = armor_radii[i];
        for (size_t j = 0; j < armors.size(); ++j) {
            if (i == j || paired_index[j]) {
                continue;
            }
            Point center2 = armor_centers[j];
            double distance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));
            if (distance < 5.5 * radius1) {
                paired_index[i] = true;
                paired_index[j] = true;
                pairs.push_back({armors[i], armors[j]});
                break;
            }
        }
    }

    for (size_t count = 0; count < pairs.size(); ++count) {
        for (size_t index = 0; index < 2; ++index) {
            vector<Point>& armor = pairs[count].first; // Accessing the first element of pair, need to fix this to iterate over both.
            if (index == 1) armor = pairs[count].second;

            Point center = getCoutourCenter(armor);
            // 参数含义：图像，圆心坐标，半径，颜色，填充
            circle(result, center, 2, Scalar(0, 0, 255), -1);

            // 绘制外接矩形
            Rect rect = boundingRect(armor);
            rectangle(result, rect, Scalar(255, 255, 255), 2);

            // 在外接矩形右上角标注灯柱序号
            // 参数含义：图像，文本，坐标，字体，字号，颜色，字体厚度
            putText(result, to_string(count + 1), Point(rect.x + rect.width, rect.y), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        }
    }

    return make_tuple(mask, result, armors);
}


int main() {
    // 读取图片: test.png
    Mat image = imread("test.png");
    if (image.empty()) {
        cerr << "Error: Cannot open image file." << endl;
        return 1;
    }

    // 将遮罩和结果分别存储于mask.jpg和result.jpg
    Mat mask, result;
    vector<vector<Point>> armors;
    tie(mask, result, armors) = drawArmorRect(image);
    imwrite("mask.jpg", mask);
    imwrite("result.jpg", result);
    cout << "Image processing complete. Results saved to mask.jpg and result.jpg" << endl;

    return 0;
}