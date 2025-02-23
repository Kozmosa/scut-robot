#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>


void debug(const std::string& msg){
    std::cout << "[DEBUG] " << msg << std::endl;
}
void info(const std::string& msg){
    std::cout << "[INFO] " << msg << std::endl;
}
void error(const std::string& msg){
    std::cout << "[ERROR] " << msg << std::endl;
}

cv::Point get_contour_center(const std::vector<cv::Point>& contour){
    cv::Moments m = cv::moments(contour);
    return cv::Point(m.m10/m.m00, m.m01/m.m00);
}

std::vector<int> compute_center_vertices_distance(const cv::Point& center, const std::vector<cv::Point>& vertices) {
    std::vector<int> distances;
    for(auto& vertex : vertices){
        distances.push_back(cv::norm(center - vertex));
    }
    return distances;
}

int compute_interior_angle(const cv::Point& point1, const cv::Point& vertex, const cv::Point& point2) {
    cv::Point v1 = point1 - vertex;
    cv::Point v2 = point2 - vertex;
    double dot = v1.dot(v2);
    double det = v1.x*v2.y - v1.y*v2.x;
    // return angle in degrees
    return std::atan2(det, dot) * 180 / M_PI;
}

std::vector<cv::Point> order_points_hull(std::vector<cv::Point>& pts) {
    /**
     * @brief 使用凸包对四边形的顶点进行排序。
     * @param pts 输入的顶点坐标，std::vector<cv::Point2f>，包含四边形的四个顶点。
     * @return 排序后的顶点坐标 (凸包顶点)，std::vector<cv::Point2f>。
     */
    cv::Mat hull_indices; // 存储凸包顶点的索引
    std::vector<cv::Point> hull_pts; // 存储凸包顶点

    // 使用 OpenCV 计算凸包
    cv::convexHull(pts, hull_indices, true, false); // 返回凸包顶点的索引，顺时针方向

    // 根据索引提取凸包顶点
    for (int i = 0; i < hull_indices.rows; ++i) {
        int index = hull_indices.at<int>(i, 0);
        hull_pts.push_back(pts[index]);
    }
    return hull_pts;
}


const char * check_contour_type(const std::vector<cv::Point>& contour){
    // get area of contour
    double area = cv::contourArea(contour);
    // get perimeter of contour
    double perimeter = cv::arcLength(contour, true);
    // draw bounding rect for contour and get its x, y ,w ,h
    cv::Rect rect = cv::boundingRect(contour);
    // get x, y, w, h
    int x = rect.x;
    int y = rect.y;
    int w = rect.width;
    int h = rect.height;

    // compute solidity
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double hull_area = cv::contourArea(hull);
    double solidity = area / hull_area;

    // compute circularity
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contour, center, radius);
    double circularity = area / (M_PI * radius * radius);

    // compute vertices count
    std::vector<cv::Point> vertices;
    cv::approxPolyDP(contour, vertices, 0.02 * perimeter, true);
    int vertices_count = vertices.size();
    std::vector<cv::Point> vertices_ordered = order_points_hull(vertices);

    // compute extent
    double extent = area / (w * h);

    // check contour type and return corresponding string
    if(vertices_count == 3) {
        return "Triangle";
    } else if(vertices_count == 4) {
        // rectangle or square
        double aspect_ratio = (double)w / h;

        std::vector<double> angles;
        for(int i = 0; i < vertices_count; i++){
            cv::Point current_vertex = vertices_ordered[i];
            cv::Point next_vertex = vertices_ordered[(i + 1) % vertices_count];
            cv::Point previous_vertex = vertices_ordered[(i - 1 + vertices_count) % vertices_count];
            angles.push_back(compute_interior_angle(previous_vertex, current_vertex, next_vertex));
        }
        // calculate the variance of angles
        double variance = 0;
        for(auto& angle : angles){
            variance += std::pow(angle - 90, 2);
        }
        variance /= vertices_count;

        if(abs(aspect_ratio - 1) < 0.05){
            return "Square";
        } else if(variance < 1){
            return "Rectangle";
        } else {
            return "Rhombus";
        }
    } else if (vertices_count == 5) {
        // pentagon or diamond shape
        cv::Point center = get_contour_center(contour);
        std::vector<int> distances = compute_center_vertices_distance(center, vertices_ordered);
        double mean_distance = std::accumulate(distances.begin(), distances.end(), 0) / distances.size();
        double variance = 0;
        for(auto& distance : distances){
            variance += std::pow(distance - mean_distance, 2);
        }
        variance /= distances.size();
        double dis_range = *std::max_element(distances.begin(), distances.end()) - *std::min_element(distances.begin(), distances.end());
        if(variance < 1 && dis_range < 0.1 * mean_distance){
            return "Pentagon";
        }
        else if(solidity > 0.99){
            return "Semicircle";
        }
        else {
            return "Diamond";
        }
    } else if (vertices_count == 6) {
        cv::Point center = get_contour_center(contour);
        std::vector<int> distances = compute_center_vertices_distance(center, vertices_ordered);
        double variance = 0;
        if(variance < 1) {
            return "Hexagon";
        } else {
            return "Unknown";
        }
    } else if (vertices_count == 7) {
        return "Arrow";
    } else if (vertices_count == 9) {
        if(solidity > 0.9) {
            return "Heart";
        } else {
            return "Lunar";
        }
    }
    else if(vertices_count == 10) {
        return "Star";
    }
    else if(vertices_count == 12) {
        return "Cross";
    }
    else if(circularity > 0.95) {
        return "Circle";
    }
    else if(circularity > 0.7 && solidity > 0.98) {
        return "Eclipse";
    }
    else {
        return "Unknown";
    }
}

int main() {
    // load test image
    info("Loading test image");
    cv::Mat img = cv::imread("test.jpg");
    if(img.empty()){
        error("Failed to load image");
        return 1;
    }

    // convert to grayscale
    info("Converting to grayscale");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    // convert to binary
    info("Converting to binary");
    cv::Mat binary;
    cv::threshold(img, binary, 128, 255, cv::THRESH_BINARY);

    // convert to binary inverse
    info("Converting to binary inverse");
    cv::Mat binary_inv;
    cv::threshold(img, binary_inv, 128, 255, cv::THRESH_BINARY_INV);

    // execute opening and closing operations towards binary_inv
    info("Executing opening and closing operations");
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::morphologyEx(binary_inv, binary_inv, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary_inv, binary_inv, cv::MORPH_CLOSE, kernel);

    // detect contours in binary_inv
    info("Detecting contours");
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_inv, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for(auto& contour : contours) {
        debug("Contour area: " + std::to_string(cv::contourArea(contour)));
        cv::Rect rect = cv::boundingRect(contour);
        cv::rectangle(img, rect, cv::Scalar(0,255,0), 2);
        const char * type = check_contour_type(contour);
        cv::putText(img, type, cv::Point(rect.x, rect.y-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
    }

    cv::imwrite("output.jpg", img);

    return 0;
}