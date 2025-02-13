#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

bool DRAW_RED_RECT = false;

cv::Point2f getCoutourCenter(const std::vector<cv::Point>& contour) {
    cv::Moments M = cv::moments(contour);
    if (M.m00 == 0) {
        return cv::Point2f(std::nanf(""), std::nanf("")); // Represent None in C++
    }
    float cx = static_cast<float>(M.m10 / M.m00);
    float cy = static_cast<float>(M.m01 / M.m00);
    return cv::Point2f(cx, cy);
}

void drawRedRect(const std::vector<cv::Point>& cnt, cv::Mat& result) {
    if (DRAW_RED_RECT) {
        std::cout << "Draw red rectangle" << std::endl;
        cv::Rect rect = cv::boundingRect(cnt);
        cv::rectangle(result, rect, cv::Scalar(0, 0, 255), 2);
    }
}

std::tuple<cv::Mat, cv::Mat, std::vector<std::vector<cv::Point>>> drawArmorRect(const cv::Mat& image, int frame_id = 0) {
    std::cout << "---Frame" << frame_id << " Start---" << std::endl;

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    cv::Scalar lower_red1(0, 110, 60);
    cv::Scalar upper_red1(15, 255, 255);
    cv::Scalar lower_red2(160, 110, 60);
    cv::Scalar upper_red2(255, 255, 255);

    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, lower_red1, upper_red1, mask1);
    cv::inRange(hsv, lower_red2, upper_red2, mask2);
    cv::bitwise_or(mask1, mask2, mask);

    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double min_area = 30.0;
    double max_area = 200.0;
    double std_wh_ratio = 7.0 / 23.0;
    cv::Mat result = image.clone();
    std::vector<std::vector<cv::Point>> armors;

    for (size_t coutourCount = 0; coutourCount < contours.size(); ++coutourCount) {
        if (coutourCount > 0) {
            std::cout << "---Coutour" << coutourCount - 1 << " End---" << std::endl;
        }
        double area = cv::contourArea(contours[coutourCount]);
        std::cout << "---Coutour" << coutourCount << " Start---" << std::endl;
        cv::Rect rect = cv::boundingRect(contours[coutourCount]);
        double rect_area = rect.width * rect.height;
        double wh_ratio = static_cast<double>(rect.width) / rect.height;
        std::cout << "Width: " << rect.width << ", Height: " << rect.height << ", Ratio: " << wh_ratio << std::endl;
        double extent = area / rect_area;
        std::cout << "Area: " << area << ", Rect Area: " << rect_area << ", Extent: " << extent << std::endl;

        std::vector<cv::Point> hull;
        cv::convexHull(contours[coutourCount], hull);
        double hull_area = cv::contourArea(hull);
        double solidity = (hull_area > 0) ? (area / hull_area) : 0;
        std::cout << "Solidity: " << solidity << std::endl;

        double perimeter = cv::arcLength(contours[coutourCount], true);
        double circularity = (perimeter > 0) ? ((4 * M_PI * area) / (perimeter * perimeter)) : 0;
        std::cout << "Circularity: " << circularity << std::endl;

        std::vector<cv::Point> approx_contour;
        double epsilon = 0.02 * perimeter;
        cv::approxPolyDP(contours[coutourCount], approx_contour, epsilon, true);
        int vertices_count = static_cast<int>(approx_contour.size());
        std::cout << "Vertices Count: " << vertices_count << std::endl;

        if (extent < 0.65 || std::abs(wh_ratio - std_wh_ratio) > 0.2 && !(area < max_area && std::abs(wh_ratio - std_wh_ratio) < 0.5)) {
            std::cout << "Extent < 0.7, skip" << std::endl;
            std::cout << "Width/Height > 0.75, skip" << std::endl;
            drawRedRect(contours[coutourCount], result);
            continue;
        } else if (solidity < 0.8) {
            std::cout << "Solidity < 0.8, skip" << std::endl;
            drawRedRect(contours[coutourCount], result);
            continue;
        } else if (circularity < 0.3) {
            std::cout << "Circularity < 0.3, skip" << std::endl;
            drawRedRect(contours[coutourCount], result);
            continue;
        } else if (vertices_count > 12) {
            std::cout << "Vertices Count > 12, skip" << std::endl;
            drawRedRect(contours[coutourCount], result);
            continue;
        } else if (area < min_area) {
            std::cout << "Area < 30, skip" << std::endl;
            drawRedRect(contours[coutourCount], result);
            continue;
        } else {
            std::cout << "Extent >= 0.7, keep" << std::endl;
            std::cout << "Solidity >= 0.8, keep" << std::endl;
            std::cout << "Circularity >= 0.3, keep" << std::endl;
            std::cout << "Vertices Count <= 12, keep" << std::endl;
            armors.push_back(contours[coutourCount]);
        }
    }

    std::vector<std::vector<cv::Point>> paired_armors;
    std::vector<std::pair<std::vector<cv::Point>, std::vector<cv::Point>>> pairs;
    std::set<int> paired_index;
    std::vector<cv::Point2f> armor_centers;
    std::vector<cv::Rect> armor_bounding_rects;
    std::vector<double> armor_radii;

    for (const auto& armor : armors) {
        armor_centers.push_back(getCoutourCenter(armor));
        armor_bounding_rects.push_back(cv::boundingRect(armor));
    }
    for (const auto& rect : armor_bounding_rects) {
        armor_radii.push_back(std::sqrt(std::pow(rect.width, 2) + std::pow(rect.height, 2)) / 2.0);
    }


    for (size_t i = 0; i < armors.size(); ++i) {
        cv::Point2f center1 = armor_centers[i];
        double radius1 = armor_radii[i];
        for (size_t j = 0; j < armors.size(); ++j) {
            if (i == j || paired_index.count(j)) {
                continue;
            }
            cv::Point2f center2 = armor_centers[j];
            double distance = std::sqrt(std::pow(center1.x - center2.x, 2) + std::pow(center1.y - center2.y, 2));
            if (distance < 5.5 * radius1) {
                std::cout << "Pair " << i << " and " << j << ". distance: " << distance << ", close to " << distance / radius1 << "x radius." << std::endl;
                paired_index.insert(i);
                paired_index.insert(j);
                pairs.push_back({armors[i], armors[j]});
                break;
            }
        }
    }
    armors = paired_armors; // armors is not used after this point in python code, so this line is not needed for functionality

    int count = 1;
    for (const auto& pair : pairs) {
        cv::Point2f center1 = getCoutourCenter(pair.first);
        cv::circle(result, center1, 2, cv::Scalar(0, 0, 255), -1);
        cv::Rect rect1 = cv::boundingRect(pair.first);
        cv::rectangle(result, rect1, cv::Scalar(255, 255, 255), 2);
        cv::putText(result, std::to_string(count) + ",1", cv::Point(rect1.x + rect1.width, rect1.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        std::cout << "---Draw " << count << ",1---" << std::endl;

        cv::Point2f center2 = getCoutourCenter(pair.second);
        cv::circle(result, center2, 2, cv::Scalar(0, 0, 255), -1);
        cv::Rect rect2 = cv::boundingRect(pair.second);
        cv::rectangle(result, rect2, cv::Scalar(255, 255, 255), 2);
        cv::putText(result, std::to_string(count) + ",2", cv::Point(rect2.x + rect2.width, rect2.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        std::cout << "---Draw " << count << ",2---" << std::endl;
        count++;
    }

    cv::putText(result, std::to_string(frame_id), cv::Point(960, 200), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 2);

    std::cout << "---Frame End---" << std::endl;

    return std::make_tuple(mask, result, armors);
}


int main() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm;
    localtime_r(&now_c, &now_tm);
    std::stringstream ss_filename_mask;
    ss_filename_mask << "mask_" << std::put_time(&now_tm, "%Y-%m-%d_%H:%M") << ".mp4";
    std::string filename_mask = ss_filename_mask.str();
    std::stringstream ss_filename_result;
    ss_filename_result << "result_" << std::put_time(&now_tm, "%Y-%m-%d_%H:%M") << ".mp4";
    std::string filename_result = ss_filename_result.str();


    cv::VideoCapture cap("test.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file." << std::endl;
        return 1;
    }

    cv::VideoWriter mask_writer, result_writer;
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = 30.0;
    int frame_width = 1920;
    int frame_height = 1080;
    cv::Size frame_size(frame_width, frame_height);

    mask_writer.open(filename_mask, fourcc, fps, frame_size, false);
    result_writer.open(filename_result, fourcc, fps, frame_size, true);


    if (!mask_writer.isOpened()) {
        std::cerr << "Error: Could not open mask_xxx.mp4 video file for writing!" << std::endl;
        return 1;
    }
    if (!result_writer.isOpened()) {
        std::cerr << "Error: Could not open result_xxx.mp4 video file for writing!" << std::endl;
        return 1;
    }

    long total_frames = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (total_frames < 0) {
        std::cerr << "无法获取视频总帧数，进度条可能无法准确显示。" << std::endl;
        total_frames = 0;
    } else {
        std::cout << "视频总帧数: " << total_frames << std::endl;
    }


    int once = 0;
    int frame_id = 0;
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            std::cout << "End of video." << std::endl;
            break;
        }

        cv::Mat mask, result;
        std::vector<std::vector<cv::Point>> armors;
        std::tie(mask, result, armors) = drawArmorRect(frame, frame_id);

        if (!once) {
            cv::imwrite("mask.jpg", mask);
            cv::imwrite("result.jpg", result);
            once = 1;
        }

        cv::Mat mask_resized, result_resized;
        cv::resize(mask, mask_resized, frame_size);
        cv::resize(result, result_resized, frame_size);


        mask_writer.write(mask_resized);
        result_writer.write(result_resized);

        frame_id++;
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    mask_writer.release();
    result_writer.release();
    cv::destroyAllWindows();
    std::cout << "Video processing complete." << std::endl;

    return 0;
}

