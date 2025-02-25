#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <deque>
#include <vector>
#include <algorithm>
#include <iostream>

class OrangeRedDetector {
private:
    cv::KalmanFilter kf;
    cv::Mat_<float> measurement;
    std::deque<cv::Point> centerPointsBuffer;
    std::deque<int> widthBuffer;
    std::deque<int> heightBuffer;
    cv::Point lastCenter;
    int lastWidth = 0;
    int lastHeight = 0;
    int lostFramesCount = 0;
    const int bufferSize = 20;
    const int minArea = 2000;

public:
    OrangeRedDetector() : kf(4, 2, 0) {
        // Initialize Kalman filter
        // State vector: [x, y, vx, vy]
        // Measurement vector: [x, y]
        
        // Transition matrix (assumes constant velocity model)
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
            
        // Measurement matrix
        kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 
            1, 0, 0, 0,
            0, 1, 0, 0);
            
        // Process noise covariance matrix
        setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-5));
        
        // Measurement noise covariance matrix
        setIdentity(kf.measurementNoiseCov, cv::Scalar::all(0.5));
        
        // Initial state covariance
        setIdentity(kf.errorCovPost, cv::Scalar::all(1000));
        
        // Initial state
        kf.statePost = (cv::Mat_<float>(4, 1) << 0, 0, 0, 0);
        
        measurement = cv::Mat_<float>(2, 1);
        measurement.setTo(cv::Scalar(0));
        
        // Set maximum buffer size
        centerPointsBuffer.resize(0);
        widthBuffer.resize(0);
        heightBuffer.resize(0);
    }
    
    void detect() {
        // Open default camera
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open camera" << std::endl;
            return;
        }
        
        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "Cannot read frame" << std::endl;
                break;
            }
            
            // Convert to HSV color space
            cv::Mat hsvFrame;
            cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
            
            // Apply Gaussian blur
            cv::Mat blurredHsvFrame;
            cv::GaussianBlur(hsvFrame, blurredHsvFrame, cv::Size(5, 5), 0);
            
            // Define orange-red color range
            cv::Scalar lowerOrangeRed(0, 100, 100);
            cv::Scalar upperOrangeRed(25, 255, 255);
            
            // Create mask for orange-red color
            cv::Mat orangeRedMask;
            cv::inRange(blurredHsvFrame, lowerOrangeRed, upperOrangeRed, orangeRedMask);
            
            // Morphological operations to remove noise
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::morphologyEx(orangeRedMask, orangeRedMask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(orangeRedMask, orangeRedMask, cv::MORPH_CLOSE, kernel);
            
            // Find contours
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(orangeRedMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            // Draw all contours for visualization
            cv::Mat contoursFrame = frame.clone();
            cv::drawContours(contoursFrame, contours, -1, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Contours", contoursFrame);
            
            // Find potential rectangles
            std::vector<std::pair<cv::Rect, std::vector<cv::Point>>> detectedRectangles;
            
            for (const auto& contour : contours) {
                // Approximate contour to check if it's a rectangle
                double perimeter = cv::arcLength(contour, true);
                std::vector<cv::Point> approx;
                cv::approxPolyDP(contour, approx, 0.04 * perimeter, true);
                
                // Calculate solidity (area / convex hull area)
                double area = cv::contourArea(contour);
                std::vector<cv::Point> hull;
                cv::convexHull(contour, hull);
                double hullArea = cv::contourArea(hull);
                
                if (hullArea <= 0) continue; // Avoid division by zero
                
                double solidity = area / hullArea;
                if (solidity < 0.1) continue; // Skip if solidity is too low
                
                // Check if it's a rectangle (4 vertices after approximation)
                if (approx.size() == 4) {
                    cv::Rect boundRect = cv::boundingRect(approx);
                    double aspectRatio = static_cast<double>(boundRect.width) / boundRect.height;
                    
                    if (area > minArea) {
                        detectedRectangles.push_back(std::make_pair(boundRect, contour));
                    }
                }
            }
            
            // Find the largest rectangle
            std::pair<cv::Rect, std::vector<cv::Point>> bestRectangle;
            double maxArea = 0;
            bool foundRectangle = false;
            
            for (const auto& rectInfo : detectedRectangles) {
                const auto& contour = rectInfo.second;
                double area = cv::contourArea(contour);
                
                if (area > maxArea) {
                    maxArea = area;
                    bestRectangle = rectInfo;
                    foundRectangle = true;
                }
            }
            
            if (foundRectangle) {
                if (lostFramesCount != 0) {
                    // Rectangle rediscovered, clear buffers
                    widthBuffer.clear();
                    heightBuffer.clear();
                    centerPointsBuffer.clear();
                    lostFramesCount = 0;
                }
                
                const cv::Rect& boundRect = bestRectangle.first;
                int centerXMeasured = boundRect.x + boundRect.width / 2;
                int centerYMeasured = boundRect.y + boundRect.height / 2;
                
                // Update measurement
                measurement(0) = centerXMeasured;
                measurement(1) = centerYMeasured;
                
                // Kalman prediction and update
                cv::Mat prediction = kf.predict();
                cv::Mat estimated = kf.correct(measurement);
                
                // Get filtered center position
                int centerXFiltered = estimated.at<float>(0);
                int centerYFiltered = estimated.at<float>(1);
                
                // Apply median filtering to center points
                centerPointsBuffer.push_back(cv::Point(centerXFiltered, centerYFiltered));
                if (centerPointsBuffer.size() > bufferSize) {
                    centerPointsBuffer.pop_front();
                }
                
                if (!centerPointsBuffer.empty()) {
                    std::vector<int> xCoords, yCoords;
                    for (const auto& point : centerPointsBuffer) {
                        xCoords.push_back(point.x);
                        yCoords.push_back(point.y);
                    }
                    
                    std::sort(xCoords.begin(), xCoords.end());
                    std::sort(yCoords.begin(), yCoords.end());
                    
                    size_t middleIdx = xCoords.size() / 2;
                    centerXFiltered = xCoords[middleIdx];
                    centerYFiltered = yCoords[middleIdx];
                }
                
                // Apply median filtering to width and height
                widthBuffer.push_back(boundRect.width);
                heightBuffer.push_back(boundRect.height);
                if (widthBuffer.size() > bufferSize) {
                    widthBuffer.pop_front();
                }
                if (heightBuffer.size() > bufferSize) {
                    heightBuffer.pop_front();
                }
                
                int wFiltered = boundRect.width;
                int hFiltered = boundRect.height;
                
                if (!widthBuffer.empty() && !heightBuffer.empty()) {
                    std::vector<int> widths(widthBuffer.begin(), widthBuffer.end());
                    std::vector<int> heights(heightBuffer.begin(), heightBuffer.end());
                    
                    std::sort(widths.begin(), widths.end());
                    std::sort(heights.begin(), heights.end());
                    
                    size_t middleIdx = widths.size() / 2;
                    wFiltered = widths[middleIdx];
                    hFiltered = heights[middleIdx];
                }
                
                // Recalculate rectangle using filtered center and dimensions
                int xFiltered = centerXFiltered - wFiltered / 2;
                int yFiltered = centerYFiltered - hFiltered / 2;
                
                lastWidth = wFiltered;
                lastHeight = hFiltered;
                lastCenter = cv::Point(centerXFiltered, centerYFiltered);
                
                // Draw rectangle
                cv::rectangle(frame, cv::Point(xFiltered, yFiltered), cv::Point(xFiltered + wFiltered, yFiltered + hFiltered), cv::Scalar(0, 255, 0), 2);
            }
            else {
                // No rectangle detected
                kf.predict(); // Still predict to maintain state
                lostFramesCount++;
                
                if (lostFramesCount < 10) {
                    // Use predicted position for a few frames
                    cv::Mat prediction = kf.statePre;
                    int centerXFiltered = prediction.at<float>(0);
                    int centerYFiltered = prediction.at<float>(1);
                    int xFiltered = centerXFiltered - lastWidth / 2;
                    int yFiltered = centerYFiltered - lastHeight / 2;
                    
                    cv::rectangle(frame, cv::Point(xFiltered, yFiltered), cv::Point(xFiltered + lastWidth, yFiltered + lastHeight), cv::Scalar(0, 255, 0), 2);
                }
                else {
                    // Clear buffers after too many lost frames
                    widthBuffer.clear();
                    heightBuffer.clear();
                    centerPointsBuffer.clear();
                }
            }
            
            // Show result
            cv::imshow("Orange-Red Rectangle Detection with Kalman Filter", frame);
            
            // Exit on 'q' key press
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
    }
};

int main() {
    OrangeRedDetector detector;
    detector.detect();
    return 0;
}