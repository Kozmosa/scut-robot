#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace cv;

int main()
{
    // 定义ArUco字典和检测参数
    Ptr<aruco::Dictionary> arucoDict = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    // 创建ArucoDetector对象（需要OpenCV 4.7+）
    aruco::ArucoDetector detector(arucoDict, detectorParams);

    // 假设的相机参数（实际使用时需要替换为校准数据）
    Mat cameraMatrix = (Mat_<float>(3, 3) << 1000, 0, 320,
                                              0, 1000, 240,
                                              0, 0, 1);
    Mat distCoeffs = Mat::zeros(5, 1, CV_32F); // 假设无镜头畸变

    // 打开摄像头
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }

    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        // 检测ArUco标记
        vector<vector<Point2f>> corners;
        vector<int> ids;
        vector<vector<Point2f>> rejected;
        detector.detectMarkers(frame, corners, ids, rejected);

        if (!ids.empty())
        {
            // 绘制检测到的标记
            aruco::drawDetectedMarkers(frame, corners, ids);

            // 估计姿态（标记实际边长以米为单位）
            double markerLength = 0.05; // 根据实际打印的标记尺寸修改
            vector<Vec3d> rvecs, tvecs;
            aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

            // 遍历每个检测到的标记
            for (size_t i = 0; i < ids.size(); i++)
            {
                // 绘制坐标轴（长度为3cm）
                drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03);

                // 提取相对位置（相机坐标系）
                double x = tvecs[i][0];
                double y = tvecs[i][1];
                double z = tvecs[i][2];

                // 显示位置信息
                stringstream ss;
                ss << "ID " << ids[i] << ": X:" << fixed << setprecision(2) << x << "m, "
                   << "Y:" << y << "m, "
                   << "Z:" << z << "m";
                putText(frame, ss.str(), Point(10, 30 + 30 * i),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
            }
        }

        imshow("ArUco Detection", frame);
        // 按键 'q' 退出
        if (waitKey(1) == 'q')
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
