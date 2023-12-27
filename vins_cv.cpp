#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
vector<uchar> status;
vector<float> err;
int MAX_CNT=50;
int MIN_DIST=20;
int ROW;
int COL;

//去除无法跟踪的特征点
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    //cvRound()：返回跟参数最接近的整数值，即四舍五入；
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1_raw = imread(argv[1], CV_LOAD_IMAGE_COLOR );
  Mat img_2_raw = imread(argv[2], CV_LOAD_IMAGE_COLOR );
  Mat img_opt = img_2_raw.clone();

  assert(img_1_raw.data != nullptr && img_2_raw.data != nullptr);

  Mat img_1,img_2;
  cvtColor(img_1_raw, img_1, COLOR_BGR2GRAY);
  cvtColor(img_2_raw, img_2, COLOR_BGR2GRAY);
  
  ROW = img_1.rows;
  COL = img_1.cols;
  /*
  

  0.01 是可接受的最低质量参数，如果某一特征点的质量小于 最佳质量*0.01，则该特征点会被拒绝采用
  */

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  Mat mask = Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
  goodFeaturesToTrack(img_1, prev_pts, MAX_CNT , 0.01, MIN_DIST, mask);
  goodFeaturesToTrack(img_2, cur_pts , MAX_CNT , 0.01, MIN_DIST, mask);

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  calcOpticalFlowPyrLK(img_1, img_2, prev_pts, forw_pts, status, err, cv::Size(21, 21), 3);

  cout << "extract FAST Corners = " << time_used.count() << " seconds. " << endl;

  for (int i = 0; i < int(forw_pts.size()); i++)
    if (status[i] && !inBorder(forw_pts[i]))
        status[i] = 0;

  reduceVector(prev_pts,status);
  reduceVector(forw_pts,status);

  // cv::Mat img_feature_vis1,img_feature_vis2;
  for (const cv::Point2f& pt : prev_pts) {
    cv::circle(img_1_raw, pt, 5, cv::Scalar(0, 0, 255), 2); // 在图像上绘制圆圈
  }
  // for (const cv::Point2f& pt : cur_pts) {
  //   cv::circle(img_2_raw, pt, 5, cv::Scalar(0, 0, 255), 2); // 在图像上绘制圆圈
  // }
  for (const cv::Point2f& pt : forw_pts) {
    cv::circle(img_2_raw, pt, 5, cv::Scalar(255, 0, 0), 2); // 在图像上绘制圆圈
  }

  for (size_t i = 0; i < prev_pts.size(); i++) {
    cv::Point2f prev_pt = prev_pts[i]; // 当前帧的角点
    cv::Point2f forw_pt = forw_pts[i]; // 下一帧的角点

    // 绘制箭头
    cv::arrowedLine(img_opt, prev_pt, forw_pt, cv::Scalar(0, 0, 255), 2);
  }

  // 可视化光流追踪效果
  cv::Mat img_visualization;
  cv::hconcat(img_1_raw, img_2_raw, img_visualization); // 水平拼接两个图像
  for (int i = 0; i < prev_pts.size(); i++) 
  {
    cv::Point2f p1 = prev_pts[i];
    cv::Point2f p2 = forw_pts[i];
    p2.x += COL;
    cv::line(img_visualization, p1, p2, cv::Scalar(0, 255, 0), 1); // 绘制箭头
  }

  cout<<"corner 1 size : " << prev_pts.size() <<endl;
  cout<<"corner 2 size : " << cur_pts.size() <<endl;
  cout<<"corner 3 size : " << forw_pts.size() <<endl;
  
  imshow("Corner Visualization 1", img_1_raw);
  imshow("Corner Visualization 2", img_2_raw);
  imshow("LK Visualization 1", img_opt);
  imshow("LK Visualization 2", img_visualization);

  waitKey(0);

  return 0;
}
