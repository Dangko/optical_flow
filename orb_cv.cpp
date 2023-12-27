#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  //-- 初始化
  //特征点，使用 opencv 提供的类
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  //描述子，直接用 Mat 表示，一行代表一个描述子？
  Mat descriptors_1, descriptors_2;
  /*
    template<typename _Tp >
    opencv2/core 中定义的  using cv::Ptr = typedef std::shared_ptr<_Tp>
    通过模板 template，创建任意模板的智能指针类
    模板是一种泛用型的编程工具，可以用于处理不同类型的数据，主要应用于函数模板和类模板中
  */
  //特征点提取，描述子提取、和描述子匹配都使用 opencv 提供的类实现
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);
  
  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;
  

  Mat outimg1,outimg2;
  drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  imshow("ORB features1", outimg1);
  drawKeypoints(img_2, keypoints_2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  imshow("ORB features2", outimg2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  // 汉明距离是计算两个二进制字符串之间的差异度量。
  //"BruteForce-Hamming" 代表暴力匹配算法，
  //计算查询图像中的每个特征点描述符与训练图像中的所有特征点描述符之间的Hamming距离，
 // 并寻找最小距离。最小距离越小，匹配越好
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  cout<<"keypoint 1 size : " << keypoints_1.size() <<endl;
  cout<<"keypoint 2 size : " << keypoints_2.size() <<endl;
  cout<<"describer1 rows : " << descriptors_1.rows <<endl;
  cout<<"describer1 cols : " << descriptors_1.cols <<endl;
  cout<<"describer2 rows : " << descriptors_2.rows <<endl;
  cout<<"describer2 cols : " << descriptors_2.cols <<endl;
  cout<<"match size : "<< matches.size()<<endl;
  cout<<"good match size : "<< good_matches.size()<<endl;

  //-- 第五步:绘制匹配结果
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
  imshow("all matches", img_match);
  imshow("good matches", img_goodmatch);
  waitKey(0);

  return 0;
}
