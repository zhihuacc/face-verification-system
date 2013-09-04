/*
 * preprocessing.cpp
 *
 *  Created on: 7 Jun, 2013
 *      Author: harvey
 */
#include "preprocessing.h"
#include <cmath>
#include <iostream>
#include "flags.h"
#include <imgproc/imgproc.hpp>

int NormalizeFaceRegion(Mat img, Mat &dest, Rect &head, Rect &face, vector<Point2d> &marks, vector<Point2d> &normalized_marks, vector<Rect> &regions)
{
  int ret = 0;
  try
  {

  vector<Point2d> abs_marks(marks.size());
  for (int i = 0; i < (int)marks.size(); i++)
  {
	  abs_marks[i].x = marks[i].x + head.x;
	  abs_marks[i].y = marks[i].y + head.y;
  }
  Point2f center(static_cast<float>(abs_marks[0].x), static_cast<float>(abs_marks[0].y));

  double w = (abs_marks[2].x + abs_marks[6].x) / 2.0 - (abs_marks[1].x + abs_marks[5].x) / 2.0;
  double h = (abs_marks[2].y + abs_marks[6].y) / 2.0 - (abs_marks[1].y + abs_marks[5].y) / 2.0;
  double angle = atan2(h, w);
  angle *= 180;
  angle /= M_PI;
  if (abs(angle) < FLAGS_max_face_horizontal_degree)
  {
	  angle = 0;
  }
  else if (angle >= FLAGS_max_face_horizontal_degree)
  {
	  angle -= FLAGS_max_face_horizontal_degree;
  }
  else if (angle <= -FLAGS_max_face_horizontal_degree)
  {
	  angle += FLAGS_max_face_horizontal_degree;
  }


  Mat rot_mat = getRotationMatrix2D(center, angle, 1);

  Mat rot_tmp(img.rows, img.cols, img.type());
  warpAffine(img, rot_tmp, rot_mat, rot_tmp.size(), INTER_LINEAR, BORDER_REPLICATE);
  //rectangle(rot_tmp, face, Scalar(0,0,255));


  vector<Point2d> tran_marks(abs_marks.size());
  for (int i = 0; i < (int)tran_marks.size(); i++)
  {
    Mat product = rot_mat * Mat(Point3d(abs_marks[i].x, abs_marks[i].y, 1.0));
    tran_marks[i].x = product.at<double>(0, 0);
    tran_marks[i].y = product.at<double>(1, 0);
  }

  Mat src_roi = Mat(rot_tmp, head);
  Mat clipped_face(head.height, head.width, rot_tmp.type());
  src_roi.copyTo(clipped_face);

  /**********************/
  double normalized_width = 0, normalized_height = 0;
  if (clipped_face.rows > clipped_face.cols)
  {
    normalized_height = FLAGS_face_normalized_max_border;
    normalized_width = normalized_height * clipped_face.cols / clipped_face.rows;
  }
  else
  {
    normalized_width = FLAGS_face_normalized_max_border;
    normalized_height = normalized_width * clipped_face.rows / clipped_face.cols;
  }


  dest = Mat(round(2 * normalized_height), round(2 * normalized_width), clipped_face.type());
  Point2f src_pts[3] = {Point2f(0,0), Point2f(clipped_face.cols - 1, 0), Point2f(0, clipped_face.rows - 1)};
  Point2f dst_pts[3] = {Point2f(static_cast<float>(normalized_width / 2), static_cast<float>(normalized_height / 2)),
                        Point2f(static_cast<float>(1.5 * normalized_width - 1), static_cast<float>(normalized_height / 2)),
                        Point2f(static_cast<float>(normalized_width / 2), static_cast<float>(1.5 * normalized_height - 1))};


  Mat stretch_mat = getAffineTransform(src_pts, dst_pts);

  warpAffine(clipped_face, dest, stretch_mat, dest.size(), INTER_LINEAR, BORDER_REPLICATE);

  normalized_marks.reserve(tran_marks.size());
  for (int i = 0; i < (int)tran_marks.size(); i++)
  {
    normalized_marks[i].x = tran_marks[i].x - head.x;
    normalized_marks[i].y = tran_marks[i].y - head.y;

    Mat product = stretch_mat * Mat(Point3d(normalized_marks[i].x, normalized_marks[i].y, 1.0));
    normalized_marks[i].x = product.at<double>(0, 0);
    normalized_marks[i].y = product.at<double>(1, 0);
  }

  }
  catch (exception &e)
  {
    cout << "Exception in NormalizeFaceRegion " << e.what() << endl;
    ret = -1;
  }
  return ret;
}

int NormalizeFaceRegion4(Mat img, Rect head, Rect bbox, vector<Point2d> &landmarks, Mat &normalized_img, vector<Point2d> &normalized_marks, Rect &normalized_bbox)
{
  int ret = 0;
  try
  {
    vector<Point2d> abs_marks(landmarks.size());
    for (int i = 0; i < (int)landmarks.size(); i++)
    {
      abs_marks[i].x = landmarks[i].x + head.x;
      abs_marks[i].y = landmarks[i].y + head.y;
    }

    Point2d bb_center(bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0);

    Point2d left_eye_mid, right_eye_mid;
    left_eye_mid = abs_marks[1] + abs_marks[5];
    left_eye_mid.x /= 2.0;
    left_eye_mid.y /= 2.0;

    right_eye_mid = abs_marks[2] + abs_marks[6];
    right_eye_mid.x /= 2.0;
    right_eye_mid.y /= 2.0;

    Point2d eyes_mid;
    eyes_mid = left_eye_mid + right_eye_mid;
    eyes_mid.x /= 2.0;
    eyes_mid.y /= 2.0;

    abs_marks.push_back(left_eye_mid);
    abs_marks.push_back(right_eye_mid);

    Point2f center(static_cast<float>(eyes_mid.x), static_cast<float>(eyes_mid.y));
    //Point2f center(static_cast<float>(abs_marks[0].x), static_cast<float>(abs_marks[0].y));

    double w = right_eye_mid.x - left_eye_mid.x;
    double h = right_eye_mid.y - left_eye_mid.y;
    double angle = atan2(h, w);
    angle *= 180;
    angle /= M_PI;
    double two_eyes_dist = sqrt(pow(left_eye_mid.x - right_eye_mid.x, 2) + pow(left_eye_mid.y - right_eye_mid.y, 2));
    double scale = 20.0 / two_eyes_dist;

    Mat rot_mat = getRotationMatrix2D(center, angle, scale);

    rot_mat.at<double>(0, 2) += 80.0 - center.x;
    rot_mat.at<double>(1, 2) += 80.0 - center.y;

    // NOTE consistent with initialize_min_max_feature_vectors()
    normalized_img = Mat(160, 160, img.type());

    warpAffine(img, normalized_img, rot_mat, normalized_img.size(), INTER_LINEAR, BORDER_REPLICATE);

    //vector<Point2d> tran_marks(abs_marks.size());
    normalized_marks.reserve(abs_marks.size());
    normalized_marks.resize(abs_marks.size());
    for (int i = 0; i < (int)abs_marks.size(); i++)
    {
      Mat product = rot_mat * Mat(Point3d(abs_marks[i].x, abs_marks[i].y, 1.0));
      normalized_marks[i].x = product.at<double>(0, 0);
      normalized_marks[i].y = product.at<double>(1, 0);

      //circle(normalized_img, normalized_marks[i], 3, Scalar(0,0,255));
    }

    Mat product = rot_mat * Mat(Point3d(bb_center.x, bb_center.y, 1.0));
    normalized_bbox.x = product.at<double>(0, 0);
    normalized_bbox.y = product.at<double>(1, 0);
    normalized_bbox.width = bbox.width * scale;
    normalized_bbox.height = bbox.height * scale;
    normalized_bbox.x -= normalized_bbox.width / 2.0;
    normalized_bbox.y -= normalized_bbox.height / 2.0;

  }
  catch (exception &e)
  {
    cout << "Exception in NormalizeFaceRegion " << e.what() << endl;
    ret = -1;
  }
  return ret;
}

Mat bigger_image(Mat src)
{
  Mat bigger_img = Mat(3 * src.rows, 3 * src.cols, src.type());
  Point2f src_pts[3] = {Point2f(0, 0), Point2f(src.cols - 1, 0), Point2f(0, src.rows - 1)};
  Point2f dst_pts[3] = {Point2f(src.cols, src.rows), Point2f(2 * src.cols - 1, src.rows), Point2f(src.cols, 2 * src.rows - 1)};


  Mat stretch_mat = getAffineTransform(src_pts, dst_pts);

  warpAffine(src, bigger_img, stretch_mat, bigger_img.size(), INTER_LINEAR, BORDER_REPLICATE);

  return  bigger_img;
}


