/*
 * preprocessing.h
 *
 *  Created on: 7 Jun, 2013
 *      Author: harvey
 */

#ifndef PREPROCESSING_H_
#define PREPROCESSING_H_

#include <imgproc/imgproc.hpp>
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

int NormalizeFaceRegion(Mat img, Mat &dest, Rect &head, Rect &face, vector<Point2d> &landmarks, vector<Point2d> &normalized_marks, vector<Rect> &regions);

Mat bigger_image(Mat src);


int NormalizeFaceRegion4(Mat img, Rect head, Rect bbox, vector<Point2d> &landmarks, Mat &normalized_img, vector<Point2d> &normalized_marks, Rect &normalized_bbox);

#endif /* PREPROCESSING_H_ */
