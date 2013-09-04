/*
 * face_detector.h
 *
 *  Created on: 27 May, 2013
 *      Author: harvey
 */

#ifndef FACE_DETECTOR_H_
#define FACE_DETECTOR_H_

#include <vector>
#include <string>
#include "objdetect/objdetect.hpp"
#include "flandmark_detector.h"

using namespace std;
using namespace cv;

class FaceDetector {
private:
	CascadeClassifier m_face_cascade;
	FLANDMARK_Model  *m_landmark_model;
public:
	~FaceDetector();
	int Initialize(const string &face_model_file_name, const string &landmark_model_file_name);
	int DetectFace(Mat img, int x0, int y0, int w, int h, int size, vector<Rect> &faces);
	int DetectLandMarks(Mat img, int x0, int y0, int w, int h, Rect &head, Rect &face, vector<Point2d> &marks);
	//int DetectLandMarks2(Mat img, int x0, int y0, int w, int h, vector<Point> &precise_marks, Rect &head, Rect &face, vector<Point> &marks);
};


#endif /* FACE_DETECTOR_H_ */
