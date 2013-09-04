/*
 * face_detector.cpp
 *
 *  Created on: 27 May, 2013
 *      Author: harvey
 */

#include <imgproc/imgproc.hpp>
#include <highgui/highgui.hpp>
#include <core/core.hpp>
#include <cmath>
#include "face_detector.h"
#include "flags.h"

int FaceDetector::Initialize(const string &face_model_file_name, const string &landmark_model_file_name)
{
	m_face_cascade.load(face_model_file_name);
	m_landmark_model = flandmark_init(landmark_model_file_name.c_str());
	return 0;
}

FaceDetector::~FaceDetector()
{
	if (m_landmark_model != NULL)
	{
		delete m_landmark_model;
		m_landmark_model = NULL;
	}
}

int FaceDetector::DetectFace(Mat img, int x0, int y0, int w, int h, int size, vector<Rect> &faces)
{
    Mat gray_img;

    cvtColor(img, gray_img, CV_BGR2GRAY);
    //equalizeHist(gray_img, gray_img);

    Mat sub_img(gray_img, Rect(x0, y0, w, h));

    m_face_cascade.detectMultiScale(sub_img, faces, 1.09, 2, CV_HAAR_DO_CANNY_PRUNING, Size(size, size), Size(w, h));

    for (int i = 0; i < (int)faces.size(); i++) {
    	faces[i].x += x0;
    	faces[i].y += y0;
    }
    return faces.size();
}

int FaceDetector::DetectLandMarks(Mat img, int x0, int y0, int w, int h, Rect &head, Rect &face, vector<Point2d> &marks)
{
  double * landmarks = NULL;
  int ret = 0;
  try
  {
    vector<Rect> faces;

    Mat gray_img;
    cvtColor(img, gray_img, CV_BGR2GRAY);


    w = min(w, img.cols - x0);
    h = min(h, img.rows - y0);
    Mat sub_img(gray_img, Rect(x0, y0, w, h));

    m_face_cascade.detectMultiScale(sub_img, faces, 1.09, 2, CV_HAAR_DO_CANNY_PRUNING, Size(20, 20), Size(max(w, h), max(w, h)));


    if (faces.size() <= 0) {
      cout << "detectMultiScale error: No Face" << endl;
    	return -1;
    }

    int max_area = 0;
    int max_face = 0;
    for (int i = 0; i < (int)faces.size(); i++)
    {
       if (faces[i].height * faces[i].width > max_area)
       {
    	   max_area = faces[i].height * faces[i].width;
         max_face = i;
       }
    }



    face = faces[max_face];
    face.x += x0;
    face.y += y0;
    //rectangle(img, face, Scalar(0,0,255));

    head = faces[max_face];
    head.x += x0;
    head.y += y0;
    head.x -= head.width * FLAGS_face_margin_left;
    head.x = max(head.x, 0);
    head.y -= head.height * FLAGS_face_margin_top;
    head.y = max(head.y, 0);
    head.width *= (1 + FLAGS_face_margin_left + FLAGS_face_margin_right);
    head.width = min(head.width, img.cols - head.x - 1);
    head.height *= (1 + FLAGS_face_margin_top + FLAGS_face_margin_bottom);
    head.height = min(head.height, img.rows - head.y - 1);

    int bbox[4];
    int margin[2] = {20, 20};
    bbox[0] = faces[max_face].x;
    bbox[1] = faces[max_face].y;
    bbox[2] = faces[max_face].x + faces[max_face].width;
    bbox[3] = faces[max_face].y + faces[max_face].height;
    landmarks = (double*)malloc(2 * m_landmark_model->data.options.M * sizeof(double));

    IplImage iplimg = sub_img;
    ret = flandmark_detect(&iplimg, bbox, m_landmark_model, landmarks, margin);
    if (ret != 0)
    {
    	cout << "flandmark_detect error: " << ret << endl;
    	return -2;
    }

    double avg = 0.0;
    for (int i = 1; i < m_landmark_model->data.options.M; i++)
    {
      avg += abs(landmarks[2 * i] - landmarks[0]);
      avg += abs(landmarks[2 * i + 1] - landmarks[1]);
    }

    avg /= (m_landmark_model->data.options.M - 1);

    if (avg < (faces[max_face].width + faces[max_face].height) / 20.0)
    {
      //cout << "Fail to detect good landmarks" << endl;
      return -3;
    }


    for (int i = 0; i < m_landmark_model->data.options.M; i++)
    {
      marks.push_back(Point2d(landmarks[2 * i] + x0 - head.x, landmarks[2 * i + 1] + y0 - head.y));
    }

  }
	catch (exception &e)
	{
	  cout << "Exception in DetectLandMarks" << e.what() << endl;
	  ret = -4;
	}

	if (landmarks != NULL)
	{
	  free(landmarks);
	}

  return ret;
}


