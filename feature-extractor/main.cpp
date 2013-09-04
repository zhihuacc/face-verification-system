/*
 * main.cpp
 *
 *  Created on: 21 May, 2013
 *      Author: harvey
 */
#include <fstream>
#include <sstream>
#include <iostream>
#include <imgproc/imgproc.hpp>
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include "flandmark_detector.h"
#include "face_detector.h"
#include "preprocessing.h"
#include "llfeatures.h"
#include "flags.h"
#include "llfeatures.h"
#include "feature_extraction_public.h"
#include "simple_sgd_svm/svm/sample.h"
#include <map>
#include "types.h"

#include <gflags/gflags.h>

using namespace std;
using namespace cv;


typedef Mat (*feature_generator)(Mat src, Mat mask, int mask_size);
typedef Mat (*normalization_technique)(Mat src);

feature_generator feature_generators[PIXEL_TYPE_NUM][AGG_TYPE_NUM];
normalization_technique nor_techs[NOR_TYPE_NUM];

map<uint64, tr1::shared_ptr<Sample> > min_feature_vectors;

map<uint64, tr1::shared_ptr<Sample> > max_feature_vectors;



int register_feature_generators()
{
  feature_generators[PIXEL_TYPE_RGB][AGG_TYPE_NONE] = rgb_none_agg;
  feature_generators[PIXEL_TYPE_RGB][AGG_TYPE_STAT] = rgb_stat;
  feature_generators[PIXEL_TYPE_RGB][AGG_TYPE_HIST] = hist_rgb2;

  feature_generators[PIXEL_TYPE_HSV][AGG_TYPE_NONE] = hsv_none_agg;
  feature_generators[PIXEL_TYPE_HSV][AGG_TYPE_STAT] = hsv_stat;
  feature_generators[PIXEL_TYPE_HSV][AGG_TYPE_HIST] = hist_hsv2;

  feature_generators[PIXEL_TYPE_INTENSITY][AGG_TYPE_NONE] = intensity_none_agg;
  feature_generators[PIXEL_TYPE_INTENSITY][AGG_TYPE_STAT] = intensity_stat;
  feature_generators[PIXEL_TYPE_INTENSITY][AGG_TYPE_HIST] = hist_intensity;

  feature_generators[PIXEL_TYPE_EDGE_MAGNITUDE][AGG_TYPE_NONE] = edge_magnitude_none_agg;
  feature_generators[PIXEL_TYPE_EDGE_MAGNITUDE][AGG_TYPE_STAT] = edge_magnitude_stat;
  feature_generators[PIXEL_TYPE_EDGE_MAGNITUDE][AGG_TYPE_HIST] = hist_edge_magnitude;

  feature_generators[PIXEL_TYPE_EDGE_ORIENTATION][AGG_TYPE_NONE] = edge_orientation_none_agg;
  feature_generators[PIXEL_TYPE_EDGE_ORIENTATION][AGG_TYPE_STAT] = edge_orientation_stat;
  feature_generators[PIXEL_TYPE_EDGE_ORIENTATION][AGG_TYPE_HIST] = hist_edge_orientation;

  feature_generators[PIXEL_TYPE_LBP][AGG_TYPE_NONE] = lbp_none_agg;
  feature_generators[PIXEL_TYPE_LBP][AGG_TYPE_STAT] = lbp_stat;
  feature_generators[PIXEL_TYPE_LBP][AGG_TYPE_HIST] = hist_lbp;

//  nor_techs[NOR_TYPE_NONE] = normalization_none;
//  nor_techs[NOR_TYPE_MEAN] = normalization_mean;
//  nor_techs[NOR_TYPE_ENERGY] = normalization_energy;

  nor_techs[NOR_TYPE_NONE] = normalization_none;
  nor_techs[NOR_TYPE_EQHIST] = equalize_hist2;


//  init_fv_set_files();

  return 0;
}


Mat transform_image(Mat src, int pixel_type)
{
	Mat dst;
	switch (pixel_type)
	{
	case PIXEL_TYPE_RGB:
		dst = rgb_type(src);
		break;
	case PIXEL_TYPE_HSV:
		dst = hsv_type(src);
		break;
	case PIXEL_TYPE_INTENSITY:
		dst = intensity_type(src);
		break;
	case PIXEL_TYPE_EDGE_MAGNITUDE:
		dst = edge_magnitude(src);
		break;
	case PIXEL_TYPE_EDGE_ORIENTATION:
		dst = edge_orientation(src);
		break;
	case PIXEL_TYPE_LBP:
	  dst = elbp(src, FLAGS_lbp_radius, FLAGS_lbp_neibors);
	  break;
	}
	return dst;
}

Mat region_mask(Mat rot_src, vector<Point2d> &landmarks, Rect normalized_bbox, int region)
{

  Mat dst = Mat(rot_src.rows, rot_src.cols, CV_8UC1);
  dst.setTo(Scalar(0));
//  rot_src.copyTo(dst);
  Point2d left_eye_mid((landmarks[1].x + landmarks[5].x) / 2, (landmarks[1].y + landmarks[5].y) / 2);
  Point2d right_eye_mid((landmarks[2].x + landmarks[6].x) / 2, (landmarks[2].y + landmarks[6].y) / 2);

  Point2d eyes_mid((left_eye_mid.x + right_eye_mid.x) / 2, (left_eye_mid.y + right_eye_mid.y) / 2);
  Point2d mouth_mid((landmarks[3].x + landmarks[4].x) / 2, (landmarks[3].y + landmarks[4].y) / 2);

  switch (region)
  {
  case REGION_EYES:
  {

    Point left_eye_mid2;
    left_eye_mid2.x = round(left_eye_mid.x);
    left_eye_mid2.y = round(left_eye_mid.y) - 1;
    Point right_eye_mid2;
    right_eye_mid2.x = round(right_eye_mid.x);
    right_eye_mid2.y = round(right_eye_mid.y) - 1;

    ellipse(dst, left_eye_mid, Size(FLAGS_eye_width / 2, FLAGS_eye_height / 2), 0, 0, 360, Scalar(255), -1);
    ellipse(dst, right_eye_mid, Size(FLAGS_eye_width / 2, FLAGS_eye_height / 2), 0, 0, 360, Scalar(255), -1);

//    ellipse(dst, left_eye_mid2, Size(FLAGS_eye_width / 2, FLAGS_eye_height / 2), 0, 0, 360, Scalar(0,0,255));
//    ellipse(dst, right_eye_mid2, Size(FLAGS_eye_width / 2, FLAGS_eye_height / 2), 0, 0, 360, Scalar(0,0,255));

    break;
  }
  case REGION_MOUTH:
  {
    Point mouth_pts[4];
    mouth_pts[0].x = round(mouth_mid.x - FLAGS_mouth_width / 2.0);
    mouth_pts[0].y = round(mouth_mid.y - FLAGS_mouth_height / 3.0);
    mouth_pts[1].x = round(mouth_pts[0].x + FLAGS_mouth_width - 1);
    mouth_pts[1].y = round(mouth_pts[0].y);
    mouth_pts[2].x = round(mouth_pts[1].x);
    mouth_pts[2].y = round(mouth_pts[1].y + FLAGS_mouth_height - 1);
    mouth_pts[3].x = round(mouth_pts[0].x);
    mouth_pts[3].y = round(mouth_pts[2].y);

    const Point *mouth_polygon[1] = {mouth_pts};
    int mouth_npts[1] = {4};
    fillPoly(dst, mouth_polygon, mouth_npts, 1, Scalar(255));
//    polylines(dst, mouth_polygon, mouth_npts, 1, true, Scalar(255, 0, 0));
    break;
  }
  case REGION_NOSE:
  {
    Point nose_pts[4];

    nose_pts[0].x = round(eyes_mid.x - FLAGS_nose_upper_width / 2.0);
    nose_pts[0].y = round(eyes_mid.y - FLAGS_nose_height * 1 / 4.0);
    nose_pts[1].x = round(nose_pts[0].x + FLAGS_nose_upper_width - 1);
    nose_pts[1].y = round(nose_pts[0].y);
    nose_pts[3].x = round(eyes_mid.x - FLAGS_nose_bottom_width / 2.0);
    nose_pts[3].y = round(nose_pts[0].y + FLAGS_nose_height - 1);
    nose_pts[2].x = round(nose_pts[3].x + FLAGS_nose_bottom_width - 1);
    nose_pts[2].y = round(nose_pts[3].y);

    const Point *nose_polygon[1] = {nose_pts};
    int nose_npts[1] = {4};
    fillPoly(dst, nose_polygon, nose_npts, 1, Scalar(255));
//    polylines(dst, nose_polygon, nose_npts, 1, true, Scalar(255, 255, 255));
    break;
  }
  case REGION_CHEEKS:
  {

    Point left_cheek_center(round(left_eye_mid.x - FLAGS_eye_width / 4.0), round((landmarks[0].y + mouth_mid.y) / 2.0));
    Point right_cheek_center(round(right_eye_mid.x + FLAGS_eye_width / 4.0), round((landmarks[0].y + mouth_mid.y) / 2.0));

    ellipse(dst, left_cheek_center, Size(FLAGS_cheek_width / 2, FLAGS_cheek_height / 2), 0, 0, 360, Scalar(255), -1);
    ellipse(dst, right_cheek_center, Size(FLAGS_cheek_width / 2, FLAGS_cheek_height / 2), 0, 0, 360, Scalar(255), -1);

//    ellipse(dst, left_cheek_center, Size(FLAGS_cheek_width / 2, FLAGS_cheek_height / 2), 0, 0, 360, Scalar(0,255,0));
//    ellipse(dst, right_cheek_center, Size(FLAGS_cheek_width / 2, FLAGS_cheek_height / 2), 0, 0, 360, Scalar(0,255,0));

    break;
  }
  case REGION_CHIN:
  {
    Point chin_center(round(mouth_mid.x), round(mouth_mid.y + FLAGS_mouth_height / 3.0 + FLAGS_chin_height / 2.0));
    ellipse(dst, chin_center, Size(FLAGS_chin_width / 2, FLAGS_chin_height / 2), 0, 0, 360, Scalar(255), -1);

//    ellipse(dst, chin_center, Size(FLAGS_chin_width / 2, FLAGS_chin_height / 2), 0, 0, 360, Scalar(0,255,0));

    break;
  }
  case REGION_EYEBROWS:
  {
    Point left_eye_brow_pts[4], right_eye_brow_pts[4];


      left_eye_brow_pts[0].x = round(left_eye_mid.x - FLAGS_eye_brow_width / 2.0);
      left_eye_brow_pts[0].y = round(left_eye_mid.y - FLAGS_eye_brow_height);
      left_eye_brow_pts[1].x = round(left_eye_brow_pts[0].x + FLAGS_eye_brow_width - 1);
      left_eye_brow_pts[1].y = round(left_eye_brow_pts[0].y);
      left_eye_brow_pts[2].x = round(left_eye_brow_pts[1].x);
      left_eye_brow_pts[2].y = round(left_eye_brow_pts[1].y + FLAGS_eye_brow_height - 1);
      left_eye_brow_pts[3].x = round(left_eye_brow_pts[0].x);
      left_eye_brow_pts[3].y = round(left_eye_brow_pts[2].y);

      right_eye_brow_pts[0].x = round(right_eye_mid.x - FLAGS_eye_brow_width / 2.0);
      right_eye_brow_pts[0].y = round(right_eye_mid.y - FLAGS_eye_brow_height);
      right_eye_brow_pts[1].x = round(right_eye_brow_pts[0].x + FLAGS_eye_brow_width - 1);
      right_eye_brow_pts[1].y = round(right_eye_brow_pts[0].y);
      right_eye_brow_pts[2].x = round(right_eye_brow_pts[1].x);
      right_eye_brow_pts[2].y = round(right_eye_brow_pts[1].y + FLAGS_eye_brow_height - 1);
      right_eye_brow_pts[3].x = round(right_eye_brow_pts[0].x);
      right_eye_brow_pts[3].y = round(right_eye_brow_pts[2].y);

    const Point *eye_brows_polygons[2] = {left_eye_brow_pts, right_eye_brow_pts};
    int eye_brows_npts[2] = {4, 4};
    fillPoly(dst, eye_brows_polygons, eye_brows_npts, 2, Scalar(255));
//    polylines(dst, eye_brows_polygons, eye_brows_npts, 2, true, Scalar(255, 0, 0));
    break;
  }
  case REGION_HAIR:
  {
    Point hair_pts[8];

//    hair_pts[0].x = round(eyes_mid.x - FLAGS_hair_upper_width / 2.0 + FLAGS_hair_height - 1);
//    hair_pts[0].y = round(normalized_bbox.y - FLAGS_hair_height + 1);
//    hair_pts[1].x = round(eyes_mid.x + FLAGS_hair_upper_width / 2.0 - FLAGS_hair_height + 1);
//    hair_pts[1].y = round(hair_pts[0].y);
//    hair_pts[2].x = round(eyes_mid.x + FLAGS_hair_upper_width / 2.0);
//    hair_pts[2].y = round(hair_pts[1].y + FLAGS_hair_height - 1);
//    hair_pts[3].x = round(hair_pts[2].x - FLAGS_hair_mid_height + 1);
//    hair_pts[3].y = round(hair_pts[2].y);
//    hair_pts[4].x = round(hair_pts[1].x);
//    hair_pts[4].y = round(hair_pts[1].y + FLAGS_hair_mid_height - 1);
//    hair_pts[5].x = round(hair_pts[0].x);
//    hair_pts[5].y = round(hair_pts[4].y);
//    hair_pts[6].x = round(eyes_mid.x - FLAGS_hair_upper_width / 2.0 + FLAGS_hair_mid_height - 1);
//    hair_pts[6].y = round(hair_pts[3].y);
//    hair_pts[7].x = round(eyes_mid.x - FLAGS_hair_upper_width / 2.0);
//    hair_pts[7].y = round(hair_pts[6].y);

    hair_pts[0].x = round(eyes_mid.x - 54 / 2.0 + 6);
    hair_pts[0].y = round(normalized_bbox.y - 14);
    hair_pts[1].x = round(eyes_mid.x + 54 / 2.0 - 6);
    hair_pts[1].y = round(hair_pts[0].y);
    hair_pts[2].x = round(eyes_mid.x + 54 / 2.0);
    hair_pts[2].y = round(hair_pts[1].y + 28);
    hair_pts[3].x = round(hair_pts[2].x - 8);
    hair_pts[3].y = round(hair_pts[2].y);
    hair_pts[4].x = round(hair_pts[1].x - 6);
    hair_pts[4].y = round(hair_pts[1].y + FLAGS_hair_mid_height - 1);
    hair_pts[5].x = round(hair_pts[0].x + 6);
    hair_pts[5].y = round(hair_pts[4].y);
    hair_pts[6].x = round(eyes_mid.x - 54 / 2.0 + 8);
    hair_pts[6].y = round(hair_pts[3].y);
    hair_pts[7].x = round(eyes_mid.x - 54 / 2.0);
    hair_pts[7].y = round(hair_pts[6].y);

    const Point *hair_polygon[1] = {hair_pts};
    int hair_npts[1] = {8};

    fillPoly(dst, hair_polygon, hair_npts, 1, Scalar(255));
//    polylines(dst, hair_polygon, hair_npts, 1, true, Scalar(0, 255, 0));
    break;
  }
  case REGION_WHOLE_FACE:
  {

    Point face_center(landmarks[0].x, landmarks[0].y);
    ellipse(dst, face_center, Size(FLAGS_whole_face_width / 2, FLAGS_whole_face_height / 2), 0, 0, 360, Scalar(255), -1);

//    ellipse(dst, face_center, Size(FLAGS_whole_face_width / 2, FLAGS_whole_face_height / 2), 0, 0, 360, Scalar(0,0,255));

    break;
  }
  }

  return dst;
}

//Mat region_mask2(Mat rot_src, vector<Point2d> &landmarks, int region)
//{
//
//  Mat dst = Mat(rot_src.rows, rot_src.cols, CV_8UC1);
//  dst.setTo(0);
//  rot_src.copyTo(dst);
////  Point2d left_eye_mid((landmarks[1].x + landmarks[5].x) / 2, (landmarks[1].y + landmarks[5].y) / 2);
////  Point2d right_eye_mid((landmarks[2].x + landmarks[6].x) / 2, (landmarks[2].y + landmarks[6].y) / 2);
////
////  Point2d eyes_mid((left_eye_mid.x + right_eye_mid.x) / 2, (left_eye_mid.y + right_eye_mid.y) / 2);
////  Point2d mouth_mid((landmarks[3].x + landmarks[4].x) / 2, (landmarks[3].y + landmarks[4].y) / 2);
//
//  switch (region)
//  {
//  case REGION_EYES:
//  {
//    Point left_eye_pts[4];
//    Point left_eye_p0, left_eye_p1, left_eye_p2, left_eye_p3, right_eye_p0, right_eye_p1, right_eye_p2, right_eye_p3;
//    left_eye_pts[0].x = FLAGS_eye_p0_x;
//    left_eye_pts[0].y = FLAGS_eye_p0_y;
//    left_eye_pts[1].x = FLAGS_eye_p1_x;
//    left_eye_pts[1].y = FLAGS_eye_p1_y;
//    left_eye_pts[2].x = FLAGS_eye_p2_x;
//    left_eye_pts[2].y = FLAGS_eye_p2_y;
//    left_eye_pts[3].x = FLAGS_eye_p3_x;
//    left_eye_pts[3].y = FLAGS_eye_p3_y;
//
//    Point right_eye_pts[4];
//    right_eye_pts[0].x = FLAGS_eye_p4_x;
//    right_eye_pts[0].y = FLAGS_eye_p4_y;
//    right_eye_pts[1].x = FLAGS_eye_p5_x;
//    right_eye_pts[1].y = FLAGS_eye_p5_y;
//    right_eye_pts[2].x = FLAGS_eye_p6_x;
//    right_eye_pts[2].y = FLAGS_eye_p6_y;
//    right_eye_pts[3].x = FLAGS_eye_p7_x;
//    right_eye_pts[3].y = FLAGS_eye_p7_y;
//
//    const Point *eye_ploygons[2] = {left_eye_pts, right_eye_pts};
//    int eye_npts[2] = {4, 4};
//    //fillPoly(dst, eye_ploygons, eye_npts, 2, Scalar(255));
//    polylines(dst, eye_ploygons, eye_npts, 2, true, Scalar(255, 255, 255));
//    break;
//  }
//  case REGION_MOUTH:
//  {
//    Point mouth_pts[4];
//    mouth_pts[0].x = FLAGS_mouth_p0_x;
//    mouth_pts[0].y = FLAGS_mouth_p0_y;
//    mouth_pts[1].x = FLAGS_mouth_p1_x;
//    mouth_pts[1].y = FLAGS_mouth_p1_y;
//    mouth_pts[2].x = FLAGS_mouth_p2_x;
//    mouth_pts[2].y = FLAGS_mouth_p2_y;
//    mouth_pts[3].x = FLAGS_mouth_p3_x;
//    mouth_pts[3].y = FLAGS_mouth_p3_y;
//
////    mouth_pts[0].x = round(mouth_mid.x - 12);
////    mouth_pts[0].y = round(mouth_mid.y - 4);
////    mouth_pts[1].x = round(mouth_pts[0].x + 23);
////    mouth_pts[1].y = round(mouth_pts[0].y);
////    mouth_pts[2].x = round(mouth_pts[1].x);
////    mouth_pts[2].y = round(mouth_pts[1].y + 11);
////    mouth_pts[3].x = round(mouth_pts[0].x);
////    mouth_pts[3].y = round(mouth_pts[2].y);
//
//    const Point *mouth_polygon[1] = {mouth_pts};
//    int mouth_npts[1] = {4};
//    //fillPoly(dst, mouth_polygon, mouth_npts, 1, Scalar(255));
//    polylines(dst, mouth_polygon, mouth_npts, 1, true, Scalar(255, 0, 0));
//    break;
//  }
//  case REGION_NOSE:
//  {
//    Point nose_pts[4];
//    nose_pts[0].x = FLAGS_nose_p0_x;
//    nose_pts[0].y = FLAGS_nose_p0_y;
//    nose_pts[1].x = FLAGS_nose_p1_x;
//    nose_pts[1].y = FLAGS_nose_p1_y;
//    nose_pts[2].x = FLAGS_nose_p2_x;
//    nose_pts[2].y = FLAGS_nose_p2_y;
//    nose_pts[3].x = FLAGS_nose_p3_x;
//    nose_pts[3].y = FLAGS_nose_p3_y;
//
//    const Point *nose_polygon[1] = {nose_pts};
//    int nose_npts[1] = {4};
//    //fillPoly(dst, nose_polygon, nose_npts, 1, Scalar(255));
//    //polylines(dst, nose_polygon, nose_npts, 1, true, Scalar(255, 255, 255));
//    break;
//  }
//  case REGION_CHEEKS:
//  {
//    Point left_cheek[4], right_cheek[4];
//    left_cheek[0].x = FLAGS_cheek_p0_x;
//    left_cheek[0].y = FLAGS_cheek_p0_y;
//    left_cheek[1].x = FLAGS_cheek_p1_x;
//    left_cheek[1].y = FLAGS_cheek_p1_y;
//    left_cheek[2].x = FLAGS_cheek_p2_x;
//    left_cheek[2].y = FLAGS_cheek_p2_y;
//    left_cheek[3].x = FLAGS_cheek_p3_x;
//    left_cheek[3].y = FLAGS_cheek_p3_y;
//
//    right_cheek[0].x = FLAGS_cheek_p4_x;
//    right_cheek[0].y = FLAGS_cheek_p4_y;
//    right_cheek[1].x = FLAGS_cheek_p5_x;
//    right_cheek[1].y = FLAGS_cheek_p5_y;
//    right_cheek[2].x = FLAGS_cheek_p6_x;
//    right_cheek[2].y = FLAGS_cheek_p6_y;
//    right_cheek[3].x = FLAGS_cheek_p7_x;
//    right_cheek[3].y = FLAGS_cheek_p7_y;
//
//    const Point *cheek_polygons[2] = {left_cheek, right_cheek};
//    int cheek_npts[2] = {4, 4};
//    //fillPoly(dst, cheek_polygons, cheek_npts, 2, Scalar(255));
//    //polylines(dst, cheek_polygons, cheek_npts, 2, true, Scalar(255, 255, 255));
//    break;
//  }
//  case REGION_CHIN:
//  {
//    Point chin_pts[4];
//    chin_pts[0].x = FLAGS_chin_p0_x;
//    chin_pts[0].y = FLAGS_chin_p0_y;
//    chin_pts[1].x = FLAGS_chin_p1_x;
//    chin_pts[1].y = FLAGS_chin_p1_y;
//    chin_pts[2].x = FLAGS_chin_p2_x;
//    chin_pts[2].y = FLAGS_chin_p2_y;
//    chin_pts[3].x = FLAGS_chin_p3_x;
//    chin_pts[3].y = FLAGS_chin_p3_y;
//
//    const Point *chin_polygon[1] = {chin_pts};
//    int chin_npts[1] = {4};
//    //fillPoly(dst, chin_polygon, chin_npts, 1, Scalar(255));
//    //polylines(dst, chin_polygon, chin_npts, 1, true, Scalar(255, 255, 255));
//    break;
//  }
//  case REGION_EYEBROWS:
//  {
//    Point left_eye_brow_pts[4], right_eye_brow_pts[4];
//    left_eye_brow_pts[0].x = FLAGS_eyebrow_p0_x;
//    left_eye_brow_pts[0].y = FLAGS_eyebrow_p0_y;
//    left_eye_brow_pts[1].x = FLAGS_eyebrow_p1_x;
//    left_eye_brow_pts[1].y = FLAGS_eyebrow_p1_y;
//    left_eye_brow_pts[2].x = FLAGS_eyebrow_p2_x;
//    left_eye_brow_pts[2].y = FLAGS_eyebrow_p2_y;
//    left_eye_brow_pts[3].x = FLAGS_eyebrow_p3_x;
//    left_eye_brow_pts[3].y = FLAGS_eyebrow_p3_y;
//
//    right_eye_brow_pts[0].x = FLAGS_eyebrow_p4_x;
//    right_eye_brow_pts[0].y = FLAGS_eyebrow_p4_y;
//    right_eye_brow_pts[1].x = FLAGS_eyebrow_p5_x;
//    right_eye_brow_pts[1].y = FLAGS_eyebrow_p5_y;
//    right_eye_brow_pts[2].x = FLAGS_eyebrow_p6_x;
//    right_eye_brow_pts[2].y = FLAGS_eyebrow_p6_y;
//    right_eye_brow_pts[3].x = FLAGS_eyebrow_p7_x;
//    right_eye_brow_pts[3].y = FLAGS_eyebrow_p7_y;
//
//    const Point *eye_brows_polygons[2] = {left_eye_brow_pts, right_eye_brow_pts};
//    int eye_brows_npts[2] = {4, 4};
//    //fillPoly(dst, eye_brows_polygons, eye_brows_npts, 2, Scalar(255));
//    //polylines(dst, eye_brows_polygons, eye_brows_npts, 2, true, Scalar(255, 0, 0));
//    break;
//  }
//  case REGION_HAIR:
//  {
//    Point hair_pts[8];
//    hair_pts[0].x = FLAGS_hair_p0_x;
//    hair_pts[0].y = FLAGS_hair_p0_y;
//    hair_pts[1].x = FLAGS_hair_p1_x;
//    hair_pts[1].y = FLAGS_hair_p1_y;
//    hair_pts[2].x = FLAGS_hair_p2_x;
//    hair_pts[2].y = FLAGS_hair_p2_y;
//    hair_pts[3].x = FLAGS_hair_p3_x;
//    hair_pts[3].y = FLAGS_hair_p3_y;
//    hair_pts[4].x = FLAGS_hair_p4_x;
//    hair_pts[4].y = FLAGS_hair_p4_y;
//    hair_pts[5].x = FLAGS_hair_p5_x;
//    hair_pts[5].y = FLAGS_hair_p5_y;
//    hair_pts[6].x = FLAGS_hair_p6_x;
//    hair_pts[6].y = FLAGS_hair_p6_y;
//    hair_pts[7].x = FLAGS_hair_p7_x;
//    hair_pts[7].y = FLAGS_hair_p7_y;
//
//    const Point *hair_polygon[1] = {hair_pts};
//    int hair_npts[1] = {8};
//
//    //fillPoly(dst, hair_polygon, hair_npts, 1, Scalar(255));
//    //polylines(dst, hair_polygon, hair_npts, 1, true, Scalar(0, 255, 0));
//    break;
//  }
//  case REGION_WHOLE_FACE:
//  {
//    Point wface_pts[4];
//    wface_pts[0].x = FLAGS_face_p0_x;
//    wface_pts[0].y = FLAGS_face_p0_y;
//    wface_pts[1].x = FLAGS_face_p1_x;
//    wface_pts[1].y = FLAGS_face_p1_y;
//    wface_pts[2].x = FLAGS_face_p2_x;
//    wface_pts[2].y = FLAGS_face_p2_y;
//    wface_pts[3].x = FLAGS_face_p3_x;
//    wface_pts[3].y = FLAGS_face_p3_y;
//
//    const Point *polygon[1] = {wface_pts};
//    const int npts[1] = {4};
//    //fillPoly(dst, polygon, npts, 1, Scalar(255));
//    polylines(dst, polygon, npts, 1, true, Scalar(0, 0, 255));
//    break;
//  }
//  }
//
//  return dst;
//}

int compose_feature_vector_line(int img_nid, int nor_type, int pixel_type, int region, int agg_type, Mat fv, string &line)
{
  stringstream fvss;
  string comment = map_feature_vector_name(img_nid, nor_type, pixel_type, region, agg_type);
  uint64 fv_id = map_feature_vector_id(img_nid, nor_type, pixel_type, region, agg_type);

  char buf[128];
  sprintf(buf, "%lu", fv_id);
  fvss << string(buf) << " ";

  int fid = 1;
  for (int i = 0; i < fv.cols; i++)
  {

    fvss << fid;
    fvss << ":";
//    if (isnan(fv.at<float>(i)))
//    {
//      cerr << "NaN Error: " << img_nid << ", " << nor_type << ", " << pixel_type<< ", " << region << ", " << agg_type << ", at " << i << endl;
//      return -1;
//    }

    double dval = static_cast<double>(fv.at<float>(i));
//    if (isnan(dval))
//    {
//      cerr << "NaN Error2: " << img_nid << ", " << nor_type << ", " << pixel_type<< ", " << region << ", " << agg_type << ", at " << i << endl;
//      return -1;
//    }
//    fvss << dval;

    char buf[128];
    sprintf(buf, "%lf", dval);
    fvss << string(buf);
    fvss << " ";
    fid++;
  }

  fvss <<" #" << comment << "\n";

  line = fvss.str();

  return 0;
}



int enumerate_all_feature_vectors2(int img_nid, Mat src, vector<Point2d> &landmarks, Rect normalized_bbox, vector<Mat> fvectors)
{
  stringstream ss;
  ss.fill('0');
  ss.width(5);
  ss << img_nid;
  ofstream ifile((FLAGS_feature_set_dir + "/" + ss.str() + ".txt").c_str());
  for (int nor_type = NOR_TYPE_START; nor_type < NOR_TYPE_NUM; nor_type++)
  {
    Mat nor_img = nor_techs[nor_type](src);
    for (int pixel_type = PIXEL_TYPE_START; pixel_type < PIXEL_TYPE_NUM; pixel_type ++)
    {
      Mat transformed_img = transform_image(nor_img, pixel_type);

      for (int region = REGION_START; region < REGION_NUM; region++)
      {
        Mat mask = region_mask(src, landmarks, normalized_bbox, region);
        int mask_size = countNonZero(mask);
        for (int agg_type = AGG_TYPE_START; agg_type < AGG_TYPE_NUM; agg_type++)
        {
          Mat fv = feature_generators[pixel_type][agg_type](transformed_img, mask, mask_size);

          string line;
          compose_feature_vector_line(img_nid, nor_type, pixel_type, region, agg_type, fv, line);

          ifile << line;
        }
      }

    }
  }


  return 0;
}

template<class T>
Mat hist_image(Mat hist, int bins)
{
  int hist_w = 1024; int hist_h = 400;
   int bin_w = cvRound( (double) hist_w/bins );

   Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

   /// Normalize the result to [ 0, histImage.rows ]
   Mat hist2;
   hist.copyTo(hist2);
   normalize(hist2, hist2, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

   /// Draw for each channel
   for( int i = 1; i <= bins; i++ )
   {
       line( histImage, Point( bin_w*(i-1), hist_h - round(hist2.at<T>(i-1)) ) ,
                        Point( bin_w*(i), hist_h - round(hist2.at<T>(i)) ),
                        Scalar( 255, 0, 0), 2, 8, 0  );
   }

   return histImage;
}

istream &operator>>(istream &f, Mat &v)
{
  int index = 1;
  for(;;) {
    int c = f.get();
    if (!f.good() || (c=='\n' || c=='\r'))
    {
      break;
    }

    if (::isspace(c))
    {
      continue;
    }
    else if (c == '#')
    {
      string comment;
      getline(f, comment);
      f.unget();
      continue;
    }


    int i;
    f.unget();
    f >> std::skipws >> i >> std::ws;
    if (i <= 0) {
      //assertfail("feature number should be positive.");
    }
    if (f.get() != ':')
    {

      f.setstate(std::ios::badbit);
      cerr << "Featur Vector in Wrong Format, ':' is required" << endl;
      return f;
      //assertfail("colon \":\" is required.");
    }
    double x;
    f >> std::skipws >> x;
    if (!f.good()) {
      //assertfail("This feature vector is in wrong format.");
//      break;
      return f;
    }
    //v.setValue(i,x);
    v.at<float>(index) = x;
    index++;
  }
  return f;
}

#include <inttypes.h>
int for_facetracer(int argc, char **argv) {

	google::ParseCommandLineFlags(&argc, &argv, true);
  setbuf(stdout, NULL);
  ifstream stat_file("/home/harvey/Dataset/facestats.txt");
  cout << "Feature Extraction Starts!" << endl;
  register_feature_generators();

//    FLANDMARK_Model *flandmark_model = flandmark_init("data/flandmark_model.dat");

//    namedWindow("Hist RGB", CV_WINDOW_NORMAL);
//    namedWindow("Hist HSV", CV_WINDOW_NORMAL);
//    namedWindow("Hist Intensity", CV_WINDOW_NORMAL);
//    namedWindow("Hist Edge Mag", CV_WINDOW_NORMAL);
//    namedWindow("Hist Edge Ori", CV_WINDOW_NORMAL);
//    namedWindow("Hist LBP", CV_WINDOW_NORMAL);
    //cvNamedWindow("Show2", CV_WINDOW_NORMAL);
    FaceDetector detector;
    detector.Initialize("data/haarcascade_frontalface_alt.xml", "data/flandmark_model.dat");

//    map<uint64, Mat>::iterator it = min_feature_vectors2.find(0);
//    if (it == min_feature_vectors2.end())
//    {
//      cerr << "ERR FOUND NOTHING" << endl;
//      return -1;
//    }

//    while (false) {
    while (!stat_file.eof()) {
      try
      {
//        namedWindow("Marks", CV_WINDOW_NORMAL);
//        namedWindow("Regions", CV_WINDOW_NORMAL);
//        namedWindow("Normalized", CV_WINDOW_NORMAL);
//        namedWindow("EdgeMag", CV_WINDOW_NORMAL);
        string face_id;
        int crop_w;
        int crop_h;
        int crop_x0;
        int crop_y0;
        int yaw;
        int pitch;
        int roll;
        int left_eye_left_x;
        int left_eye_left_y;
        int left_eye_right_x;
        int left_eye_right_y;
        int right_eye_left_x;
        int right_eye_left_y;
        int right_eye_right_x;
        int right_eye_right_y;
        int mouth_left_x;
        int mouth_left_y;
        int mouth_right_x;
        int mouth_right_y;
        stat_file >> face_id >> crop_w >> crop_h >> crop_x0 >> crop_y0 >> yaw >> pitch >> roll >> left_eye_left_x \
        >> left_eye_left_y >> left_eye_right_x >> left_eye_right_y >> right_eye_left_x >> right_eye_left_y >> right_eye_right_x \
        >> right_eye_right_y >> mouth_left_x >> mouth_left_y >> mouth_right_x >> mouth_right_y;
          if (stat_file.fail()) {
            break;
          }

//          if (atoi(face_id.c_str()) < 1984)
//          {
//            continue;
//          }

        stringstream ss;
//        ss << "/home/harvey/Dataset/test_fe/test_facetracer/";
        ss << "/home/harvey/Dataset/good-facetracer/";

        ss.fill('0');
        ss.width(5);

        ss << face_id << ".jpg";

//        stringstream ss2;
//        ss2.fill('0');
//        ss2.width(5);
//        ss2 << face_id;
//
//        string img_sid = ss2.str();
        try {
        Mat img = imread(ss.str());

        if (img.empty()) {
          //cout << "Read image fail: " << ss.str() << endl;
          continue;
        }

        if (img.depth() != 0 || img.channels() != 3)
        {
          cout << "Image Format Error: " << ss.str() << endl;
          continue;
        }

//        Mat eqimg = equalize_hist(img);
//
//        imshow("show", img);
//
//        imshow("eqhist", eqimg);
//
//        Mat gray, eqgray;
//        cvtColor(img, gray, CV_BGR2GRAY);
//        cvtColor(eqimg,eqgray, CV_BGR2GRAY);
//        imshow("gray", gray);
//        imshow("eqgray", eqgray);
//        waitKey();



//        vector<Point2d> marks;
//        Rect head, face;
//        int ret = detector.DetectLandMarks(img, crop_x0, crop_y0, crop_w, crop_h, head, face, marks);
//        if (ret != 0)
//        {
//          cout << img_sid << ".jpg, DetectlandMarks failed: " << ret << endl;
//          continue;
//        }

//          Mat bigger_img = bigger_image(img);

          vector<Point2d> marks;
          Rect head, face;
          int ret = detector.DetectLandMarks(img, crop_x0, crop_y0, crop_w, crop_h, head, face, marks);
          if (ret != 0)
          {
            cout <<"DetectlandMarks failed: " << ret << endl;
            continue;
          }



 //       namedWindow("Normalized", CV_WINDOW_NORMAL);
//        for (int i = 0; i < 8; i++)
//        {
//          circle(bigger_img, marks[i] + Point2d(head.x, head.y), 3, Scalar(0,0,255));
//        }


        Mat normalized_img;
        vector<Point2d> normalized_marks;
        Rect normalized_bbox;
        NormalizeFaceRegion4(img, head, face, marks, normalized_img, normalized_marks, normalized_bbox);
        if (ret != 0)
        {
          cout << "NormalizeFaceRegion error: " << ret << endl;
          continue;
        }

        vector<Mat> fvs;
        enumerate_all_feature_vectors2(atoi(face_id.c_str()), normalized_img, normalized_marks, normalized_bbox, fvs);

        /************** Normalized2 ********************/
//        namedWindow("Show", CV_WINDOW_NORMAL);
//        namedWindow("Normalized", CV_WINDOW_NORMAL);
//        vector<Point2d> normalized_marks;
//        Mat normalized_img;
//        NormalizeFaceRegion2(img, head, marks, normalized_img, normalized_marks);
//        imshow("Show", img);
//        imshow("Normalized", normalized_img);
//        waitKey();

        /********************************************/

        /************** Show Marks ************/
        Mat marks_;
        normalized_img.copyTo(marks_);
        for (int i = 0; i < 8; i++)
        {
              circle(marks_, normalized_marks[i], 3, Scalar(0,0,255));
        }
        rectangle(img, face, Scalar(0,255,0), 3);
//        namedWindow("Marks", CV_WINDOW_NORMAL);
//        imshow("Marks", marks_);
        namedWindow("Face", CV_WINDOW_NORMAL);
                imshow("Face", img);
        waitKey();
        /****************** Show Marks **************/

        /*************** Show Regions *************/

//        Mat regions_= Mat::zeros(normalized_img.rows, normalized_img.cols, CV_8UC1);
//        Mat regions_;
//        normalized_img.copyTo(regions_);
//  			for (int i = REGION_START; i < REGION_NUM; i++)
//  			{
//  			  regions_ = region_mask(regions_, normalized_marks, i);
//  			}
//        regions_ = region_mask(regions_, normalized_marks, REGION_HAIR);
//        regions_ = region_mask(regions_, normalized_marks, REGION_CHIN);
//  			namedWindow("Regions", CV_WINDOW_NORMAL);
//  			namedWindow("Show", CV_WINDOW_NORMAL);
//  			imshow("Regions", regions_);
//  			imshow("Show", normalized_img);
//  			waitKey();
  			/************** Show Regions ***************/

/************ Demo ******************/
//  			Mat mask = region_mask(normalized_img, normalized_marks, REGION_MOUTH);
//        Mat mask = Mat::ones(normalized.rows, normalized.cols, CV_8UC1);
//  			normalized_img.setTo(0);
//        rectangle(normalized_img, Rect(20, 20, 50, 50), Scalar(255,255,255));
//        rectangle(normalized_img, Rect(30, 30, 50, 50), Scalar(255,255,255));
//        rectangle(normalized_img, Rect(40, 40, 50, 50), Scalar(255,255,255));
//        rectangle(normalized_img, Rect(45, 45, 50, 50), Scalar(255,255,255));
//        //circle(normalized, Point(30,30), 10, Scalar(255,255,255));
//
//  			Mat rgb_mat = rgb_type(normalized_img);
//  			Mat hist_mat = hist_rgb2(normalized_img, mask, countNonZero(mask));
//  			imshow("Hist RGB", hist_image<float>(hist_mat, 256 * 3));
//
//  			Mat hsv_mat = hsv_type(normalized_img);
//  			hist_mat = hist_hsv2(normalized_img, mask, countNonZero(mask));
//        imshow("Hist HSV", hist_image<float>(hist_mat, (256 + 256 + 180)));
//
//        Mat intensity_mat = intensity_type(normalized_img);
//        hist_mat = hist_intensity(intensity_mat, mask, countNonZero(mask));
//        imshow("Hist Intensity", hist_image<float>(hist_mat, 256));
//
//
//        Mat edge_mag = edge_magnitude(normalized_img);
//        //imshow("Edge Mag", edge_mag);
//        hist_mat = hist_edge_magnitude(edge_mag, mask, countNonZero(mask));
//        imshow("Hist Edge Mag", hist_image<float>(hist_mat, FLAGS_edge_mag_hist_bins));
//
//        Mat edge_ori = edge_orientation(normalized_img);
//        //imshow("Edge Ori", edge_ori);
//        hist_mat = hist_edge_orientation(edge_ori, mask, countNonZero(mask));
//        imshow("Hist Edge Ori", hist_image<float>(hist_mat, FLAGS_edge_ori_hist_bins));
//
//        Mat lbp = elbp(normalized_img, FLAGS_lbp_radius, FLAGS_lbp_neibors);
//        Mat lbp_img;
//        lbp.convertTo(lbp_img, CV_8UC1);
//        imshow("LBP", lbp_img);
//        hist_mat = hist_lbp(lbp, mask, countNonZero(mask));
//        imshow("Hist LBP", hist_image<float>(hist_mat, FLAGS_lbp_hist_bins));
//       waitKey();
//        pause();
/********************** Demo *********************/





  //			imshow("Normalized", regions_img);
        //rectangle(img, head, Scalar(255, 0, 0), 4);
  //			circle(img, Point(marks[0].x + head.x, marks[0].y + head.y), 3, Scalar(0,0,255));
  //			for (int i = 1; i < (int)marks.size(); i++)
  //			{
  //				circle(img, Point(marks[i].x + head.x, marks[i].y + head.y), 3, Scalar(255, 0,0));
  //			}
        //imshow("Show", img);
        //imshow("Normalized", normalized);

  //			stringstream ss2;
  //			ss2 << "/home/harvey/Dataset/regions/";
  //	        ss2.fill('0');
  //	        ss2.width(5);
  //	        ss2 << face_id << ".jpg";
  //			imwrite(ss2.str(), regions_img);
        } catch (cv::Exception &e) {
          cout << "Exception in  main: " << e.what() << endl;
        }
      }
      catch (exception &e)
      {
        cout << e.what() << endl;
      }
    }

    cout << "Feature Extraction Done!" << endl;



    cout << endl << endl << "Feature Vector Scaling ..." << endl;

    cout << "   Find min&max of all feature vectors ..." << min_feature_vectors.size() << ", " << max_feature_vectors.size() << endl;
    int fv_file_id = 0;
     while (fv_file_id <= 15000)
     {
       stringstream ss;
         ss.width(5);
         ss.fill('0');
         ss << fv_file_id;

         string fname = FLAGS_feature_set_dir + ss.str() + ".txt";
         ifstream fv_file(fname.c_str());
         if (!fv_file.good())
         {
           fv_file_id++;
           continue;
         }

         while (fv_file.good())
         {
           tr1::shared_ptr<Sample> fv(new Sample);
           fv_file >> *fv;
           if (!fv_file.good())
           {
             break;
           }

           uint64 feature_type = static_cast<uint64>(round(fv->label())) & 0x0000FFFF;
           tr1::shared_ptr<Sample> min_fv, max_fv;
           map<uint64, tr1::shared_ptr<Sample> >::iterator min_it = min_feature_vectors.find(feature_type);
           map<uint64, tr1::shared_ptr<Sample> >::iterator max_it = max_feature_vectors.find(feature_type);

           //cout << "feature_type " << feature_type << endl;
           if (min_it == min_feature_vectors.end())
           {
             min_fv.reset(new Sample());
             for (int i = 1; i <= fv->x()->size(); i++)
             {
               min_fv->x()->setValue(i, fv->x()->valueAt(i));
             }
             min_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(feature_type, min_fv));
             min_it = min_feature_vectors.find(feature_type);
           }
           //cout << "here " << min_feature_vectors.size() << endl;
           min_fv = min_it->second;

           for (int i = 1; i <= fv->x()->size(); i++)
           {
              if (fv->x()->valueAt(i) < min_fv->x()->valueAt(i))
              {
                min_fv->x()->setValue(i, fv->x()->valueAt(i));
              }
           }

           if (max_it == max_feature_vectors.end())
           {
             max_fv.reset(new Sample());
             for (int i = 1; i <= fv->x()->size(); i++)
             {
               max_fv->x()->setValue(i, fv->x()->valueAt(i));
             }
             max_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(feature_type, max_fv));
             max_it = max_feature_vectors.find(feature_type);
           }
           max_fv = max_it->second;

           for (int i = 1; i <= fv->x()->size(); i++)
           {
              if (fv->x()->valueAt(i) > max_fv->x()->valueAt(i))
              {
                max_fv->x()->setValue(i, fv->x()->valueAt(i));
              }
           }
         }

         cout << "Searched " << fv_file_id << endl;
         fv_file_id++;
     }

     cout << "Min Vectors: " << min_feature_vectors.size() << endl;
     cout << "Max Vectors: " << max_feature_vectors.size() << endl;

    fv_file_id = 0;
    while (fv_file_id <= 15000)
    {
      stringstream ss;
        ss.width(5);
        ss.fill('0');
        ss << fv_file_id;

        string fname = FLAGS_feature_set_dir + ss.str() + ".txt";
        ifstream fv_file(fname.c_str());
        if (!fv_file.good())
        {
          fv_file_id++;
          continue;
        }


        string new_fname = string("/home/harvey/Dataset/feature_set2_scaled/") + ss.str() + ".txt";
        ofstream fv_new_file(new_fname.c_str());
        if (!fv_new_file.good())
        {
          fv_file_id++;
          continue;
        }

        while (fv_file.good())
        {
          tr1::shared_ptr<Sample> fv(new Sample);
          fv_file >> *fv;
          if (!fv_file.good())
          {
            continue;
          }

          uint64 feature_type = static_cast<uint64>(fv->label()) & 0x0000FFFF;
          map<uint64, tr1::shared_ptr<Sample> >::iterator min_it = min_feature_vectors.find(feature_type);
          map<uint64, tr1::shared_ptr<Sample> >::iterator max_it = max_feature_vectors.find(feature_type);
          if (min_it == min_feature_vectors.end() || max_it == max_feature_vectors.end())
          {
            cerr << "FILE " << fv_file_id << ", NO MIN or MAX Feature vector for type " << feature_type << endl;
            continue;
          }

          tr1::shared_ptr<Sample> min_fv = min_it->second;
          tr1::shared_ptr<Sample> max_fv = max_it->second;

          if (min_fv->x()->size() != max_fv->x()->size() || min_fv->x()->size() != fv->x()->size())
          {
            cerr << "FILE " << fv_file_id << ", MIN MAX FV WRONG FORMAT for feature type  " << feature_type << endl;
            cerr << "min_fv is of size " << min_fv->x()->size() << endl;
            cerr << "max_fv is of size " << max_fv->x()->size() << endl;
            cerr << "this fv is of size " << fv->x()->size() << ", label is " << fv->label() << endl;
            continue;
          }

          for (int i = 1; i <= fv->x()->size(); i++)
          {
            double scaled = 0.0;
            if ((max_fv->x()->valueAt(i) - min_fv->x()->valueAt(i)) > 0.00001)
            {
              scaled = -1 + ((fv->x()->valueAt(i) - min_fv->x()->valueAt(i)) * (2)) / (max_fv->x()->valueAt(i) - min_fv->x()->valueAt(i));
            }

            fv->x()->setValue(i, scaled);
          }

          fv_new_file << *fv;
        }
        fv_new_file.close();
        cout << "Scaled " << fv_file_id << endl;
        fv_file_id++;
    }

    cout << "Feature Vector Scaling Done!" << endl;
    return 0;
}

map<uint64, string> pubfig_names_mapping;

int main(int argc, char **argv) {

  google::ParseCommandLineFlags(&argc, &argv, true);
//  setbuf(stdout, NULL);
  ifstream stat_file("/home/harvey/Dataset/all-stat.txt");
  cout << "Feature Extraction Starts!" << endl;

  ifstream names_mapping_file("/home/harvey/Dataset/pubfig-mapping.txt");

  while (!names_mapping_file.eof())
  {
    string ids;
    string name;

    string line;
    getline(names_mapping_file, line);
    if (!names_mapping_file.good())
    {
      break;
    }

    stringstream ss(line);
    getline(ss, name, '\t');
    getline(ss, ids, '\t');

    int id = 0;
    sscanf(ids.c_str(), "%d", &id);
    pubfig_names_mapping[static_cast<uint64>(id)] = name;
  }

  register_feature_generators();

//    FLANDMARK_Model *flandmark_model = flandmark_init("data/flandmark_model.dat");

//    namedWindow("Hist RGB", CV_WINDOW_NORMAL);
//    namedWindow("Hist HSV", CV_WINDOW_NORMAL);
//    namedWindow("Hist Intensity", CV_WINDOW_NORMAL);
//    namedWindow("Hist Edge Mag", CV_WINDOW_NORMAL);
//    namedWindow("Hist Edge Ori", CV_WINDOW_NORMAL);
//    namedWindow("Hist LBP", CV_WINDOW_NORMAL);
    //cvNamedWindow("Show2", CV_WINDOW_NORMAL);
    FaceDetector detector;
    detector.Initialize("data/haarcascade_frontalface_alt.xml", "data/flandmark_model.dat");


//    while (false) {
    while (!stat_file.eof()) {
      try
      {
        int face_id;
        int crop_w;
        int crop_h;
        int crop_x0;
        int crop_y0;
        int yaw;
        int pitch;
        int roll;
        int left_eye_left_x;
        int left_eye_left_y;
        int left_eye_right_x;
        int left_eye_right_y;
        int right_eye_left_x;
        int right_eye_left_y;
        int right_eye_right_x;
        int right_eye_right_y;
        int mouth_left_x;
        int mouth_left_y;
        int mouth_right_x;
        int mouth_right_y;
        stat_file >> face_id >> crop_w >> crop_h >> crop_x0 >> crop_y0 >> yaw >> pitch >> roll >> left_eye_left_x \
        >> left_eye_left_y >> left_eye_right_x >> left_eye_right_y >> right_eye_left_x >> right_eye_left_y >> right_eye_right_x \
        >> right_eye_right_y >> mouth_left_x >> mouth_left_y >> mouth_right_x >> mouth_right_y;
          if (!stat_file.good()) {
            break;
          }

//          if (face_id < 85501)
//          {
//            continue;
//          }

        stringstream ss;
//        ss << "/home/harvey/Dataset/test_fe/test_facetracer/";
        ss << "/home/harvey/Dataset/facetracer-pubfig/";

        map<uint64, string>::iterator it = pubfig_names_mapping.find(face_id);
        if (it == pubfig_names_mapping.end())
        {
          ss.fill('0');
          ss.width(5);

          ss << face_id << ".jpg";
        }
        else
        {
          ss << it->second << ".jpg";


        }


        try {
          Mat img = imread(ss.str());

          if (img.empty()) {
//            cerr << "Read image fail: " << ss.str() << endl;
            continue;
          }

//          Mat eq_hist = equalize_hist2(img);
//          imshow("Orignal", img);
//          imshow("HE", eq_hist);
//          waitKey();
//          continue;

          if (it != pubfig_names_mapping.end())
          {
            crop_x0 -= crop_w * 0.5;
            crop_x0 = max(crop_x0, 0);
            crop_y0 -= crop_h * 0.5;
            crop_y0 = max(crop_y0, 0);

            crop_w *= 2;
            crop_w = min(img.cols - crop_x0 - 1, crop_w);
            crop_h *= 2;
            crop_h = min(img.rows - crop_y0 - 1, crop_h);
          }


          vector<Point2d> marks;
          Rect head, face;
          int ret = detector.DetectLandMarks(img, crop_x0, crop_y0, crop_w, crop_h, head, face, marks);
          if (ret != 0)
          {
            cout << "DetectlandMarks failed: " << ret << endl;
            continue;
          }



        Mat normalized_img;
        vector<Point2d> normalized_marks;
        Rect normalized_bbox;
        NormalizeFaceRegion4(img, head, face, marks, normalized_img, normalized_marks, normalized_bbox);
        if (ret != 0)
        {
          cout << "NormalizeFaceRegion error: " << ret << endl;
          continue;
        }

        vector<Mat> fvs;
        enumerate_all_feature_vectors2(face_id, normalized_img, normalized_marks, normalized_bbox, fvs);

//        imwrite("/home/harvey/Dataset/normalized.jpg", normalized_img);
//        pause();

//        Mat hist_eq = equalize_hist(normalized_img);
//        imwrite("/home/harvey/Dataset/hist-eq.jpg", hist_eq);
//
//        Mat gray;
//        cvtColor(normalized_img, gray, CV_BGR2GRAY);
//        imwrite("/home/harvey/Dataset/gray.jpg", gray);
//
//        Mat hist_gray;
//        cvtColor(hist_eq, hist_gray, CV_BGR2GRAY);
//        imwrite("/home/harvey/Dataset/hist-gray.jpg", hist_gray);
//
//
//        blur(hist_gray, hist_gray, Size(3, 3));            //
//
//        Mat canny_edge = edge_operator(hist_gray);
//
//        imwrite("/home/harvey/Dataset/edge.jpg", canny_edge);
//        waitKey();

        /************** Normalized2 ********************/
//        namedWindow("Show", CV_WINDOW_NORMAL);
//        namedWindow("Normalized", CV_WINDOW_NORMAL);
//        vector<Point2d> normalized_marks;
//        Mat normalized_img;
//        NormalizeFaceRegion2(img, head, marks, normalized_img, normalized_marks);
//        imshow("Show", img);
//        imshow("Normalized", normalized_img);
//        waitKey();

        /********************************************/

        /************** Show Marks ************/
//        Mat marks_;
//        normalized_img.copyTo(marks_);
//        for (int i = 0; i < 8; i++)
//        {
//              circle(marks_, normalized_marks[i], 3, Scalar(0,0,255));
//        }
//        rectangle(img, face, Scalar(0,255,0), 3);
//        namedWindow("Marks", CV_WINDOW_NORMAL);
//        imshow("Marks", marks_);
//        namedWindow("Face", CV_WINDOW_NORMAL);
//        imshow("Face", img);
//        waitKey();
        /****************** Show Marks **************/

        /*************** Show Regions *************/

////        Mat regions_= Mat::zeros(normalized_img.rows, normalized_img.cols, CV_8UC1);
//        Mat regions_;
//        normalized_img.copyTo(regions_);
//        for (int i = 0; i < 8; i++)
//        {
//              circle(regions_, normalized_marks[i], 1, Scalar(0,255,255));
//        }
//        for (int i = REGION_START; i < REGION_NUM; i++)
//        {
//          regions_ = region_mask(regions_, normalized_marks, normalized_bbox, i);
//        }
//
//
////        regions_ = region_mask(regions_, normalized_marks, REGION_HAIR);
////        regions_ = region_mask(regions_, normalized_marks, REGION_CHIN);
//        namedWindow("Regions", CV_WINDOW_NORMAL);
//        rectangle(regions_, normalized_bbox, Scalar(255, 0, 0), 1);
//        imshow("Regions", regions_);
////        namedWindow("Show", CV_WINDOW_NORMAL);
////        rectangle(normalized_img, normalized_bbox, Scalar(255, 0, 0), 1);
////        imshow("Show", normalized_img);
//        imwrite("/home/harvey/Dataset/ppt.jpg", regions_);
//        waitKey();
        /************** Show Regions ***************/

/************ Demo ******************/
//        Mat mask = region_mask(normalized_img, normalized_marks, REGION_MOUTH);
//        Mat mask = Mat::ones(normalized.rows, normalized.cols, CV_8UC1);
//        normalized_img.setTo(0);
//        rectangle(normalized_img, Rect(20, 20, 50, 50), Scalar(255,255,255));
//        rectangle(normalized_img, Rect(30, 30, 50, 50), Scalar(255,255,255));
//        rectangle(normalized_img, Rect(40, 40, 50, 50), Scalar(255,255,255));
//        rectangle(normalized_img, Rect(45, 45, 50, 50), Scalar(255,255,255));
//        //circle(normalized, Point(30,30), 10, Scalar(255,255,255));
//
//        Mat rgb_mat = rgb_type(normalized_img);
//        Mat hist_mat = hist_rgb2(normalized_img, mask, countNonZero(mask));
//        imshow("Hist RGB", hist_image<float>(hist_mat, 256 * 3));
//
//        Mat hsv_mat = hsv_type(normalized_img);
//        hist_mat = hist_hsv2(normalized_img, mask, countNonZero(mask));
//        imshow("Hist HSV", hist_image<float>(hist_mat, (256 + 256 + 180)));
//
//        Mat intensity_mat = intensity_type(normalized_img);
//        hist_mat = hist_intensity(intensity_mat, mask, countNonZero(mask));
//        imshow("Hist Intensity", hist_image<float>(hist_mat, 256));
//
//
//        Mat edge_mag = edge_magnitude(normalized_img);
//        //imshow("Edge Mag", edge_mag);
//        hist_mat = hist_edge_magnitude(edge_mag, mask, countNonZero(mask));
//        imshow("Hist Edge Mag", hist_image<float>(hist_mat, FLAGS_edge_mag_hist_bins));
//
//        Mat edge_ori = edge_orientation(normalized_img);
//        //imshow("Edge Ori", edge_ori);
//        hist_mat = hist_edge_orientation(edge_ori, mask, countNonZero(mask));
//        imshow("Hist Edge Ori", hist_image<float>(hist_mat, FLAGS_edge_ori_hist_bins));
//
//        Mat lbp = elbp(normalized_img, FLAGS_lbp_radius, FLAGS_lbp_neibors);
//        Mat lbp_img;
//        lbp.convertTo(lbp_img, CV_8UC1);
//        imshow("LBP", lbp_img);
//        hist_mat = hist_lbp(lbp, mask, countNonZero(mask));
//        imshow("Hist LBP", hist_image<float>(hist_mat, FLAGS_lbp_hist_bins));
//       waitKey();
//        pause();
/********************** Demo *********************/


        } catch (cv::Exception &e) {
          cout << "Exception in  main: " << e.what() << endl;
        }
      }
      catch (exception &e)
      {
        cout << e.what() << endl;
      }
    }

    cout << "Feature Extraction Done!" << endl;



    cout << endl << endl << "Feature Vector Scaling ..." << endl;

    cout << "   Find min&max of all feature vectors ..." << min_feature_vectors.size() << ", " << max_feature_vectors.size() << endl;
    int fv_file_id = 1;
     while (fv_file_id < 99999)
     {
       stringstream ss;
         ss.width(5);
         ss.fill('0');
         ss << fv_file_id;

         string fname = FLAGS_feature_set_dir + "/" + ss.str() + ".txt";
         ifstream fv_file(fname.c_str());
         if (!fv_file.good())
         {
           fv_file_id++;
           continue;
         }

         while (fv_file.good())
         {
           tr1::shared_ptr<Sample> fv(new Sample);
           fv_file >> *fv;
           if (!fv_file.good())
           {
             break;
           }

           uint64 feature_type = fv->label() & 0x0000FFFFUL;
           tr1::shared_ptr<Sample> min_fv, max_fv;
           map<uint64, tr1::shared_ptr<Sample> >::iterator min_it = min_feature_vectors.find(feature_type);
           map<uint64, tr1::shared_ptr<Sample> >::iterator max_it = max_feature_vectors.find(feature_type);

           //cout << "feature_type " << feature_type << endl;
           if (min_it == min_feature_vectors.end())
           {
             min_fv.reset(new Sample());
             for (int i = 1; i <= fv->x()->size(); i++)
             {
               min_fv->x()->setValue(i, fv->x()->valueAt(i));
             }
             min_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(feature_type, min_fv));
             min_it = min_feature_vectors.find(feature_type);
           }
           //cout << "here " << min_feature_vectors.size() << endl;
           min_fv = min_it->second;

           if (fv->x()->size() != min_fv->x()->size())
           {
             cerr << "Error SIZE HERE **************" << endl;
             cerr << fv->label() << ", " << feature_type << endl;
             exit(-1);
           }

           for (int i = 1; i <= fv->x()->size(); i++)
           {
              if (fv->x()->valueAt(i) < min_fv->x()->valueAt(i))
              {
                min_fv->x()->setValue(i, fv->x()->valueAt(i));
              }
           }

           if (max_it == max_feature_vectors.end())
           {
             max_fv.reset(new Sample());
             for (int i = 1; i <= fv->x()->size(); i++)
             {
               max_fv->x()->setValue(i, fv->x()->valueAt(i));
             }
             max_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(feature_type, max_fv));
             max_it = max_feature_vectors.find(feature_type);
           }
           max_fv = max_it->second;

           if (fv->x()->size() != max_fv->x()->size())
           {
             cerr << "Error SIZE HERE **************" << endl;
             cerr << fv->label() << ", " << feature_type << endl;
             exit(-1);
           }

           for (int i = 1; i <= fv->x()->size(); i++)
           {
              if (fv->x()->valueAt(i) > max_fv->x()->valueAt(i))
              {
                max_fv->x()->setValue(i, fv->x()->valueAt(i));
              }
           }
         }

         cout << "Searched " << fv_file_id << endl;
         fv_file_id++;
     }

     cout << "Min Vectors: " << min_feature_vectors.size() << endl;
     cout << "Max Vectors: " << max_feature_vectors.size() << endl;

    fv_file_id = 1;
    while (fv_file_id < 99999)
    {
      stringstream ss;
        ss.width(5);
        ss.fill('0');
        ss << fv_file_id;

        string fname = FLAGS_feature_set_dir + "/" + ss.str() + ".txt";
        ifstream fv_file(fname.c_str());
        if (!fv_file.good())
        {
          fv_file_id++;
          continue;
        }


        string new_fname = string("/var/all-low-level-features-scaled/") + ss.str() + ".txt";
        ofstream fv_new_file(new_fname.c_str());
        if (!fv_new_file.good())
        {
          fv_file_id++;
          continue;
        }

        while (fv_file.good())
        {
          tr1::shared_ptr<Sample> fv(new Sample);
          fv_file >> *fv;
          if (!fv_file.good())
          {
            continue;
          }

          uint64 feature_type = fv->label() & 0x0000FFFFUL;
          map<uint64, tr1::shared_ptr<Sample> >::iterator min_it = min_feature_vectors.find(feature_type);
          map<uint64, tr1::shared_ptr<Sample> >::iterator max_it = max_feature_vectors.find(feature_type);
          if (min_it == min_feature_vectors.end() || max_it == max_feature_vectors.end())
          {
            cerr << "FILE " << fv_file_id << ", NO MIN or MAX Feature vector for type " << feature_type << endl;
            continue;
          }

          tr1::shared_ptr<Sample> min_fv = min_it->second;
          tr1::shared_ptr<Sample> max_fv = max_it->second;

          if (min_fv->x()->size() != max_fv->x()->size() || min_fv->x()->size() != fv->x()->size())
          {
            cerr << "FILE " << fv_file_id << ", MIN MAX FV WRONG FORMAT for feature type  " << feature_type << endl;
            cerr << "min_fv is of size " << min_fv->x()->size() << endl;
            cerr << "max_fv is of size " << max_fv->x()->size() << endl;
            cerr << "this fv is of size " << fv->x()->size() << ", label is " << fv->label() << endl;
            continue;
          }

          for (int i = 1; i <= fv->x()->size(); i++)
          {
            double scaled = 0.0;
            if ((max_fv->x()->valueAt(i) - min_fv->x()->valueAt(i)) > 0.00001)
            {
              scaled = -1 + ((fv->x()->valueAt(i) - min_fv->x()->valueAt(i)) * (2)) / (max_fv->x()->valueAt(i) - min_fv->x()->valueAt(i));
            }

            fv->x()->setValue(i, scaled);
          }

          fv_new_file << *fv;
        }
        fv_new_file.close();
        cout << "Scaled " << fv_file_id << endl;
        fv_file_id++;
    }

    cout << "Feature Vector Scaling Done!" << endl;
}
