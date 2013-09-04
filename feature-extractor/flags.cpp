/*
 * flags.cpp
 *
 *  Created on: 10 Jun, 2013
 *      Author: harvey
 */

#include "flags.h"

DEFINE_double(face_margin_top, 1.0 / 3.0, "");
DEFINE_double(face_margin_left, 1.0 / 4.0, "");
DEFINE_double(face_margin_right, 1.0 / 4.0, "");
DEFINE_double(face_margin_bottom, 1.0 / 4.0, "");
//DEFINE_int32(face_normalized_width, 256, "");
//DEFINE_int32(face_normalized_height, 256, "");
//
//DEFINE_double(eye_ratio, 1.5, "");
//DEFINE_double(eye_h_w_ratio, 0.8, "");
//DEFINE_double(mouth_ratio, 1.4, "");
//DEFINE_double(mouth_h_w_ratio, 0.6, "");
//DEFINE_double(nose_ratio, 1.8, "");
//DEFINE_double(nose_h_w_ratio, 1.3, "");
//DEFINE_double(eye_brow_ratio, 1.4, "");
//DEFINE_double(eye_brow_h_w_ratio, 0.8, "");
//
//DEFINE_double(cheek_ratio, 1, "");
//DEFINE_double(cheek_h_w_ratio, 1.2, "");
//
//DEFINE_double(chin_ratio, 2.2, "");
//DEFINE_double(chin_h_w_ratio, 0.4, "");
//
//DEFINE_double(hair_ratio, 1.8, "");
//DEFINE_double(hair_h_w_ratio, 0.5, "");


DEFINE_int32(max_face_horizontal_degree, 3, "");

//DEFINE_double(whole_face_ratio, 1.4, "");
//DEFINE_double(whole_face_h_w_ratio, 1.5, "");

DEFINE_double(edge_detect_threshold1, 32, "");
DEFINE_double(edge_detect_threshold2, 96, "");
DEFINE_double(edge_detect_aperture_size, 3, "");
DEFINE_bool(edge_detect_l2gradient, false, "");

/************************256***********************/

//DEFINE_int32(eye_width, 60, "");
//DEFINE_int32(eye_height, 30, "");
//
//DEFINE_int32(nose_upper_width, 30, "");
//DEFINE_int32(nose_bottom_width, 70, "");
//DEFINE_int32(nose_height, 70, "");
//
//DEFINE_int32(mouth_width, 80, "");
//DEFINE_int32(mouth_height, 50, "");
//
//DEFINE_int32(chin_width, 90, "");
//DEFINE_int32(chin_height, 50, "");
//DEFINE_double(chin_dist, 0.8, "");
//
//DEFINE_int32(cheek_width, 60, "");
//DEFINE_int32(cheek_height, 70, "");
//DEFINE_double(cheek_dist, 1.1, "");
//
//DEFINE_int32(eye_brow_width, 70, "");
//DEFINE_int32(eye_brow_height, 40, "");
//DEFINE_double(eye_brow_dist, 0.9, "");
//
//DEFINE_int32(hair_upper_width, 180, "");
//DEFINE_int32(hair_mid_width, 120, "");
//DEFINE_int32(hair_height, 120, "");
//DEFINE_int32(hair_mid_height, 60, "");
//DEFINE_double(hair_dist, 1.8, "");
//
//DEFINE_int32(whole_face_width, 150, "");
//DEFINE_int32(whole_face_height, 180, "");

/********************128***********************/

DEFINE_int32(eye_width, 16, "");
DEFINE_int32(eye_height, 8, "");

DEFINE_int32(nose_upper_width, 12, "");
DEFINE_int32(nose_bottom_width, 20, "");
DEFINE_int32(nose_height, 24, "");

DEFINE_int32(mouth_width, 24, "");
DEFINE_int32(mouth_height, 14, "");

DEFINE_int32(chin_width, 22, "");
DEFINE_int32(chin_height, 12, "");
DEFINE_double(chin_dist, 0.7, "");

DEFINE_int32(cheek_width, 12, "");
DEFINE_int32(cheek_height, 24, "");
DEFINE_double(cheek_dist, 0.9, "");

DEFINE_int32(eye_brow_width, 22, "");
DEFINE_int32(eye_brow_height, 14, "");
DEFINE_double(eye_brow_dist, 0.8, "");

DEFINE_int32(hair_upper_width, 50, "");
DEFINE_int32(hair_mid_width, 30, "");
DEFINE_int32(hair_height, 14, "");
DEFINE_int32(hair_mid_height, 8, "");
DEFINE_double(hair_dist, 1.8, "");

DEFINE_int32(whole_face_width, 38, "");
DEFINE_int32(whole_face_height, 50, "");

DEFINE_int32(face_normalized_max_border, 96, "");
//DEFINE_double(region_ratio_scale, 1, "");

/************************/

DEFINE_int32(eyes_mid_x, 50, "");
DEFINE_int32(eyes_mid_y, 50, "");
DEFINE_int32(two_eyes_dist, 30, "");
DEFINE_int32(normalized_face_width, 100, "");

DEFINE_int32(eye_p0_x, FLAGS_eyes_mid_x - 22, "");
DEFINE_int32(eye_p0_y, FLAGS_eyes_mid_y - 8, "");
DEFINE_int32(eye_p1_x, FLAGS_eyes_mid_x - 4, "");
DEFINE_int32(eye_p1_y, FLAGS_eyes_mid_y - 8, "");
DEFINE_int32(eye_p2_x, FLAGS_eyes_mid_x - 4, "");
DEFINE_int32(eye_p2_y, FLAGS_eyes_mid_y + 4, "");
DEFINE_int32(eye_p3_x, FLAGS_eyes_mid_x - 22, "");
DEFINE_int32(eye_p3_y, FLAGS_eyes_mid_y + 4, "");

DEFINE_int32(eye_p4_x, FLAGS_eyes_mid_x + 4, "");
DEFINE_int32(eye_p4_y, FLAGS_eyes_mid_y - 8, "");
DEFINE_int32(eye_p5_x, FLAGS_eyes_mid_x + 22, "");
DEFINE_int32(eye_p5_y, FLAGS_eyes_mid_y - 8, "");
DEFINE_int32(eye_p6_x, FLAGS_eyes_mid_x + 22, "");
DEFINE_int32(eye_p6_y, FLAGS_eyes_mid_y + 4, "");
DEFINE_int32(eye_p7_x, FLAGS_eyes_mid_x + 4, "");
DEFINE_int32(eye_p7_y, FLAGS_eyes_mid_y + 4, "");

DEFINE_int32(eyebrow_p0_x, FLAGS_eyes_mid_x - 22, "");
DEFINE_int32(eyebrow_p0_y, FLAGS_eyes_mid_y - 13, "");
DEFINE_int32(eyebrow_p1_x, FLAGS_eyes_mid_x - 4, "");
DEFINE_int32(eyebrow_p1_y, FLAGS_eyes_mid_y - 13, "");
DEFINE_int32(eyebrow_p2_x, FLAGS_eyes_mid_x - 4, "");
DEFINE_int32(eyebrow_p2_y, FLAGS_eyes_mid_y - 3, "");
DEFINE_int32(eyebrow_p3_x, FLAGS_eyes_mid_x - 22, "");
DEFINE_int32(eyebrow_p3_y, FLAGS_eyes_mid_y - 3, "");

DEFINE_int32(eyebrow_p4_x, FLAGS_eyes_mid_x + 4, "");
DEFINE_int32(eyebrow_p4_y, FLAGS_eyes_mid_y - 13, "");
DEFINE_int32(eyebrow_p5_x, FLAGS_eyes_mid_x + 22, "");
DEFINE_int32(eyebrow_p5_y, FLAGS_eyes_mid_y - 13, "");
DEFINE_int32(eyebrow_p6_x, FLAGS_eyes_mid_x + 22, "");
DEFINE_int32(eyebrow_p6_y, FLAGS_eyes_mid_y - 3, "");
DEFINE_int32(eyebrow_p7_x, FLAGS_eyes_mid_x + 4, "");
DEFINE_int32(eyebrow_p7_y, FLAGS_eyes_mid_y - 3, "");

DEFINE_int32(nose_p0_x, FLAGS_eyes_mid_x - 5, "");
DEFINE_int32(nose_p0_y, FLAGS_eyes_mid_y - 5, "");
DEFINE_int32(nose_p1_x, FLAGS_eyes_mid_x + 5, "");
DEFINE_int32(nose_p1_y, FLAGS_eyes_mid_y - 5, "");
DEFINE_int32(nose_p2_x, FLAGS_eyes_mid_x + 12, "");
DEFINE_int32(nose_p2_y, FLAGS_eyes_mid_y + 17, "");
DEFINE_int32(nose_p3_x, FLAGS_eyes_mid_x - 12, "");
DEFINE_int32(nose_p3_y, FLAGS_eyes_mid_y + 17, "");

DEFINE_int32(mouth_p0_x, FLAGS_eyes_mid_x - 12, "");
DEFINE_int32(mouth_p0_y, FLAGS_eyes_mid_y + 20, "");
DEFINE_int32(mouth_p1_x, FLAGS_eyes_mid_x + 12, "");
DEFINE_int32(mouth_p1_y, FLAGS_eyes_mid_y + 20, "");
DEFINE_int32(mouth_p2_x, FLAGS_eyes_mid_x + 12, "");
DEFINE_int32(mouth_p2_y, FLAGS_eyes_mid_y + 32, "");
DEFINE_int32(mouth_p3_x, FLAGS_eyes_mid_x - 12, "");
DEFINE_int32(mouth_p3_y, FLAGS_eyes_mid_y + 32, "");

DEFINE_int32(chin_p0_x, FLAGS_eyes_mid_x - 12, "");
DEFINE_int32(chin_p0_y, FLAGS_eyes_mid_y + 32, "");
DEFINE_int32(chin_p1_x, FLAGS_eyes_mid_x + 12, "");
DEFINE_int32(chin_p1_y, FLAGS_eyes_mid_y + 32, "");
DEFINE_int32(chin_p2_x, FLAGS_eyes_mid_x + 12, "");
DEFINE_int32(chin_p2_y, FLAGS_eyes_mid_x + 46, "");
DEFINE_int32(chin_p3_x, FLAGS_eyes_mid_x - 12, "");
DEFINE_int32(chin_p3_y, FLAGS_eyes_mid_y + 46, "");

DEFINE_int32(cheek_p0_x, FLAGS_eyes_mid_x - 27, "");
DEFINE_int32(cheek_p0_y, FLAGS_eyes_mid_y + 15, "");
DEFINE_int32(cheek_p1_x, FLAGS_eyes_mid_x - 13, "");
DEFINE_int32(cheek_p1_y, FLAGS_eyes_mid_y + 15, "");
DEFINE_int32(cheek_p2_x, FLAGS_eyes_mid_x - 13, "");
DEFINE_int32(cheek_p2_y, FLAGS_eyes_mid_y + 35, "");
DEFINE_int32(cheek_p3_x, FLAGS_eyes_mid_x - 27, "");
DEFINE_int32(cheek_p3_y, FLAGS_eyes_mid_y + 35, "");

DEFINE_int32(cheek_p4_x, FLAGS_eyes_mid_x + 13, "");
DEFINE_int32(cheek_p4_y, FLAGS_eyes_mid_y + 15, "");
DEFINE_int32(cheek_p5_x, FLAGS_eyes_mid_x + 27, "");
DEFINE_int32(cheek_p5_y, FLAGS_eyes_mid_y + 15, "");
DEFINE_int32(cheek_p6_x, FLAGS_eyes_mid_x + 27, "");
DEFINE_int32(cheek_p6_y, FLAGS_eyes_mid_y + 35, "");
DEFINE_int32(cheek_p7_x, FLAGS_eyes_mid_x + 13, "");
DEFINE_int32(cheek_p7_y, FLAGS_eyes_mid_y + 35, "");

DEFINE_int32(hair_p0_x, FLAGS_eyes_mid_x - 26, "");
DEFINE_int32(hair_p0_y, FLAGS_eyes_mid_y - 40, "");
DEFINE_int32(hair_p1_x, FLAGS_eyes_mid_x + 26, "");
DEFINE_int32(hair_p1_y, FLAGS_eyes_mid_y - 40, "");
DEFINE_int32(hair_p2_x, FLAGS_eyes_mid_x + 26, "");
DEFINE_int32(hair_p2_y, FLAGS_eyes_mid_y - 24, "");
DEFINE_int32(hair_p3_x, FLAGS_eyes_mid_x + 18, "");
DEFINE_int32(hair_p3_y, FLAGS_eyes_mid_y - 24, "");
DEFINE_int32(hair_p4_x, FLAGS_eyes_mid_x + 18, "");
DEFINE_int32(hair_p4_y, FLAGS_eyes_mid_y - 32, "");
DEFINE_int32(hair_p5_x, FLAGS_eyes_mid_x - 18, "");
DEFINE_int32(hair_p5_y, FLAGS_eyes_mid_y - 32, "");
DEFINE_int32(hair_p6_x, FLAGS_eyes_mid_x - 18, "");
DEFINE_int32(hair_p6_y, FLAGS_eyes_mid_y - 24, "");
DEFINE_int32(hair_p7_x, FLAGS_eyes_mid_x - 26, "");
DEFINE_int32(hair_p7_y, FLAGS_eyes_mid_y - 24, "");

DEFINE_int32(face_p0_x, FLAGS_eyes_mid_x - 20, "");
DEFINE_int32(face_p0_y, FLAGS_eyes_mid_y - 14, "");
DEFINE_int32(face_p1_x, FLAGS_eyes_mid_x + 20, "");
DEFINE_int32(face_p1_y, FLAGS_eyes_mid_y - 14, "");
DEFINE_int32(face_p2_x, FLAGS_eyes_mid_x + 20, "");
DEFINE_int32(face_p2_y, FLAGS_eyes_mid_y + 36, "");
DEFINE_int32(face_p3_x, FLAGS_eyes_mid_x - 20, "");
DEFINE_int32(face_p3_y, FLAGS_eyes_mid_y + 36, "");

/************************/

/****************feature**********************/
DEFINE_int32(lbp_radius, 1, "");
DEFINE_int32(lbp_neibors, 8, "");

DEFINE_int32(RGB_hist_bins, 128, "");
DEFINE_int32(HSV_H_hist_bins, 90, "");
DEFINE_int32(HSV_SV_hist_bins, 128, "");
DEFINE_int32(intensity_hist_bins, 128, "");
DEFINE_int32(edge_mag_hist_bins, 128, "");
DEFINE_int32(edge_ori_hist_bins, 180, "");
DEFINE_int32(lbp_hist_bins, 256, "");

//DEFINE_string(feature_set_dir, "/home/harvey/Dataset/test_fe/test_feature_set/", "");
DEFINE_string(feature_set_dir, "/home/harvey/Dataset/all-low-level-features/", "");

