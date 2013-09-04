/*
 * flags.h
 *
 *  Created on: 6 Jun, 2013
 *      Author: harvey
 */

#ifndef FLAGS_H_
#define FLAGS_H_


#include <gflags/gflags.h>

DECLARE_double(face_margin_top);
DECLARE_double(face_margin_left);
DECLARE_double(face_margin_right);
DECLARE_double(face_margin_bottom);
DECLARE_int32(max_face_horizontal_degree);
//DECLARE_int32(face_normalized_width);
//DECLARE_int32(face_normalized_height);
//
//DECLARE_double(eye_ratio);
//DECLARE_double(eye_h_w_ratio);
//DECLARE_double(mouth_ratio);
//DECLARE_double(mouth_h_w_ratio);
//DECLARE_double(nose_ratio);
//DECLARE_double(nose_h_w_ratio);
//DECLARE_double(eye_brow_ratio);
//DECLARE_double(eye_brow_h_w_ratio);
//
//DECLARE_double(cheek_ratio);
//DECLARE_double(cheek_h_w_ratio);
//
//DECLARE_double(chin_ratio);
//DECLARE_double(chin_h_w_ratio);
//
//DECLARE_double(hair_ratio);
//DECLARE_double(hair_h_w_ratio);
//DECLARE_double(hair_dist);
//
//DECLARE_double(whole_face_ratio);
//DECLARE_double(whole_face_h_w_ratio);

DECLARE_double(edge_detect_threshold1);
DECLARE_double(edge_detect_threshold2);
DECLARE_double(edge_detect_aperture_size);
DECLARE_bool(edge_detect_l2gradient);

DECLARE_int32(eye_width);
DECLARE_int32(eye_height);

DECLARE_int32(nose_upper_width);
DECLARE_int32(nose_bottom_width);
DECLARE_int32(nose_height);

DECLARE_int32(mouth_width);
DECLARE_int32(mouth_height);

DECLARE_int32(chin_width);
DECLARE_int32(chin_height);
DECLARE_double(chin_dist);

DECLARE_int32(cheek_width);
DECLARE_int32(cheek_height);
DECLARE_double(cheek_dist);

DECLARE_int32(eye_brow_width);
DECLARE_int32(eye_brow_height);
DECLARE_double(eye_brow_dist);
DECLARE_double(eye_brow_dist);

DECLARE_int32(hair_upper_width);
DECLARE_int32(hair_mid_width);
DECLARE_int32(hair_height);
DECLARE_int32(hair_mid_height);
DECLARE_double(hair_dist);

DECLARE_int32(whole_face_width);
DECLARE_int32(whole_face_height);

DECLARE_int32(face_normalized_max_border);
DECLARE_double(region_ratio_scale);

/****************************/

DECLARE_int32(eyes_mid_x);
DECLARE_int32(eyes_mid_y);
DECLARE_int32(two_eyes_dist);
DECLARE_int32(normalized_face_width);

DECLARE_int32(eye_p0_x);
DECLARE_int32(eye_p0_y);
DECLARE_int32(eye_p1_x);
DECLARE_int32(eye_p1_y);
DECLARE_int32(eye_p2_x);
DECLARE_int32(eye_p2_y);
DECLARE_int32(eye_p3_x);
DECLARE_int32(eye_p3_y);

DECLARE_int32(eye_p4_x);
DECLARE_int32(eye_p4_y);
DECLARE_int32(eye_p5_x);
DECLARE_int32(eye_p5_y);
DECLARE_int32(eye_p6_x);
DECLARE_int32(eye_p6_y);
DECLARE_int32(eye_p7_x);
DECLARE_int32(eye_p7_y);

DECLARE_int32(eyebrow_p0_x);
DECLARE_int32(eyebrow_p0_y);
DECLARE_int32(eyebrow_p1_x);
DECLARE_int32(eyebrow_p1_y);
DECLARE_int32(eyebrow_p2_x);
DECLARE_int32(eyebrow_p2_y);
DECLARE_int32(eyebrow_p3_x);
DECLARE_int32(eyebrow_p3_y);

DECLARE_int32(eyebrow_p4_x);
DECLARE_int32(eyebrow_p4_y);
DECLARE_int32(eyebrow_p5_x);
DECLARE_int32(eyebrow_p5_y);
DECLARE_int32(eyebrow_p6_x);
DECLARE_int32(eyebrow_p6_y);
DECLARE_int32(eyebrow_p7_x);
DECLARE_int32(eyebrow_p7_y);

DECLARE_int32(nose_p0_x);
DECLARE_int32(nose_p0_y);
DECLARE_int32(nose_p1_x);
DECLARE_int32(nose_p1_y);
DECLARE_int32(nose_p2_x);
DECLARE_int32(nose_p2_y);
DECLARE_int32(nose_p3_x);
DECLARE_int32(nose_p3_y);

DECLARE_int32(mouth_p0_x);
DECLARE_int32(mouth_p0_y);
DECLARE_int32(mouth_p1_x);
DECLARE_int32(mouth_p1_y);
DECLARE_int32(mouth_p2_x);
DECLARE_int32(mouth_p2_y);
DECLARE_int32(mouth_p3_x);
DECLARE_int32(mouth_p3_y);

DECLARE_int32(chin_p0_x);
DECLARE_int32(chin_p0_y);
DECLARE_int32(chin_p1_x);
DECLARE_int32(chin_p1_y);
DECLARE_int32(chin_p2_x);
DECLARE_int32(chin_p2_y);
DECLARE_int32(chin_p3_x);
DECLARE_int32(chin_p3_y);

DECLARE_int32(cheek_p0_x);
DECLARE_int32(cheek_p0_y);
DECLARE_int32(cheek_p1_x);
DECLARE_int32(cheek_p1_y);
DECLARE_int32(cheek_p2_x);
DECLARE_int32(cheek_p2_y);
DECLARE_int32(cheek_p3_x);
DECLARE_int32(cheek_p3_y);

DECLARE_int32(cheek_p4_x);
DECLARE_int32(cheek_p4_y);
DECLARE_int32(cheek_p5_x);
DECLARE_int32(cheek_p5_y);
DECLARE_int32(cheek_p6_x);
DECLARE_int32(cheek_p6_y);
DECLARE_int32(cheek_p7_x);
DECLARE_int32(cheek_p7_y);

DECLARE_int32(hair_p0_x);
DECLARE_int32(hair_p0_y);
DECLARE_int32(hair_p1_x);
DECLARE_int32(hair_p1_y);
DECLARE_int32(hair_p2_x);
DECLARE_int32(hair_p2_y);
DECLARE_int32(hair_p3_x);
DECLARE_int32(hair_p3_y);
DECLARE_int32(hair_p4_x);
DECLARE_int32(hair_p4_y);
DECLARE_int32(hair_p5_x);
DECLARE_int32(hair_p5_y);
DECLARE_int32(hair_p6_x);
DECLARE_int32(hair_p6_y);
DECLARE_int32(hair_p7_x);
DECLARE_int32(hair_p7_y);

DECLARE_int32(face_p0_x);
DECLARE_int32(face_p0_y);
DECLARE_int32(face_p1_x);
DECLARE_int32(face_p1_y);
DECLARE_int32(face_p2_x);
DECLARE_int32(face_p2_y);
DECLARE_int32(face_p3_x);
DECLARE_int32(face_p3_y);


/***************************/

DECLARE_int32(lbp_radius);
DECLARE_int32(lbp_neibors);

DECLARE_int32(RGB_hist_bins);
DECLARE_int32(HSV_H_hist_bins);
DECLARE_int32(HSV_SV_hist_bins);
DECLARE_int32(intensity_hist_bins);
DECLARE_int32(edge_mag_hist_bins);
DECLARE_int32(edge_ori_hist_bins);
DECLARE_int32(lbp_hist_bins);

DECLARE_string(feature_set_dir);

#endif /* FLAGS_H_ */
