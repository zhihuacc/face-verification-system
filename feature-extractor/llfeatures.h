/*
 * llfeatures.h
 *
 *  Created on: 13 Jun, 2013
 *      Author: harvey
 */

#include <core/core.hpp>
using namespace cv;

Mat elbp(InputArray src, int radius, int neighbors);
Mat spatial_histogram(InputArray _src, int numPatterns, int grid_x, int grid_y, bool /*normed*/);

Mat edge_operator(InputArray src);

Mat edge_orientation(InputArray src);
Mat edge_magnitude(InputArray src);
Mat rgb_type(Mat src);
Mat intensity_type(Mat src);
Mat hsv_type(Mat src);

Mat hist_rgb2(Mat src, Mat mask, int mask_size);
Mat hist_hsv2(Mat src, Mat mask, int mask_size);
Mat hist_intensity(Mat src, Mat mask, int mask_size);
Mat hist_edge_orientation(Mat src, Mat mask, int mask_size);
Mat hist_edge_magnitude(Mat src, Mat mask, int mask_size);
Mat hist_lbp(Mat src, Mat mask, int mask_size);



Mat rgb_stat(Mat src, Mat mask, int mask_size);
Mat hsv_stat(Mat src, Mat mask, int mask_size);
Mat intensity_stat(Mat src, Mat mask, int mask_size);
Mat edge_magnitude_stat(Mat src, Mat mask, int mask_size);
Mat edge_orientation_stat(Mat src, Mat mask, int mask_size);
Mat lbp_stat(Mat src, Mat mask, int mask_size);

Mat rgb_none_agg(Mat src, Mat mask, int mask_size);
Mat hsv_none_agg(Mat src, Mat mask, int mask_size);
Mat intensity_none_agg(Mat src, Mat mask, int mask_size);
Mat edge_magnitude_none_agg(Mat src, Mat mask, int mask_size);
Mat edge_orientation_none_agg(Mat src, Mat mask, int mask_size);
Mat lbp_none_agg(Mat src, Mat mask, int mask_size);

//Mat histc(InputArray _src, int minVal, int maxVal, bool normed);
Mat histc(InputArray _src, int channel, Mat mask, int minVal, int maxVal, int bins, bool normed);
Mat equalize_hist(Mat src);
Mat equalize_hist2(Mat src);

Mat local_equalize_hist(Mat src);

Mat normalization_none(Mat src);
Mat normalization_mean(Mat src, Scalar mean, Scalar dev);
Mat normalization_energy(Mat src, Scalar mean, Scalar dev);
