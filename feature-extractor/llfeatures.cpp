/*
 * llfeatures.cpp
 *
 *  Created on: 13 Jun, 2013
 *      Author: harvey
 */


#include "llfeatures.h"
#include "flags.h"
#include "utils.h"

#include <imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

template <typename _Tp> static
inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors) {
    //get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    //_dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    _dst.create(src.rows, src.cols, CV_32SC1);
    Mat dst = _dst.getMat();
    // zero
    dst.setTo(0);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbors)));
        float y = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                // calculate interpolated value
                float t = static_cast<float>(w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx));
                // floating point precision, so check some machine-dependent epsilon
                //dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
                dst.at<int>(i,j) += ((t > src.at<_Tp>(i,j)) || (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

static void elbp(InputArray src, OutputArray dst, int radius, int neighbors)
{
    int type = src.type();
    switch (type) {
    case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors); break;
    case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
    case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors); break;
    case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors); break;
    case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors); break;
    case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors); break;
    case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors); break;
    default:
        string error_msg = format("Using Original Local Binary Patterns for feature extraction only works on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
        CV_Error(CV_StsNotImplemented, error_msg);
        break;
    }
}

Mat elbp(InputArray src, int radius, int neighbors) {
    Mat dst;
    Mat src_mat, gray;
    src_mat = src.getMat();
    cvtColor(src_mat, gray, CV_BGR2GRAY);
    elbp(gray, dst, radius, neighbors);
    dst.convertTo(dst, CV_32FC1);
    return dst;
}


static Mat
histc_(const Mat& src, int channel, Mat mask, int minVal, int maxVal, int bins, bool normed=false)
{
    Mat result;
    // Establish the number of bins.
    //int histSize = maxVal-minVal+1;
    int histSize = bins;
    // Set the ranges.
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal+1) };
    const float* histRange = { range };
    // calc histogram
    calcHist(&src, 1, &channel, mask, result, 1, &histSize, &histRange, true, false);
    // normalize
//    if(normed) {
//        result /= (int)src.total();
//    }
    return result.reshape(1,1);
}

Mat histc(InputArray _src, int channel, Mat mask, int minVal, int maxVal, int bins, bool normed)
{
    Mat src = _src.getMat();
    switch (src.type()) {
        case CV_8SC1:
        case CV_8SC2:
        case CV_8SC3:
            return histc_(Mat_<float>(src), channel, mask, minVal, maxVal, bins, normed);
            break;
        case CV_8UC1:
        case CV_8UC2:
        case CV_8UC3:
            return histc_(src, channel, mask, minVal, maxVal, bins, normed);
            break;
        case CV_16SC1:
        case CV_16SC2:
        case CV_16SC3:
            return histc_(Mat_<float>(src), channel, mask, minVal, maxVal, bins, normed);
            break;
        case CV_16UC1:
        case CV_16UC2:
        case CV_16UC3:
            return histc_(src, channel, mask, minVal, maxVal, bins, normed);
            break;
        case CV_32SC1:
        case CV_32SC2:
        case CV_32SC3:
            return histc_(Mat_<float>(src), channel, mask, minVal, maxVal, bins, normed);
            break;
        case CV_32FC1:
        case CV_32FC2:
        case CV_32FC3:
            return histc_(src, channel, mask, minVal, maxVal, bins, normed);
            break;
        default:
            CV_Error(CV_StsUnmatchedFormats, "This type is not implemented yet."); break;
    }
    return Mat();
}


Mat edge_operator(InputArray src)
{
	Mat dst;
	Canny(src, dst, FLAGS_edge_detect_threshold1, FLAGS_edge_detect_threshold2, FLAGS_edge_detect_aperture_size, FLAGS_edge_detect_l2gradient);
	return dst;
}

Mat edge_orientation(InputArray src)
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	Mat src_mat = src.getMat();
	Mat gray;
	cvtColor(src_mat, gray, CV_BGR2GRAY);

	blur(gray, gray, Size(3,3));

	Mat canny_edge = edge_operator(gray);

	Sobel( gray, grad_x, CV_32F, 1, 0);
	convertScaleAbs( grad_x, abs_grad_x );

	Sobel( gray, grad_y, CV_32F, 0, 1);
	convertScaleAbs( grad_y, abs_grad_y );


	Mat dst(src_mat.rows, src_mat.cols, CV_32FC1);
	for (int i = 0; i < (int)dst.rows; i++)
	{
		for (int j = 0; j < (int)dst.cols; j++)
		{
			dst.at<float>(i, j) = atan2(grad_y.at<float>(i, j), grad_x.at<float>(i, j)) * 180 / M_PI;
			if (dst.at<float>(i, j) < 0.0)
			{
				dst.at<float>(i, j) += 360.0;
			}

//			if (abs(grad_y.at<float>(i, j)) < 0.00001 && abs(grad_x.at<float>(i, j)) < 0.00001)
//			{
//				dst.at<float>(i, j) = -3;
//			}

      if (canny_edge.at<unsigned char>(i, j) ==  0)
      {
        dst.at<float>(i, j) = 0;
      }
		}
	}

//  for (int i = 0; i < (int)dst.rows; i++)
//  {
//    for (int j = 0; j < (int)dst.cols; j++)
//    {
//      if (canny_edge.at<unsigned char>(i, j) == 0)
//      {
//        dst.at<float>(i, j) = 0;
//      }
//    }
//  }

	return dst;
}


Mat edge_magnitude(InputArray src)
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat gray;
	Mat src_mat = src.getMat();

	cvtColor(src_mat, gray, CV_BGR2GRAY);

	blur(gray, gray, Size(3, 3));            //

	Mat canny_edge = edge_operator(gray);

	Sobel( gray, grad_x, CV_32F, 1, 0);
	convertScaleAbs( grad_x, abs_grad_x );


	Sobel( gray, grad_y, CV_32F, 0, 1);
	convertScaleAbs( grad_y, abs_grad_y );


	Mat dst(src_mat.rows, src_mat.cols, CV_32FC1);

	for (int i = 0; i < (int)dst.rows; i++)
	{
		for (int j = 0; j < (int)dst.cols; j++)
		{
			dst.at<float>(i, j) = sqrt(pow(grad_x.at<float>(i, j), 2) + pow(grad_y.at<float>(i, j), 2));

			if (canny_edge.at<unsigned char>(i, j) == 0)
			{
			  dst.at<float>(i, j) = 0;
			}
		}
	}

	normalize(dst, dst, 0, FLAGS_edge_mag_hist_bins - 1, NORM_MINMAX);

	for (int i = 0; i < (int)dst.rows; i++)
	{
	  for (int j = 0; j < (int)dst.cols; j++)
	  {
      if (canny_edge.at<unsigned char>(i, j) == 0)
      {
        dst.at<float>(i, j) = 0;
      }
	  }
	}

	return dst;
}

Mat hist_rgb2(Mat src, Mat mask, int mask_size)
{
  Mat result(3, FLAGS_RGB_hist_bins, CV_32FC1);
  Mat hist_r = histc(src, 0, mask, 0, 255, FLAGS_RGB_hist_bins, false);
  hist_r.convertTo(result.row(0), CV_32FC1);
  Mat hist_g = histc(src, 1, mask, 0, 255, FLAGS_RGB_hist_bins, false);
  hist_g.convertTo(result.row(1), CV_32FC1);
  Mat hist_b = histc(src, 2, mask, 0, 255, FLAGS_RGB_hist_bins, false);
  hist_b.convertTo(result.row(2), CV_32FC1);

//  result /= mask_size;
  //result *= 100;


  return result.reshape(1, 1);
}

Mat hist_hsv2(Mat src, Mat mask, int mask_size)
{
//  vector<float> fv;
  Mat hist_h = histc(src, 0, mask, 0, 179, FLAGS_HSV_H_hist_bins, false);
//  for (int i = 0; i < hist_h.cols; i++)
//  {
//    if (isnan(hist_h.at<float>(i)))
//    {
//      cerr << "NaN Error: H at " << i << endl;
//      return Mat();
//    }
//    fv.push_back(hist_h.at<float>(i));
//  }
  Mat hist_s = histc(src, 1, mask, 0, 255, FLAGS_HSV_SV_hist_bins, false);
//  for (int i = 0; i < hist_s.cols; i++)
//  {
//    if (isnan(hist_s.at<float>(i)))
//    {
//      cerr << "NaN Error: S at " << i << endl;
//      return Mat();
//    }
//    fv.push_back(hist_s.at<float>(i));
//  }
  Mat hist_v = histc(src, 2, mask, 0, 255, FLAGS_HSV_SV_hist_bins, false);
//  for (int i = 0; i < hist_v.cols; i++)
//  {
//    if (isnan(hist_v.at<float>(i)))
//    {
//      cerr << "NaN Error: V at " << i << endl;
//      return Mat();
//    }
//    fv.push_back(hist_v.at<float>(i));
//  }

  int total_length = hist_h.cols + hist_s.cols + hist_v.cols;

  Mat result(1, total_length, hist_h.type());

  for (int i = 0; i < hist_h.cols; i++)
  {
    result.at<float>(i) = hist_h.at<float>(i);
  }

  for (int i = 0; i < hist_s.cols; i++)
  {
    result.at<float>(i + hist_h.cols) = hist_v.at<float>(i);
  }

  for (int i = 0; i < hist_v.cols; i++)
  {
    result.at<float>(i + hist_h.cols + hist_v.cols) = hist_v.at<float>(i);
  }

  //Mat result(fv, true);

//  result /= mask_size;
//  result *= 100;
//  return result.reshape(1, 1);
  return result;
}

Mat hist_intensity(Mat src, Mat mask, int mask_size)
{
  Mat result = histc(src, 0, mask, 0, 255, FLAGS_intensity_hist_bins, false);
//  result /= mask_size;
//  result *= 100;
  return result;
}

Mat hist_edge_magnitude(Mat src, Mat mask, int mask_size)
{
  double max_val;
  minMaxLoc(src, NULL, &max_val);

  int max_val_int = round(max_val);

  Mat result = histc(src, 0, mask, 0, max_val_int, FLAGS_edge_mag_hist_bins, false);

//  result /= mask_size;
//  result *= 100;
  return result;
}

Mat hist_edge_orientation(Mat src, Mat mask, int mask_size)
{
  Mat result = histc(src, 0, mask, 0, 359, FLAGS_edge_ori_hist_bins, false);
//  result /= mask_size;
 // result *= 100;
  return result;
}

Mat hist_lbp(Mat src, Mat mask, int mask_size)
{
  int numPatterns = static_cast<int>(pow(2.0, FLAGS_lbp_neibors));
  Mat result = histc(src, 0, mask, 0, numPatterns - 1, FLAGS_lbp_hist_bins, false);
//  result /= mask_size;
//  result *= 100;
  return result;
}


Mat rgb_type(Mat orig)
{
   Mat dst;
   cvtColor(orig, dst, CV_BGR2RGB);

   dst.convertTo(dst, CV_32FC3);

   return dst;
}

Mat intensity_type(Mat orig)
{
  Mat dst;

  cvtColor(orig, dst, CV_BGR2GRAY);
  dst.convertTo(dst, CV_32FC1);
  return dst;
}

Mat hsv_type(Mat orig)
{
  Mat dst;
  cvtColor(orig, dst, CV_BGR2HSV);

  dst.convertTo(dst, CV_32FC3);
  return dst;
}

Mat rgb_stat(Mat src, Mat mask, int mask_size)
{
  Scalar mean, dev;
  meanStdDev(src, mean, dev, mask);

  Mat result(1, 3 * 2, CV_32FC1);
  result.at<float>(0, 0) = static_cast<float>(mean.val[0]);
  result.at<float>(0, 1) = static_cast<float>(dev.val[0]);
  result.at<float>(0, 2) = static_cast<float>(mean.val[1]);
  result.at<float>(0, 3) = static_cast<float>(dev.val[1]);
  result.at<float>(0, 4) = static_cast<float>(mean.val[2]);
  result.at<float>(0, 5) = static_cast<float>(dev.val[2]);

  return result;
}

Mat hsv_stat(Mat src, Mat mask, int mask_size)
{
  Scalar mean, dev;
  meanStdDev(src, mean, dev, mask);

  Mat result(1, 3 * 2, CV_32FC1);
  result.at<float>(0, 0) = static_cast<float>(mean.val[0]);
  result.at<float>(0, 1) = static_cast<float>(dev.val[0]);
  result.at<float>(0, 2) = static_cast<float>(mean.val[1]);
  result.at<float>(0, 3) = static_cast<float>(dev.val[1]);
  result.at<float>(0, 4) = static_cast<float>(mean.val[2]);
  result.at<float>(0, 5) = static_cast<float>(dev.val[2]);

  return result;
}

Mat intensity_stat(Mat src, Mat mask, int mask_size)
{
  Scalar mean, dev;
  meanStdDev(src, mean, dev, mask);

  Mat result(1, 2, CV_32FC1);
  result.at<float>(0, 0) = static_cast<float>(mean.val[0]);
  result.at<float>(0, 1) = static_cast<float>(dev.val[0]);

  return result;
}

Mat edge_magnitude_stat(Mat src, Mat mask, int mask_size)
{
  Scalar mean, dev;
  meanStdDev(src, mean, dev, mask);

  Mat result(1, 2, CV_32FC1);
  result.at<float>(0, 0) = static_cast<float>(mean.val[0]);
  result.at<float>(0, 1) = static_cast<float>(dev.val[0]);

  return result;
}

Mat edge_orientation_stat(Mat src, Mat mask, int mask_size)
{
  Scalar mean, dev;
  meanStdDev(src, mean, dev, mask);

  Mat result(1, 2, CV_32FC1);
  result.at<float>(0, 0) = static_cast<float>(mean.val[0]);
  result.at<float>(0, 1) = static_cast<float>(dev.val[0]);

  return result;
}

Mat lbp_stat(Mat src, Mat mask, int mask_size)
{
  Scalar mean, dev;
  meanStdDev(src, mean, dev, mask);

  Mat result(1, 2, CV_32FC1);
  result.at<float>(0, 0) = static_cast<float>(mean.val[0]);
  result.at<float>(0, 1) = static_cast<float>(dev.val[0]);

  return result;
}


int
none_aggregation32fc3(Mat src, int channel, Mat mask, vector<float> &fv)
{
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      if (mask.at<unsigned char>(i, j) == 0)
      {
        continue;
      }

      fv.push_back(src.at<Vec3f>(i, j)[channel]);
    }
  }

  return 0;
}

int
none_aggregation32fc1(Mat src, Mat mask, vector<float> &fv)
{
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      if (mask.at<unsigned char>(i, j) == 0)
      {
        continue;
      }

      fv.push_back(src.at<float>(i, j));
    }
  }

  return 0;
}

Mat rgb_none_agg(Mat src, Mat mask, int mask_size)
{
  vector<float> pixels;
  none_aggregation32fc3(src, 0, mask, pixels);
  none_aggregation32fc3(src, 1, mask, pixels);
  none_aggregation32fc3(src, 2, mask, pixels);

  return Mat(pixels).t();
}

Mat hsv_none_agg(Mat src, Mat mask, int mask_size)
{
  vector<float> pixels;
  none_aggregation32fc3(src, 0, mask, pixels);
  none_aggregation32fc3(src, 1, mask, pixels);
  none_aggregation32fc3(src, 2, mask, pixels);

  return Mat(pixels).t();
}

Mat intensity_none_agg(Mat src, Mat mask, int mask_size)
{
  vector<float> pixels;
  none_aggregation32fc1(src, mask, pixels);
  return Mat(pixels).t();
}

Mat edge_magnitude_none_agg(Mat src, Mat mask, int mask_size)
{
  vector<float> pixels;
  none_aggregation32fc1(src, mask, pixels);
  return Mat(pixels).t();
}

Mat edge_orientation_none_agg(Mat src, Mat mask, int mask_size)
{
  vector<float> pixels;
  none_aggregation32fc1(src, mask, pixels);
  return Mat(pixels).t();
}

Mat lbp_none_agg(Mat src, Mat mask, int mask_size)
{
  vector<float> pixels;
  none_aggregation32fc1(src, mask, pixels);
  return Mat(pixels).t();
}

Mat normalization_none(Mat src)
{
  return src;
}
//
//Mat normalization_mean(Mat src, Scalar mean, Scalar dev)
//{
//  vector<Mat> channels(src.channels());
//  split(src, channels);
//
//  for (int i = 0; i < src.channels(); i++)
//  {
//    channels[i] /= mean.val[i];
//  }
//
//  Mat dst;
//  merge(channels, dst);
//  return dst;
//}
//
//Mat normalization_energy(Mat src, Scalar mean, Scalar dev)
//{
//  vector<Mat> channels(src.channels());
//  split(src, channels);
//
//  for (int i = 0; i < src.channels(); i++)
//  {
//    channels[i] -= mean.val[i];
//    channels[i] /= dev.val[i];
//  }
//
//  Mat dst;
//  merge(channels, dst);
//  return dst;
//}

Mat local_equalize_hist(Mat src)
{
  if (src.type() != CV_8UC1)
  {
    return Mat();
  }

  Mat floatGray;
  src.convertTo(floatGray, CV_32FC1, 1.0 / 255.0);
  double sigma1 = 2, sigma2 = 20;


  Mat blurred1, blurred2, temp1, temp2, res;
  int blur1 = 2*ceil(-NormInv(0.05, 0, sigma1))+1;
  cv::GaussianBlur(floatGray, blurred1, cv::Size(blur1,blur1), sigma1);
  temp1 = floatGray-blurred1;

  cv::pow(temp1, 2.0, temp2);
  int blur2 = 2*ceil(-NormInv(0.05, 0, sigma2))+1;
  cv::GaussianBlur(temp2, blurred2, cv::Size(blur2,blur2), sigma2);

  cv::pow(blurred2, 0.5, temp2);
  floatGray = temp1/temp2;
  normalize(floatGray, res, 0, 255, NORM_MINMAX, CV_8UC1);

  return res;
}



Mat equalize_hist(Mat src)
{

  if(src.channels() >= 3)
  {
      Mat ycrcb;

      cvtColor(src,ycrcb,CV_BGR2YCrCb);

      vector<Mat> channels;
      split(ycrcb,channels);

      //equalizeHist(channels[0], channels[0]);
      channels[0] = local_equalize_hist(channels[0]);

      Mat result;
      merge(channels,ycrcb);

      cvtColor(ycrcb,result,CV_YCrCb2BGR);

      return result;
  }
  else
  {
    cout << "equalize_hist: too few channels" << endl;
  }
  return Mat();

}

Mat equalize_hist2(Mat src)
{
  if(src.channels() >= 3)
   {

       vector<Mat> channels;
       split(src,channels);

       for (int i = 0; i < src.channels(); i++)
       {
         equalizeHist(channels[i], channels[i]);
       }

       Mat result;
       merge(channels,result);

       return result;
   }
   else
   {
     cout << "equalize_hist: too few channels" << endl;
   }
   return Mat();
}
