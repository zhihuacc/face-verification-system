/*
 * utils.h
 *
 *  Created on: 16 Jun, 2013
 *      Author: harvey
 */

#ifndef UTILS_H_
#define UTILS_H_


template<class T>
float compute_mean(InputArray src, int channel, Mat mask)
{
    Mat src_mat = src.getMat();
    float sum = 0;
    int   count = 0;
    for (int i = 0; i < src_mat.rows; i++)
    {
      for (int j = 0; j < src_mat.cols; j++)
      {
            if (mask.at<unsigned char>(i, j, channel) == 0)
            {
              continue;
            }
            count++;
            sum += src_mat.at<T>(i, j);
      }
    }

    return sum / count;
}

template<class T>
float compute_var(InputArray src, int channel, Mat mask)
{
  Mat src_mat = src.getMat();
    float sum = 0;
    int count = 0;
    float mean = compute_mean<T>(src, mask, channel);
    for (int i = 0; i < src_mat.rows; i++)
    {
      for (int j = 0; j < src_mat.cols; j++)
      {
        if (mask.at<unsigned char>(i, j, channel) == 0)
        {
          continue;
        }

        count++;
        sum += pow(src_mat.at<T>(i, j) - mean, 2);
      }
    }
  return sum / count;
}

double NormInv(double probability, double mean, double sigma);
double NormInv(double probability);


#endif /* UTILS_H_ */
