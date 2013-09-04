/*
 * binary_classifier_interface.h
 *
 *  Created on: 13 Jul, 2013
 *      Author: harvey
 */

#ifndef CLASSIFIER_INTERFACE_H_
#define CLASSIFIER_INTERFACE_H_

#include <tr1/memory>
#include <vector>
#include "sample.h"

using namespace std;

class Classifier
{
public:
  virtual ~Classifier();
  virtual int learn(const vector<tr1::shared_ptr<Sample> > &training_samples) = 0;
  virtual int learn(tr1::shared_ptr<Sample> training_sample) = 0;
  virtual int test(const vector<tr1::shared_ptr<Sample> > &test_samples,
                   vector<bool> &test_result, double &total_error_rate,
                   double &false_pos_rate, double &false_neg_rate,
                   double &true_pos_rate, double &true_neg_rate) = 0;
  virtual int test(tr1::shared_ptr<Sample> test_sample, double &prediction) = 0;
//  virtual int cross_validate(const vector<tr1::shared_ptr<Sample> > &training_samples);

};

#endif

