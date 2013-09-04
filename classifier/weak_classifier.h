/*
 * weak_classifier.h
 *
 *  Created on: 25 Jun, 2013
 *      Author: harvey
 */

#ifndef WEAK_CLASSIFIER_H_
#define WEAK_CLASSIFIER_H_

#include "classifier_interface.h"
#include "types.h"
#include <vector>
#include "svm.h"

using namespace std;


class WeakClassifier:public Classifier
{
public:
  WeakClassifier(uint64 id);
  virtual ~WeakClassifier();

  virtual int learn(const vector<tr1::shared_ptr<Sample> > &training_samples) = 0;
  virtual int learn(tr1::shared_ptr<Sample> training_sample) = 0;
  virtual int test(const vector<tr1::shared_ptr<Sample> > &test_samples,
                   vector<bool> &test_result, double &total_error_rate,
                   double &false_pos_rate, double &false_neg_rate,
                   double &true_pos_rate, double &true_neg_rate) = 0;
  virtual int test(tr1::shared_ptr<Sample> test_sample, double &prediction) = 0;
//  virtual int cross_validate(const vector<tr1::shared_ptr<Sample> > &training_samples);

  virtual int set_alpha(double alpha) = 0;
  virtual int alpha(double &alhpa) = 0;
  virtual int set_feature_type(uint64 feature_type)  = 0;
  virtual int feature_type(uint64 &feature_type) = 0;
  virtual int weighted_error(double &error) = 0;
  virtual int calc_weighted_error(const vector<double> &weights, double &weighted_error) = 0;
  virtual int get_last_test_result(vector<bool> &last_test_result) = 0;

//  virtual int learn(const svm_problem &problem_set);

//  int collect_training_feature_vectors(const map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors, const vector<uint> &pos_instance_id, const vector<uint> &neg_instance_id);
//  int collect_test_feature_vectors(const map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors, const vector<uint> &pos_instance_id, const vector<uint> &neg_instance_id);
//
//
//
//  const vector<bool> &get_test_result();
//  int shrink_test_result_set();

protected:
  uint64 m_feature_type_id;

//  int  m_max_feature_vector_dim;
////  vector<std::tr1::shared_ptr<Sample> > m_training_sample_set;
//  int  m_training_positive_sample_num;
//  int  m_training_negative_sample_num;
////  vector<std::tr1::shared_ptr<Sample> > m_test_sample_set;
//  int  m_test_postive_sample_num;
//  int  m_test_negative_sample_num;


//  std::tr1::shared_ptr<SgdSvm> m_svm;
  double m_alpha;
//  double m_error_rate;
  double m_weighted_error;
};

#endif /* WEAK_CLASSIFIER_H_ */
