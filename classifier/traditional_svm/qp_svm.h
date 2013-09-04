/*
 * QpSvm.h
 *
 *  Created on: 21 Aug, 2013
 *      Author: harvey
 */

#ifndef QPSVM_H_
#define QPSVM_H_

#include "../weak_classifier.h"
#include "svm.h"

class QpSvm : public WeakClassifier
{
public:
  QpSvm(uint64);
  QpSvm();
  ~QpSvm();
  int learn(const vector<tr1::shared_ptr<Sample> > &training_samples);
  int learn(tr1::shared_ptr<Sample> training_sample);
  int test(const vector<tr1::shared_ptr<Sample> > &test_samples,
                   vector<bool> &test_result, double &total_error_rate,
                   double &false_pos_rate, double &false_neg_rate,
                   double &true_pos_rate, double &true_neg_rate);
  int test(tr1::shared_ptr<Sample> test_sample, double &prediction);
  int set_alpha(double a);
  void reset();
  int detach_temp_variables();

  int alpha(double &alpha);
  int set_feature_type(uint64);
  int feature_type(uint64&);
  int get_last_test_result(std::vector<bool>&);
  int calc_weighted_error(const vector<double> &weights, double &weighted_error);
  int weighted_error(double &error);

//  int learn(const svm_problem &problem_set);

  svm_problem     m_problem_set;
//  svm_node        *m_space;
  svm_parameter   m_param;
  svm_model       *m_model;
  int             m_fv_length;

  int convert_to_internal_vectors(const vector<tr1::shared_ptr<Sample> > &training_set, svm_problem &prolem_set);
  int free_problem_set(svm_problem &problem_set);
  int convert_to_internal_vector(tr1::shared_ptr<Sample> this_sample, svm_node **internal_vector);
  int free_internal_vector(svm_node *v);

private:


  vector<bool> m_last_test_result;
//  uint64       m_feature_type_id;
//  double       m_alpha;
};

#endif /* QPSVM_H_ */
