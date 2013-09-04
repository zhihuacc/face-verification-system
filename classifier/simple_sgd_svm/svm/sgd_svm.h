/*
 * sgd_svm.h
 *
 *  Created on: 27 Apr, 2013
 *      Author: harvey
 */

#ifndef SGD_SVM_H_
#define SGD_SVM_H_

//#include <iostream>
#include <vector>
#include <map>
#include <tr1/memory>
#include "sample.h"
#include "../../weak_classifier.h"

//#include "classifier_interface.h"

#include "../../types.h"

using namespace std;

#define MAX_TRAINING_SAMPLE_SIZE 100000


class SgdSvm : public WeakClassifier
{
public:
  SgdSvm();
	SgdSvm(uint64 feature_type, double lambda, int t, int batch_size, int num_iter_to_avg);
	SgdSvm(uint64 feature_type);
  void reset();
//  int LoadTrainingSampleFile(const string &file_name, int max_rows);
//  int LoadTestSampleFile(const string &file_name, int max_rows);
//  int load_training_samples(vector<std::tr1::shared_ptr<Sample> > &samples);
//  int load_test_samples(vector<std::tr1::shared_ptr<Sample> > &samples);
//  int collect_training_feature_vectors(const map<unsigned long long, tr1::shared_ptr<Sample> >&all_fv, const vector<unsigned long long> &pos_images, const vector<unsigned long long> &neg_images);
//
//  int collect_test_feature_vectors(const map<unsigned long long, tr1::shared_ptr<Sample> > &all_fv, const vector<unsigned long long> &pos_images, const vector<unsigned long long> &neg_images);
//
//  void detach_training_samples();
//  void detach_test_samples();

  void PrintLearningParameters();
//  int Learn();
//  double Test();
//  double test(tr1::shared_ptr<Sample> sample);

  int learn(const vector<tr1::shared_ptr<Sample> > &training_samples);
  int learn(tr1::shared_ptr<Sample> training_sample);
  int test(const vector<tr1::shared_ptr<Sample> > &test_samples,
                   vector<bool> &test_result, double &total_error_rate,
                   double &false_pos_rate, double &false_neg_rate,
                   double &true_pos_rate, double &true_neg_rate);
  int test(tr1::shared_ptr<Sample> test_sample, double &prediction);
//  virtual int cross_validate(const vector<tr1::shared_ptr<Sample> > &training_samples);

  int set_feature_type(uint64 feature_type);
  int set_alpha(double a);
  int alpha(double &alpha);
  int feature_type(uint64 &feature_type);
  int weighted_error(double &error);
  int calc_weighted_error(const vector<double> &weights, double &weighted_error);
  int get_last_test_result(vector<bool> &last_test_result);
  int detach_temp_variables();
	int shuffle();
	void set_T(int t);
	void set_lambda(double lambda);
	void set_batch_size(int n);
	void set_num_iter_to_average(int n);


//	void set_max_feature_vector_dim(int dim);


//  void shrink_test_result_set();
//
//  const vector<bool> &get_test_result();
//  int get_positive_sample_num();
//  int get_negative_sample_num();
//
//  double get_test_error_rate();
  friend ostream &operator<<(ostream &f, SgdSvm &svm);
  friend istream &operator>>(istream &f, SgdSvm &svm);
private:
//	int LoadSampleFile(const string &file_name, int max_rows, vector<std::tr1::shared_ptr<Sample> > &sample_set, int &max_dim);

//	int      m_max_feature_vector_dim;
//  vector<std::tr1::shared_ptr<Sample> > m_training_sample_set;
//  int      m_positive_sample_num;
//  int      m_negative_sample_num;
//  vector<std::tr1::shared_ptr<Sample> > m_test_sample_set;


  double   m_lambda;
  int      m_T;
  int      m_batch_size;
  int      m_num_iter_to_average;

  DensityVector m_learned_weights;
  double   m_learned_bias;
  DensityVector m_learned_average_weights;
  double   m_learned_average_bias;

  vector<bool> m_last_test_result;
  double m_last_test_error_rate;
};


#endif /* SGD_SVM_H_ */
