/*
 * strong_classifier.h
 *
 *  Created on: 13 Jul, 2013
 *      Author: harvey
 */

#ifndef ATTR_CLASSIFIER_H_
#define ATTR_CLASSIFIER_H_

#include <tr1/memory>
#include <map>
#include <vector>
#include "types.h"
#include "classifier_interface.h"
#include "adaboost.h"

using namespace std;

class AttrClassifier : public Classifier, public AdaBoost2
{
public:
  AttrClassifier(uint64 attr):Classifier(), AdaBoost2(), m_attr_id(attr) {}

  ~AttrClassifier();
  int learn(const vector<tr1::shared_ptr<Sample> > &training_samples);
  int learn(tr1::shared_ptr<Sample> training_sample);
  int test(const vector<tr1::shared_ptr<Sample> > &test_samples,
                   vector<bool> &test_result, double &total_error_rate,
                   double &false_pos_rate, double &false_neg_rate,
                   double &true_pos_rate, double &true_neg_rate);
  int test(tr1::shared_ptr<Sample> test_sample, double &prediction);

//  int assign_training_data(const map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors, const vector<uint> &pos_instance_id, const vector<uint> &neg_instance_id);

  int load_all_training_samples(const string &lables_dir_name, const string &fv_set_dir);
  int load_all_test_samples(const string &lables_dir_name, const string &fv_set_dir);
  int detach_training_data();
  int detach_test_data();

  void assign_total_training_instance_id(const vector<uint64> &all_pos_instance_id, const vector<uint64> &all_neg_instance_id);
  void assign_this_time_training_instance_id(const vector<uint64> &training_pos_instance_id,
                            const vector<uint64> &training_neg_instance_id,
                            const vector<uint64> &test_pos_instance_id,
                            const vector<uint64> &test_neg_instance_id);
  void assign_training_data(const map<uint64, tr1::shared_ptr<Sample> > &training_feature_vectors, const map<uint64, tr1::shared_ptr<Sample> > &test_feature_vectors);
  int cross_validate();

  void assign_this_time_test_instance_id();

  int blind_predict(const map<uint64, tr1::shared_ptr<Sample> > &all_feature_set, const vector<uint64> &instance_id, vector<double> &prediction);

//  int create_one_weak_classifier(tr1::shared_ptr<WeakClassifier> wc);
//  int learn_all_weak_classifiers(const map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
//                                 const vector<uint64> &pos_instance_id,
//                                 const vector<uint64> &neg_instance_id);
//  int boosting();

  friend ostream &operator<<(ostream &f, AttrClassifier &classifier);
  friend istream &operator>>(istream &f, AttrClassifier &classifier);

private:
  int split_fold(int folds, int iter, const vector<uint64> &whole_set, vector<uint64> &training_set, vector<uint64> &test_set);

  int m_attr_id;

  vector<uint64> m_all_pos_instance_id;
  vector<uint64> m_all_neg_instance_id;

  map<uint64, tr1::shared_ptr<Sample> > m_all_training_feature_vectors;
  vector<uint64> m_training_pos_instance_id;
  vector<uint64> m_training_neg_instance_id;

  map<uint64, tr1::shared_ptr<Sample> > m_all_test_feature_vectors;
  vector<uint64> m_test_pos_instance_id;
  vector<uint64> m_test_neg_instance_id;
  svm_problem m_problem_set;

//  map<uint64, tr1::shared_ptr<WeakClassifier> > m_all_weak_classifiers;
//  map<uint64, tr1::shared_ptr<vector<bool> > > m_all_weak_classifier_test_result;
//  vector<double> m_instance_weights;
};


#endif /* STRONG_CLASSIFIER_H_ */
