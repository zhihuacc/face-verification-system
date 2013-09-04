/*
 * ada_boost.h
 *
 *  Created on: 25 Jun, 2013
 *      Author: harvey
 */

#ifndef ADA_BOOST_H_
#define ADA_BOOST_H_

#include <vector>
#include <map>
#include "weak_classifier.h"

using namespace std;

//class AdaBoost
//{
//public:
//  AdaBoost(int attribute);
//  int initialize_weights();
//  double min_weighted_error();
//  void calc_weighted_error();
//  int update_weights();
//  void clear_test_stat();
//  int load_all_training_samples(const string &labels_dir_name, const string &fv_set_dir);
//  int load_all_test_samples(const string &labels_dir_name, const string &fv_set_dir);
//  int learn_weak_classifiers();
//  int boosting();
//  int test();
//  double test(std::tr1::shared_ptr<DensityVector> x);
//
//  int shrink_memory();
//
//
//private:
//  vector<unsigned long long> m_positive_training_image_id;
//  vector<unsigned long long> m_negative_training_image_id;
//
//  map<unsigned long long, std::tr1::shared_ptr<WeakClassifier> > m_all_weak_classifiers;
//  vector<double> m_boost_weights;
//  vector<std::tr1::shared_ptr<WeakClassifier> > m_picked_classifiers;
//
//  double m_min_weighted_error;
//  map<unsigned long long, std::tr1::shared_ptr<WeakClassifier> >::iterator    m_min_error_classifier_it;
//
//  unsigned long long m_attribute;
//  map<unsigned long long, tr1::shared_ptr<Sample> > m_all_training_feature_vectors;
//  vector<unsigned long long> m_positive_test_image_id;
//  vector<unsigned long long> m_negative_test_image_id;
//  map<unsigned long long, tr1::shared_ptr<Sample> > m_all_test_feature_vectors;
//
//
//};

// AdaBoost 2: public AdaBoostInterface

class AdaBoostInterface
{
public:
  virtual ~AdaBoostInterface();
  virtual int create_one_weak_classifier(tr1::shared_ptr<WeakClassifier> wc) = 0;
  virtual int learn_all_weak_classifiers(map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
                                 const vector<uint64> &pos_instance_id,
                                 const vector<uint64> &neg_instance_id) = 0;
  virtual int boosting() = 0;
  virtual int calc_all_weighted_error() = 0;
  virtual int update_weights() = 0;
};

class AdaBoost2 : public AdaBoostInterface
{
public:
  AdaBoost2();
  ~AdaBoost2(){}
  int create_one_weak_classifier(tr1::shared_ptr<WeakClassifier> wc);
  int learn_all_weak_classifiers(map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
                                 const vector<uint64> &pos_instance_id,
                                 const vector<uint64> &neg_instance_id);
  int boosting();
  int calc_all_weighted_error();
  int update_weights();
  int reset();

  friend ostream &operator<<(ostream &f, AdaBoost2 &strong_calssifier);
  friend istream &operator>>(istream &f, AdaBoost2 &strong_classifier);
protected:
  map<uint64, tr1::shared_ptr<WeakClassifier> > m_all_weak_classifiers;
  vector<double> m_instance_weights;

  double m_min_weighted_error;
  map<uint64, std::tr1::shared_ptr<WeakClassifier> >::iterator    m_min_error_classifier_it;
  vector<std::tr1::shared_ptr<WeakClassifier> > m_picked_classifiers;
};


#endif /* ADA_BOOST_H_ */
