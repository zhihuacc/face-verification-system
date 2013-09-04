/*
 * weak_classifier.cpp
 *
 *  Created on: 25 Jun, 2013
 *      Author: harvey
 */

#ifndef WEAK_CLASSIFIER_CPP_
#define WEAK_CLASSIFIER_CPP_

#include "weak_classifier.h"

WeakClassifier::WeakClassifier(uint64 id):m_feature_type_id(id), m_alpha(0){
}

WeakClassifier::~WeakClassifier()
{
}
//
//double WeakClassifier::calc_weighted_error(const vector<double> &weights)
//{
//  m_weighted_error = 1;
//  m_error_rate = 1;
//  const vector<bool> &test_result = m_svm->get_test_result();
//  if (test_result.size() != weights.size())
//  {
//    return -1;
//  }
//
//  m_weighted_error = 0;
//  m_error_rate = 0;
//  for (int i = 0; i < (int)test_result.size(); i++)
//  {
//    if (!test_result[i])
//    {
//      m_weighted_error += weights[i];
//    }
//  }
//
////  m_error_rate = m_svm->get_test_error_rate();
//  return m_weighted_error;
//}
//
//const vector<bool> &WeakClassifier::get_test_result()
//{
//  return m_svm->get_test_result();
//}
//
//int WeakClassifier::shrink_test_result_set()
//{
//  m_svm->shrink_test_result_set();
//  return 0;
//}
//
//int WeakClassifier::collect_training_feature_vectors(
//    const map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
//    const vector<uint> &pos_instance_id, const vector<uint> &neg_instance_id)
//{
//
//  return 0;
//}
//
//int WeakClassifier::collect_test_feature_vectors(
//    const map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
//    const vector<uint> &pos_instance_id, const vector<uint> &neg_instance_id)
//{
//
//  return 0;
//}


#endif /* WEAK_CLASSIFIER_CPP_ */
