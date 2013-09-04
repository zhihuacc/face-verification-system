/*
 * verifier.h
 *
 *  Created on: 6 Aug, 2013
 *      Author: harvey
 */

#ifndef VERIFIER_H_
#define VERIFIER_H_

#include <string>
#include <vector>
#include <map>
#include <set>
#include "types.h"
#include "classifier_interface.h"
#include "sgd_svm.h"
#include "qp_svm.h"

using namespace std;


typedef struct VerificationPair_
{
  uint64 first;
  uint64 second;
} VerificationPair;

typedef struct Fold_
{
  vector<VerificationPair> pos_pairs;
  vector<VerificationPair> neg_pairs;
} Fold;


class Verifier
{
public:
//  int learn(const vector<tr1::shared_ptr<Sample> > &training_samples);
//  int learn(tr1::shared_ptr<Sample> training_sample);
//  int test(const vector<tr1::shared_ptr<Sample> > &test_samples,
//                   vector<bool> &test_result, double &total_error_rate,
//                   double &false_pos_rate, double &false_neg_rate);
//  int test(tr1::shared_ptr<Sample> test_sample, double &prediction);

  int detach_training_data();
  int reset();

  int cross_validate();

  int load_verification_benchmark_feature_vectors(const string &fname);

  int produce_cross_validation_files();
private:
  map<uint64, tr1::shared_ptr<Sample> > m_all_feature_vectors;

//  SgdSvm m_svm;
  QpSvm m_svm;

  vector<uint64> m_pos_instance_id;
  vector<uint64> m_neg_instance_id;
  vector<int>    m_pos_break_points;
  vector<int>    m_neg_break_points;
};



#endif /* VERIFIER_H_ */
