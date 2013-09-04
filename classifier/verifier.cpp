/*
 * simile_verifier.cpp
 *
 *  Created on: 6 Aug, 2013
 *      Author: harvey
 */

#include <fstream>
#include "verifier.h"
#include "feature_extraction_public.h"
#include "sample.h"
#include "simile_classifier.h"
#include "flags.h"

#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <algorithm>


vector<Fold> all_folds;
set<uint64> all_benchmark_instance_id;

int Verifier::load_verification_benchmark_feature_vectors(const string &fname)
{
  //string fv_fname = FLAGS_verifier_feature_set_dir + "/simile-verifier.txt";
  ifstream fv_file(fname.c_str());

  while (!fv_file.eof())
  {
    tr1::shared_ptr<Sample> fv(new Sample);

    fv_file >> *fv;
    if (!fv_file.good())
    {
      break;
    }

    m_all_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->feature_vector_id(), fv));

  }

  m_pos_break_points.push_back(0);
  m_neg_break_points.push_back(0);

  for (int iter = 0; iter < (int)all_folds.size(); iter++)
  {

    Fold &this_fold = all_folds[iter];

    for (int i = 0; i < (int)this_fold.pos_pairs.size(); i++)
    {
      uint64 id = this_fold.pos_pairs[i].first << 32 | this_fold.pos_pairs[i].second;

      map<uint64, tr1::shared_ptr<Sample> >::iterator it = m_all_feature_vectors.find(id);
      if (it == m_all_feature_vectors.end())
      {
        continue;
      }

      m_pos_instance_id.push_back(id);
    }

    m_pos_break_points.push_back(m_pos_instance_id.size());

    for (int i = 0; i < (int)this_fold.neg_pairs.size(); i++)
    {
      uint64 id = this_fold.neg_pairs[i].first << 32 | this_fold.neg_pairs[i].second;

      map<uint64, tr1::shared_ptr<Sample> >::iterator it = m_all_feature_vectors.find(id);
      if (it == m_all_feature_vectors.end())
      {
        continue;
      }

      m_neg_instance_id.push_back(id);
    }

    m_neg_break_points.push_back(m_neg_instance_id.size());
  }

  cout << "Verification Bechmark Dataset: " << endl;
  cout << "   Pos: ";
  for (int i = 0; i < (int)m_pos_break_points.size(); i++)
  {
    cout << m_pos_break_points[i] << " ";
  }
  cout << endl;

  cout << "   Neg: ";
  for (int i = 0; i < (int)m_neg_break_points.size(); i++)
  {
    cout << m_neg_break_points[i] << " ";
  }
  cout << endl;

  return 0;
}

int Verifier::produce_cross_validation_files()
{
  for (int iter = 0; iter < (int)all_folds.size(); iter++)
    {
     stringstream ss1;
     stringstream ss2;

     ss1 << "/home/harvey/Dataset/cross-validation/cross-train-";
     ss2 << "/home/harvey/Dataset/cross-validation/cross-test-";

     ss1 << iter << ".txt";
     ss2 << iter << ".txt";

     string fname = ss1.str().c_str();
     ofstream cross_train(fname.c_str());

     fname = ss2.str().c_str();
     ofstream cross_test(fname.c_str());

      vector<uint64> training_pos_instance_id, training_neg_instance_id, test_pos_instance_id, test_neg_instance_id;

      training_pos_instance_id = m_pos_instance_id;

      vector<uint64>::iterator start = training_pos_instance_id.begin() + m_pos_break_points[iter];
      vector<uint64>::iterator stop = training_pos_instance_id.begin() + m_pos_break_points[iter + 1];
      test_pos_instance_id.assign(start, stop);
      training_pos_instance_id.erase(start, stop);

      training_neg_instance_id = m_neg_instance_id;
      start = training_neg_instance_id.begin() + m_neg_break_points[iter];
      stop = training_neg_instance_id.begin() + m_neg_break_points[iter + 1];
      test_neg_instance_id.assign(start, stop);
      training_neg_instance_id.erase(start, stop);

      vector<tr1::shared_ptr<Sample> > training_samples, test_samples;
      for (vector<uint64>::iterator it = training_pos_instance_id.begin();
           it != training_pos_instance_id.end(); it++)
      {
        map<uint64, tr1::shared_ptr<Sample> >::iterator it2 = m_all_feature_vectors.find(*it);
        if (it2 == m_all_feature_vectors.end())
        {
          continue;
        }
        it2->second->set_label(1.0);
//        training_samples.push_back(it2->second);
        cross_train << *(it2->second);
      }

      for (vector<uint64>::iterator it = training_neg_instance_id.begin();
           it != training_neg_instance_id.end(); it++)
      {
        map<uint64, tr1::shared_ptr<Sample> >::iterator it2 = m_all_feature_vectors.find(*it);
        if (it2 == m_all_feature_vectors.end())
        {
          continue;
        }
        it2->second->set_label(-1.0);
//        training_samples.push_back(it2->second);
        cross_train << *(it2->second);
      }

      for (vector<uint64>::iterator it = test_pos_instance_id.begin();
           it != test_pos_instance_id.end(); it++)
      {
        map<uint64, tr1::shared_ptr<Sample> >::iterator it2 = m_all_feature_vectors.find(*it);
        if (it2 == m_all_feature_vectors.end())
        {
          continue;
        }
        it2->second->set_label(1.0);
//        test_samples.push_back(it2->second);
        cross_test << *(it2->second);
      }

      for (vector<uint64>::iterator it = test_neg_instance_id.begin();
           it != test_neg_instance_id.end(); it++)
      {
        map<uint64, tr1::shared_ptr<Sample> >::iterator it2 = m_all_feature_vectors.find(*it);
        if (it2 == m_all_feature_vectors.end())
        {
          continue;
        }
        it2->second->set_label(-1.0);
//        test_samples.push_back(it2->second);
        cross_test << *(it2->second);
      }

    }

  return 0;
}

int Verifier::cross_validate()
{
  double average_err_rate = 0, avg_false_pos_rate = 0, avg_false_neg_rate = 0, avg_true_pos_rate = 0, avg_true_neg_rate = 0;

  for (int iter = 0; iter < (int)all_folds.size(); iter++)
  {
    vector<uint64> training_pos_instance_id, training_neg_instance_id, test_pos_instance_id, test_neg_instance_id;

    training_pos_instance_id = m_pos_instance_id;

    vector<uint64>::iterator start = training_pos_instance_id.begin() + m_pos_break_points[iter];
    vector<uint64>::iterator stop = training_pos_instance_id.begin() + m_pos_break_points[iter + 1];
    test_pos_instance_id.assign(start, stop);
    training_pos_instance_id.erase(start, stop);

    training_neg_instance_id = m_neg_instance_id;
    start = training_neg_instance_id.begin() + m_neg_break_points[iter];
    stop = training_neg_instance_id.begin() + m_neg_break_points[iter + 1];
    test_neg_instance_id.assign(start, stop);
    training_neg_instance_id.erase(start, stop);

    vector<tr1::shared_ptr<Sample> > training_samples, test_samples;
    for (vector<uint64>::iterator it = training_pos_instance_id.begin();
         it != training_pos_instance_id.end(); it++)
    {
      map<uint64, tr1::shared_ptr<Sample> >::iterator it2 = m_all_feature_vectors.find(*it);
      if (it2 == m_all_feature_vectors.end())
      {
        continue;
      }
      it2->second->set_label(1.0);
      training_samples.push_back(it2->second);
    }

    for (vector<uint64>::iterator it = training_neg_instance_id.begin();
         it != training_neg_instance_id.end(); it++)
    {
      map<uint64, tr1::shared_ptr<Sample> >::iterator it2 = m_all_feature_vectors.find(*it);
      if (it2 == m_all_feature_vectors.end())
      {
        continue;
      }
      it2->second->set_label(-1.0);
      training_samples.push_back(it2->second);
    }

    for (vector<uint64>::iterator it = test_pos_instance_id.begin();
         it != test_pos_instance_id.end(); it++)
    {
      map<uint64, tr1::shared_ptr<Sample> >::iterator it2 = m_all_feature_vectors.find(*it);
      if (it2 == m_all_feature_vectors.end())
      {
        continue;
      }
      it2->second->set_label(1.0);
      test_samples.push_back(it2->second);
    }

    for (vector<uint64>::iterator it = test_neg_instance_id.begin();
         it != test_neg_instance_id.end(); it++)
    {
      map<uint64, tr1::shared_ptr<Sample> >::iterator it2 = m_all_feature_vectors.find(*it);
      if (it2 == m_all_feature_vectors.end())
      {
        continue;
      }
      it2->second->set_label(-1.0);
      test_samples.push_back(it2->second);
    }

    random_shuffle(training_samples.begin(), training_samples.end());
    m_svm.reset();
//    m_svm.set_T(30 * training_samples.size());
//    m_svm.set_batch_size(1);
//    m_svm.set_lambda(0.1);
//    m_svm.set_num_iter_to_average(30);

      m_svm.m_param.svm_type = C_SVC;
      m_svm.m_param.kernel_type = RBF;
      m_svm.m_param.degree = 3;
      m_svm.m_param.gamma = 0;        // 1/num_features
      m_svm.m_param.coef0 = 0;
      m_svm.m_param.nu = 0.5;
      m_svm.m_param.cache_size = 100;
      m_svm.m_param.C = 1;
      m_svm.m_param.eps = 1e-3;
      m_svm.m_param.p = 0.1;
      m_svm.m_param.shrinking = 1;
      m_svm.m_param.probability = 0;
      m_svm.m_param.nr_weight = 0;
      m_svm.m_param.weight_label = NULL;
      m_svm.m_param.weight = NULL;

      m_svm.learn(training_samples);

    vector<bool> test_result;
    double total_err_rate = 0, false_pos_rate = 0, false_neg_rate = 0, true_pos_rate = 0, true_neg_rate;
    m_svm.test(test_samples, test_result, total_err_rate, false_pos_rate, false_neg_rate, true_pos_rate, true_neg_rate);

    average_err_rate += total_err_rate;
    avg_false_pos_rate += false_pos_rate;
    avg_false_neg_rate += false_neg_rate;
    avg_true_pos_rate += true_pos_rate;
    avg_true_neg_rate += true_neg_rate;
  }

  average_err_rate /= all_folds.size();
  avg_false_pos_rate /= all_folds.size();
  avg_false_neg_rate /= all_folds.size();
  avg_true_pos_rate /= all_folds.size();
  avg_true_neg_rate /= all_folds.size();

  cout << "Cross Validation: " << all_folds.size() << " folds." << endl;
  cout << "  Average Error Rate: " << average_err_rate << endl;
  cout << "  Average False Positive Error Rate: " << avg_false_pos_rate << endl;
  cout << "  Average False Negative Error Rate: " << avg_false_neg_rate << endl;
  cout << "  Average True Positive Error Rate: " << avg_true_pos_rate << endl;
  cout << "  Average True Negative Error Rate: " << avg_true_neg_rate << endl;

  return 0;
}

int Verifier::detach_training_data()
{
  m_all_feature_vectors.clear();
  map<uint64, tr1::shared_ptr<Sample> >().swap(m_all_feature_vectors);

  vector<uint64>().swap(m_pos_instance_id);
  vector<uint64>().swap(m_neg_instance_id);
  vector<int>().swap(m_pos_break_points);
  vector<int>().swap(m_neg_break_points);


  return 0;
}

int Verifier::reset()
{
	m_svm.reset();
	return 0;
}


