/*
 * adaboost.cpp
 *
 *  Created on: 25 Jun, 2013
 *      Author: harvey
 */

#include "adaboost.h"
#include "feature_extraction_public.h"
#include "flags.h"
#include <cmath>
#include <sstream>
#include <iostream>

#include "sgd_svm.h"
#include "qp_svm.h"

//AdaBoost::AdaBoost(int attr):m_min_weighted_error(1), m_attribute(attr)
//{
//}
//
//double AdaBoost::min_weighted_error()
//{
//  return m_min_weighted_error;
//}
//
//void AdaBoost::calc_weighted_error()
//{
//  m_min_weighted_error = 1;
//  m_min_error_classifier_it = m_all_weak_classifiers.end();
//  for (map<unsigned long long, tr1::shared_ptr<WeakClassifier> >::iterator it = m_all_weak_classifiers.begin();
//      it != m_all_weak_classifiers.end(); it++)
//  {
//    double this_weighted_error = it->second->calc_weighted_error(m_boost_weights);
//    if (this_weighted_error < m_min_weighted_error)
//    {
//      m_min_weighted_error = this_weighted_error;
//      m_min_error_classifier_it = it;
//    }
//  }
//}
//
//int AdaBoost::update_weights()
//{
//  if (m_min_weighted_error >= 0.5)
//  {
//    return -1;
//  }
//  if (m_min_error_classifier_it == m_all_weak_classifiers.end())
//  {
//    return -2;
//  }
//
//  std::tr1::shared_ptr<WeakClassifier> min_error_classifier = m_min_error_classifier_it->second;
//  const vector<bool> &result = min_error_classifier->get_test_result();
//  if (result.size() != m_boost_weights.size())
//  {
//    return -3;
//  }
//
//  double sum = 0;
//  double beta = m_min_weighted_error / (1 - m_min_weighted_error);
//
//  for (int i = 0; i < (int)m_boost_weights.size(); i++)
//  {
//    if (result[i])
//    {
//      m_boost_weights[i] *= beta;
//    }
//
//    sum += m_boost_weights[i];
//  }
//
//  for (int i = 0; i < (int)m_boost_weights.size(); i++)
//  {
//    m_boost_weights[i] /= sum;
//  }
//
//  // add this weak classifier to strong classifier.
//  min_error_classifier->m_alpha = log(1.0 / beta);
//  m_picked_classifiers.push_back(min_error_classifier);
//  m_all_weak_classifiers.erase(m_min_error_classifier_it);
//  m_min_error_classifier_it = m_all_weak_classifiers.end();
//
//  return 0;
//}
//
//
////int AdaBoost::add_weak_classifier(unsigned long long id, std::tr1::shared_ptr<SgdSvm> feature)
////{
////  std::tr1::shared_ptr<WeakClassifier> wc(new WeakClassifier(id, feature));
////  m_all_weak_classifiers.insert(pair<unsigned long long, std::tr1::shared_ptr<WeakClassifier> >(id, wc));
////  return 0;
////}
//
////int AdaBoost::initialize(int attribute)
////{
////  string full_name = FLAGS_labeling_dir + "/" + attribute_names[attribute] + "/positive.txt";
////  ifstream infile(full_name.c_str());
////
////
////  while (infile.good())
////  {
////    unsigned long long img_id;
////    infile >> img_id;
////
////    if (!infile.good())
////    {
////      break;
////    }
////
////    m_positive_image_id.push_back(img_id);
////  }
////
////  infile.close();
////
////  full_name = FLAGS_labeling_dir + "/" + attribute_names[attribute] + "/negative.txt";
////  infile.open(full_name.c_str());
////  while (infile.good())
////  {
////    unsigned long long img_id;
////    infile >> img_id;
////
////    if (!infile.good())
////    {
////      break;
////    }
////
////    m_negative_image_id.push_back(img_id);
////  }
////
////  infile.close();
////
////  int pos = m_positive_image_id.size();
////  int neg = m_negative_image_id.size();
////  m_boost_weights.reserve(pos + neg);
////
////  for (int i = 0; i < pos; i++)
////  {
////    m_boost_weights[i] = 1.0 / pos;
////  }
////
////  for (int i = pos; i < pos + neg; i++)
////  {
////    m_boost_weights[i] = 1.0 / neg;
////  }
////
////  m_min_weighted_error = 100;
////
////  return 0;
////}
//
////const vector<unsigned long long> &AdaBoost::get_positive_image_id()
////{
////  return m_positive_image_id;
////}
////const vector<unsigned long long> &AdaBoost::get_negative_image_id()
////{
////  return m_negative_image_id;
////}
//
////const map<unsigned long long, tr1::shared_ptr<Sample> > &AdaBoost::get_all_feature_vectors()
////{
////  return m_all_training_feature_vectors;
////}
//
//
//int AdaBoost2::load_all_training_samples(const string &lables_dir_name, const string &fv_set_dir)
//{
//  string full_name = lables_dir_name + "/positive.txt";
//  ifstream infile(full_name.c_str());
//
//
//  while (infile.good())
//  {
//    unsigned long long img_id;
//    infile >> img_id;
//
//    if (!infile.good())
//    {
//      break;
//    }
//
//    m_positive_training_image_id.push_back(img_id);
//  }
//
//  infile.close();
//
//  full_name = lables_dir_name + "/negative.txt";
//  infile.open(full_name.c_str());
//  while (infile.good())
//  {
//    unsigned long long img_id;
//    infile >> img_id;
//
//    if (!infile.good())
//    {
//      break;
//    }
//
//    m_negative_training_image_id.push_back(img_id);
//  }
//
//  infile.close();
//
//
//  for (vector<unsigned long long>::iterator it = m_positive_training_image_id.begin();
//      it != m_positive_training_image_id.end();)
//  {
//    stringstream ss;
//    ss << FLAGS_feature_set_dir;
//    ss << "/";
//    ss.width(5);
//    ss.fill('0');
//    ss << *it << ".txt";
//
//    ifstream fv_file(ss.str().c_str());
//    if (!fv_file.good())
//    {
//      cout << "File " << ss.str() << " not good. Next one" << endl;
//      it = m_positive_training_image_id.erase(it);
//      continue;
//    }
//
//    while (fv_file.good())
//    {
//      tr1::shared_ptr<Sample> fv(new Sample);
//
//      fv_file >> *fv;
//      if (!fv_file.good())
//      {
//        break;
//      }
//
//      m_all_training_feature_vectors.insert(pair<unsigned long long, tr1::shared_ptr<Sample> >(fv->label(), fv));
//
//    }
//    it++;
//  }
//
//  for (vector<unsigned long long>::iterator it = m_negative_training_image_id.begin();
//      it != m_negative_training_image_id.end();)
//  {
//    stringstream ss;
//    ss << FLAGS_feature_set_dir;
//    ss << "/";
//    ss.width(5);
//    ss.fill('0');
//    ss << *it << ".txt";
//
//    ifstream fv_file(ss.str().c_str());
//    if (!fv_file.good())
//    {
//      cout << "File " << ss.str() << " not good. Next one" << endl;
//      it = m_negative_training_image_id.erase(it);
//      continue;
//    }
//
//    while (fv_file.good())
//    {
//      tr1::shared_ptr<Sample> fv(new Sample);
//
//      fv_file >> *fv;
//      if (!fv_file.good())
//      {
//        break;
//      }
//
//      m_all_training_feature_vectors.insert(pair<unsigned long long, tr1::shared_ptr<Sample> >(fv->label(), fv));
//
//    }
//    it++;
//  }
//
//  return 0;
//}
//
//int AdaBoost::initialize_weights()
//{
//    int pos = m_positive_training_image_id.size();
//    int neg = m_negative_training_image_id.size();
//    m_boost_weights.clear();
//
//    for (int i = 0; i < pos; i++)
//    {
//      m_boost_weights.push_back(1.0 / pos);
//    }
//
//    for (int i = pos; i < pos + neg; i++)
//    {
//      m_boost_weights.push_back(1.0 / neg);
//    }
//
//    m_min_weighted_error = 1;
//    m_min_error_classifier_it = m_all_weak_classifiers.end();
//
//    return 0;
//}
//
//int AdaBoost::load_all_test_samples(const string &lables_dir_name, const string &fv_set_dir)
//{
//  string full_name = lables_dir_name + "/positive.txt";
//  ifstream infile(full_name.c_str());
//
//
//  while (infile.good())
//  {
//    unsigned long long img_id;
//    infile >> img_id;
//
//    if (!infile.good())
//    {
//      break;
//    }
//
//    m_positive_test_image_id.push_back(img_id);
//  }
//
//  infile.close();
//
//  full_name = lables_dir_name + "/negative.txt";
//  infile.open(full_name.c_str());
//  while (infile.good())
//  {
//    unsigned long long img_id;
//    infile >> img_id;
//
//    if (!infile.good())
//    {
//      break;
//    }
//
//    m_negative_test_image_id.push_back(img_id);
//  }
//
//  infile.close();
//
//  for (vector<unsigned long long>::iterator it = m_positive_test_image_id.begin();
//      it != m_positive_test_image_id.end();)
//  {
//    stringstream ss;
//    ss << fv_set_dir;
//    ss << "/";
//    ss.width(5);
//    ss.fill('0');
//    ss << *it << ".txt";
//
//    ifstream fv_file(ss.str().c_str());
//    if (!fv_file.good())
//    {
//      cout << "File " << ss.str() << " not good. Next one" << endl;
//      it = m_positive_test_image_id.erase(it);
//      continue;
//    }
//
//    while (fv_file.good())
//    {
//      tr1::shared_ptr<Sample> fv(new Sample);
//
//      fv_file >> *fv;
//      if (!fv_file.good())
//      {
//        break;
//      }
//
//      m_all_test_feature_vectors.insert(pair<unsigned long long, tr1::shared_ptr<Sample> >(fv->label(), fv));
//    }
//
//    it++;
//  }
//
//  for (vector<unsigned long long>::iterator it = m_negative_test_image_id.begin();
//      it != m_negative_test_image_id.end();)
//  {
//    stringstream ss;
//    ss << fv_set_dir;
//    ss << "/";
//    ss.width(5);
//    ss.fill('0');
//    ss << *it << ".txt";
//
//    ifstream fv_file(ss.str().c_str());
//    if (!fv_file.good())
//    {
//      cout << "File " << ss.str() << " not good. Next one" << endl;
//      it = m_negative_test_image_id.erase(it);
//      continue;
//    }
//
//    while (fv_file.good())
//    {
//      tr1::shared_ptr<Sample> fv(new Sample);
//
//      fv_file >> *fv;
//      if (!fv_file.good())
//      {
//        break;
//      }
//
//      m_all_test_feature_vectors.insert(pair<unsigned long long, tr1::shared_ptr<Sample> >(fv->label(), fv));
//    }
//
//    it++;
//  }
//
//  return 0;
//}
//
//int AdaBoost::learn_weak_classifiers()
//{
//   for (int i = 0; i < NOR_TYPE_NUM; i++)
//   {
//     for (int j = 0; j < PIXEL_TYPE_NUM; j++)
//     {
//       for (int k = 0; k < REGION_NUM; k++)
//       {
//         for (int l = 0; l < AGG_TYPE_NUM; l++)
//         {
//           unsigned long long feature_type = map_feature_type_id(i, j, k, l);
//           std::tr1::shared_ptr<SgdSvm> svm(new SgdSvm);
//
//
//           svm->collect_training_feature_vectors(m_all_training_feature_vectors, m_positive_training_image_id,
//                                                 m_negative_training_image_id, feature_type);
//           svm->collect_test_feature_vectors(m_all_training_feature_vectors, m_positive_training_image_id,
//                                             m_negative_training_image_id, feature_type);
//
//           svm->shuffle();
//           /* Set parameters for svm */
//           svm->set_T(10 * (svm->get_positive_sample_num() + svm->get_negative_sample_num()));
//           svm->set_lambda(0.0001);
//           svm->set_batch_size(1);
//           svm->set_num_iter_to_average(1);
//
//           svm->Learn();
//           svm->Test();
//           svm->detach_training_samples();
//
//           std::tr1::shared_ptr<WeakClassifier> wc(new WeakClassifier(feature_type, svm));
//           m_all_weak_classifiers.insert(pair<unsigned long long, tr1::shared_ptr<WeakClassifier> >(feature_type, wc));
//
//         }
//       }
//     }
//   }
//  return 0;
//}
//
//int AdaBoost::boosting()
//{
//  int T = NOR_TYPE_NUM * PIXEL_TYPE_NUM * REGION_NUM * AGG_TYPE_NUM;
//
//  for (int t = 0; t < T; t++)
//  {
//      calc_weighted_error();
//
//      if (min_weighted_error() >= 0.5 || m_picked_classifiers.size() >= 10)
//      {
//        // abort
//        cout << "Attr: " << attribute_names[m_attribute] << " Abort boosting at iteration " << t << endl;
//        break;
//      }
//
//      update_weights();
//
//  }
//
//  cout << "Strong classifier:" << endl;
//  for (vector<tr1::shared_ptr<WeakClassifier> >::iterator it = m_picked_classifiers.begin();
//      it != m_picked_classifiers.end();
//      it++)
//  {
//    cout << (*it)->m_alpha << ": " << map_feature_type_name((*it)->m_feature_type_id) << ":" << (*it)->m_weighted_error << endl;
//    (*it)->shrink_test_result_set();
//  }
//
//  return 0;
//}
//
//int AdaBoost::test()
//{
//  int misclassification = 0;
//
//  for (vector<unsigned long long>::iterator test_img_it = m_positive_test_image_id.begin();
//      test_img_it != m_positive_test_image_id.end();
//      test_img_it++)
//  {
//    double result = 0;
//    for (vector<tr1::shared_ptr<WeakClassifier> >::iterator wc_it = m_picked_classifiers.begin();
//        wc_it != m_picked_classifiers.end();
//        wc_it++)
//    {
//      tr1::shared_ptr<WeakClassifier> wc = *wc_it;
//      unsigned long long fv_id = *test_img_it << 16 | wc->m_feature_type_id;
//      map<unsigned long long, tr1::shared_ptr<Sample> >::iterator sample_it = m_all_test_feature_vectors.find(fv_id);
//
//      if (sample_it == m_all_test_feature_vectors.end())
//      {
//        continue;
//      }
//
//      result += (wc->m_alpha * wc->m_svm->test(sample_it->second));
//    }
//
//    if (result < 0.0)
//    {
//      misclassification++;
//    }
//  }
//
//  for (vector<unsigned long long>::iterator test_img_it = m_negative_test_image_id.begin();
//      test_img_it != m_negative_test_image_id.end();
//      test_img_it++)
//  {
//    double result = 0;
//    for (vector<tr1::shared_ptr<WeakClassifier> >::iterator wc_it = m_picked_classifiers.begin();
//        wc_it != m_picked_classifiers.end();
//        wc_it++)
//    {
//      tr1::shared_ptr<WeakClassifier> wc = *wc_it;
//      unsigned long long fv_id = *test_img_it << 16 | wc->m_feature_type_id;
//      map<unsigned long long, tr1::shared_ptr<Sample> >::iterator sample_it = m_all_test_feature_vectors.find(fv_id);
//
//      if (sample_it == m_all_test_feature_vectors.end())
//      {
//        continue;
//      }
//
//      result += (wc->m_alpha * wc->m_svm->test(sample_it->second));
//    }
//
//    if (result >= 0.0)
//    {
//      misclassification++;
//    }
//  }
//
//  cout << "******************** Test for Attribute Classifier ****************" << endl
//      << attribute_names[m_attribute] << " error rate: "
//      << (double)misclassification / (m_positive_test_image_id.size() + m_negative_test_image_id.size()) << endl
//      << "test set size: " << (m_positive_test_image_id.size() + m_negative_test_image_id.size()) << endl;
//  return 0;
//}
//
//
//
//int AdaBoost::shrink_memory()
//{
//  m_all_training_feature_vectors.clear();
//  m_all_test_feature_vectors.clear();
//  m_all_weak_classifiers.clear();
//  m_positive_training_image_id.clear();
//  m_negative_training_image_id.clear();
//  m_positive_test_image_id.clear();
//  m_negative_test_image_id.clear();
//
//  return 0;
//}

/*******************************/

AdaBoostInterface::~AdaBoostInterface()
{
}

AdaBoost2::AdaBoost2()
{
  m_min_weighted_error = 1.0;
  m_min_error_classifier_it = m_all_weak_classifiers.end();
}

int AdaBoost2::create_one_weak_classifier(tr1::shared_ptr<WeakClassifier> wc)
{
//  tr1::shared_ptr<WeakClassifier> wc(new WeakClassifier(feature_type));

  uint64 feature_type;
  wc->feature_type(feature_type);
  m_all_weak_classifiers.insert(pair<uint64, tr1::shared_ptr<WeakClassifier> >(feature_type, wc));
  return 0;
}
int AdaBoost2::learn_all_weak_classifiers(map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
                               const vector<uint64> &pos_instance_id,
                               const vector<uint64> &neg_instance_id)
{

  bool initialized = false;

  for (map<uint64, tr1::shared_ptr<WeakClassifier> >::const_iterator wc_it = m_all_weak_classifiers.begin();
      wc_it != m_all_weak_classifiers.end();
      wc_it++)
  {

    vector<tr1::shared_ptr<Sample> > training_samples;
    uint64 feature_type = wc_it->first & 0xFFFFUL;
//    double lambda = 1.0 / pow(10, (wc_it->first >> 16) + 1);
    tr1::shared_ptr<WeakClassifier> wc = wc_it->second;

//    cout << "Training weak classifier: " << map_feature_type_name(feature_type) << "lambda-" << lambda << endl;
    cout << "Training weak classifier: " << map_feature_type_name(feature_type) << endl;

    int pos_num = 0;
    map<uint64, tr1::shared_ptr<Sample> >::iterator fv_it = all_feature_vectors.end();
    for (vector<uint64>::const_iterator inst_it = pos_instance_id.begin();
        inst_it != pos_instance_id.end(); inst_it++)
    {
      uint64 fv_id = *inst_it << 16 | feature_type;

      fv_it = all_feature_vectors.find(fv_id);
      if (fv_it == all_feature_vectors.end())
      {
        cerr << "Cannot find feature vector for this image: " << *inst_it << endl;
        continue;
      }
      fv_it->second->set_label(+1);
      training_samples.push_back(fv_it->second);
      all_feature_vectors.erase(fv_it);
      pos_num++;
    }

    int neg_num = 0;
    fv_it = all_feature_vectors.end();
    for (vector<uint64>::const_iterator inst_it = neg_instance_id.begin();
        inst_it != neg_instance_id.end(); inst_it++)
    {
      uint64 fv_id = *inst_it << 16 | feature_type;
      fv_it = all_feature_vectors.find(fv_id);
      if (fv_it == all_feature_vectors.end())
      {
        cerr << "Cannot find feature vector for this image: " << *inst_it << endl;
        continue;
      }
      fv_it->second->set_label(-1);
      training_samples.push_back(fv_it->second);
      all_feature_vectors.erase(fv_it);
      neg_num++;
    }

    if (pos_num <= 0 || neg_num <= 0)
    {
      return -1;
    }

    if (!initialized)
    {
      m_instance_weights.reserve(pos_num + neg_num);
      m_instance_weights.resize(pos_num + neg_num);
      for (int i = 0; i < pos_num; i++)
      {
        m_instance_weights[i] = 1.0 / pos_num;
      }

      for (int i = 0; i < neg_num; i++)
      {
        m_instance_weights[pos_num + i] = 1.0 / neg_num;
      }
      initialized = true;
    }

//    SgdSvm *svm = dynamic_cast<SgdSvm*>(wc.get());
//    if (svm == NULL)
//    {
//      continue;
//    }
//
//    svm->set_T(30 * training_samples.size());
//    svm->set_lambda(lambda);
//    svm->set_batch_size(1);
//    svm->set_num_iter_to_average(30);
    QpSvm *svm = dynamic_cast<QpSvm*>(wc.get());
    if (svm == NULL)
    {
      continue;
    }
    svm->m_param.svm_type = C_SVC;
    svm->m_param.kernel_type = RBF;
    svm->m_param.degree = 3;
    svm->m_param.gamma = 0;        // 1/num_features
    svm->m_param.coef0 = 0;
    svm->m_param.nu = 0.5;
    svm->m_param.cache_size = 100;
    svm->m_param.C = 1;
    svm->m_param.eps = 1e-3;
    svm->m_param.p = 0.1;
    svm->m_param.shrinking = 1;
    svm->m_param.probability = 0;
    svm->m_param.nr_weight = 0;
    svm->m_param.weight_label = NULL;
    svm->m_param.weight = NULL;

    cout << endl;
    wc->learn(training_samples);
    cout << endl;

    vector<bool> test_result;
    double total_err_rate, false_pos_rate, false_neg_rate, true_pos_rate,  true_neg_rate;
    wc->test(training_samples, test_result, total_err_rate, false_pos_rate, false_neg_rate, true_pos_rate, true_neg_rate);
    cout << endl;

   // m_all_weak_classifier_test_result.insert(pair<uint64, tr1::shared_ptr<vector<bool> > >(feature_type, test_result));
  }

  return 0;
}

int AdaBoost2::calc_all_weighted_error()
{
  m_min_weighted_error = 1;
  m_min_error_classifier_it = m_all_weak_classifiers.end();
  for (map<uint64, tr1::shared_ptr<WeakClassifier> >::iterator it = m_all_weak_classifiers.begin();
      it != m_all_weak_classifiers.end(); it++)
  {
    double this_weighted_error = 1.0;
    int ret = it->second->calc_weighted_error(m_instance_weights, this_weighted_error);
    if (ret != 0)
    {
      continue;
    }

    if (this_weighted_error < m_min_weighted_error)
    {
      m_min_weighted_error = this_weighted_error;
      m_min_error_classifier_it = it;
    }
  }

  return 0;
}
#include <cstdlib>
int AdaBoost2::update_weights()
{
  int ret = 0;
  if (m_min_weighted_error >= 0.5)
  {
    return -1;
  }

  if (m_min_error_classifier_it == m_all_weak_classifiers.end())
  {
    return -2;
  }

  std::tr1::shared_ptr<WeakClassifier> min_error_classifier = m_min_error_classifier_it->second;

  vector<bool> test_result;
  ret = min_error_classifier->get_last_test_result(test_result);
  if (ret != 0)
  {
    return -3;
  }

  if (test_result.size() != m_instance_weights.size())
  {
    return -4;
  }

  double sum = 0;
  double beta = m_min_weighted_error / (1 - m_min_weighted_error);

  if (abs(beta) < 0.00001)
  {
    beta = 0.00001;
  }


  for (int i = 0; i < (int)m_instance_weights.size(); i++)
  {
    if (test_result[i])
    {
      m_instance_weights[i] *= beta;
    }

    sum += m_instance_weights[i];
  }

  for (int i = 0; i < (int)m_instance_weights.size(); i++)
  {
    m_instance_weights[i] /= sum;
  }

  // add this weak classifier to strong classifier.
  min_error_classifier->set_alpha(log(1.0 / beta));
  m_picked_classifiers.push_back(min_error_classifier);
  m_all_weak_classifiers.erase(m_min_error_classifier_it);
  m_min_error_classifier_it = m_all_weak_classifiers.end();

  return 0;
}

int AdaBoost2::boosting()
{
  int T = m_all_weak_classifiers.size();

  for (int t = 0; t < T; t++)
  {
      calc_all_weighted_error();

      if (m_min_weighted_error >= 0.5 || m_picked_classifiers.size() >= 6)
      {
        if (m_picked_classifiers.size() <= 0)
        {
          cerr << "Error in boosting, No single weak classifier.." << endl;
        }
        // abort
        //cout << "Attr: " << attribute_names[m_attribute] << " Abort boosting at iteration " << t << endl;
        break;
      }

      update_weights();

  }

  cout << "Strong classifier:" << endl;
  for (vector<tr1::shared_ptr<WeakClassifier> >::iterator it = m_picked_classifiers.begin();
      it != m_picked_classifiers.end();
      it++)
  {
    double alpha, error, lambda;
    uint64 feature_type;
    (*it)->alpha(alpha);
    (*it)->weighted_error(error);
    (*it)->feature_type(feature_type);
//    lambda = 1.0 / pow(10, (feature_type >> 16) + 1);
    feature_type &= 0xFFFFUL;

//    cout << alpha << ": " << map_feature_type_name(feature_type) << " lambda-" << lambda << " :" << error << endl;
    cout << alpha << ": " << map_feature_type_name(feature_type) <<  " :" << error << endl;

  }

  return 0;
}

int AdaBoost2::reset()
{
  for (vector<tr1::shared_ptr<WeakClassifier> >::iterator it = m_picked_classifiers.begin();
       it != m_picked_classifiers.end(); )
  {
    uint64 feature_type;
    (*it)->feature_type(feature_type);
    m_all_weak_classifiers.insert(pair<uint64, tr1::shared_ptr<WeakClassifier> >(feature_type, (*it)));
    it = m_picked_classifiers.erase(it);
  }

  for (map<uint64, tr1::shared_ptr<WeakClassifier> >::iterator it = m_all_weak_classifiers.begin();
       it != m_all_weak_classifiers.end(); it++)
  {
//    ((SgdSvm*)(it->second.get()))->reset();
    ((QpSvm*)(it->second.get()))->reset();
  }
  return 0;
}

ostream &operator<<(ostream &f, AdaBoost2 &strong_classifier)
{
  f << strong_classifier.m_picked_classifiers.size() << endl;

  for (vector<tr1::shared_ptr<WeakClassifier> >::iterator it = strong_classifier.m_picked_classifiers.begin();
       it != strong_classifier.m_picked_classifiers.end();
       it++)
  {
    char buf[128];
    uint64 feature_type;
    double d;
    (*it)->feature_type(feature_type);
    sprintf(buf, "%lu", feature_type);
    f << string(buf) << "\t";

    (*it)->alpha(d);
    sprintf(buf, "%lf", d);
    f << string(buf) << endl;
    f << *(SgdSvm*)((*it).get());

    if ((it + 1) != strong_classifier.m_picked_classifiers.end())
    {
      f << endl;
    }
  }

  return f;
}

istream &operator>>(istream &f, AdaBoost2 &strong_classifier)
{
  int num = 0;

  f >> num;

  strong_classifier.m_picked_classifiers.clear();

  for (int i = 0; i < num; i++)
  {

    SgdSvm *svm = new SgdSvm;

    string str;
    uint64 feature_type;
    double d = 0.0;

    f >> str;
    sscanf(str.c_str(), "%lu", &feature_type);
    svm->set_feature_type(feature_type);

    f >> str;
    sscanf(str.c_str(), "%lf", &d);
    svm->set_alpha(d);


    f >> *svm;

    if (!f.good())
    {
      cerr << "Error in loading Strong Classifier: " << __FILE__ << ", " << __LINE__ << endl;
      break;
    }

    strong_classifier.m_picked_classifiers.push_back(tr1::shared_ptr<WeakClassifier>(svm));
  }

  if ((int)strong_classifier.m_picked_classifiers.size() != num)
  {
    cerr << "Error in loading Strong Classifier: " << __FILE__ << ", " << __LINE__ << endl;
  }

  return f;
}
