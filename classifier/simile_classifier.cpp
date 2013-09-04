/*
 * simile_classifier.cpp
 *
 *  Created on: 21 Jul, 2013
 *      Author: harvey
 */

#include "simile_classifier.h"
#include "feature_extraction_public.h"

#include <map>
#include <fstream>
#include <sstream>
#include <cmath>

#include "sgd_svm.h"
#include "qp_svm.h"
#include "flags.h"


extern string ref_person_names[REF_PERSON_NUM];

int SimileClassifier::learn_all_weak_classifiers(map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
                               const vector<uint64> &pos_instance_id,
                               const vector<uint64> &neg_instance_id)
{
  bool initialized = false;



  for (map<uint64, tr1::shared_ptr<WeakClassifier> >::const_iterator wc_it = m_all_weak_classifiers.begin();
      wc_it != m_all_weak_classifiers.end();
      wc_it++)
  {
    vector<tr1::shared_ptr<Sample> > training_samples;
    uint64 feature_type = wc_it->first & 0xFF0FUL;
//    int lambda_type = wc_it->first >> 16;
//    double lambda = 1.0 / pow(10, (wc_it->first >> 16) + 1);
    feature_type |= (m_region_id << 4);                          // NOTE

//    cout << "Training weak classifier: " << map_feature_type_name(feature_type) << " lambda-" << lambda << endl;
    cout << "Training weak classifier: " << map_feature_type_name(feature_type) << endl;

    tr1::shared_ptr<WeakClassifier> wc = wc_it->second;

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
    double total_err_rate, false_pos_rate, false_neg_rate, true_pos_rate, true_neg_rate;
    wc->test(training_samples, test_result, total_err_rate, false_pos_rate, false_neg_rate, true_pos_rate, true_neg_rate);
    cout << endl;
  }
	return 0;
}

int SimileClassifier::learn(const vector<tr1::shared_ptr<Sample> > &training_samples)
{
  learn_all_weak_classifiers(m_all_training_feature_vectors, m_training_pos_instance_id, m_training_neg_instance_id);
  boosting();
  cout << "ref-person-"<<ref_person_names[m_ref_person_id] << ", region-" << region_name[m_region_id] << ", end" << endl << endl;
   return 0;
}

int SimileClassifier::learn(tr1::shared_ptr<Sample> training_sample)
{
  learn_all_weak_classifiers(m_all_training_feature_vectors, m_training_pos_instance_id, m_training_neg_instance_id);
  boosting();
  cout << "ref-person-"<<ref_person_names[m_ref_person_id] << ", region-" << region_name[m_region_id] << ", end" << endl << endl;
  return 0;
}
int SimileClassifier::test(const vector<tr1::shared_ptr<Sample> > &test_samples,
                 vector<bool> &test_result, double &total_error_rate,
                 double &false_pos_rate, double &false_neg_rate,
                 double &true_pos_rate, double &true_neg_rate)
{
  tr1::shared_ptr<Sample> dummy1;
  double dummy2;
  test(dummy1, dummy2);
  return 0;
}
int SimileClassifier::test(tr1::shared_ptr<Sample> test_sample, double &prediction)
{
  int misclassification = 0;

  for (vector<uint64>::iterator test_inst_it = m_test_pos_instance_id.begin();
      test_inst_it != m_test_pos_instance_id.end();
      test_inst_it++)
  {
    double pos_weight = 0.0, neg_weight = 0.0;
    for (vector<tr1::shared_ptr<WeakClassifier> >::iterator wc_it = m_picked_classifiers.begin();
        wc_it != m_picked_classifiers.end();
        wc_it++)
    {
      tr1::shared_ptr<WeakClassifier> wc = *wc_it;

      uint64 feature_type;
      wc->feature_type(feature_type);

      uint64 fv_id = *test_inst_it << 16 | m_region_id << 4| feature_type;
      map<uint64, tr1::shared_ptr<Sample> >::iterator sample_it = m_all_test_feature_vectors.find(fv_id);

      if (sample_it == m_all_test_feature_vectors.end())
      {
        continue;
      }

      double alpha, prediction;
      wc->alpha(alpha);
      wc->test(sample_it->second, prediction);
      if (prediction > 0.0)
      {
        pos_weight += alpha;
      }
      else
      {
        neg_weight += alpha;
      }
    }

    if (pos_weight < neg_weight)
    {
      misclassification++;
    }
  }

  for (vector<uint64>::iterator test_inst_it = m_test_neg_instance_id.begin();
      test_inst_it != m_test_neg_instance_id.end();
      test_inst_it++)
  {
    double pos_weight = 0.0, neg_weight = 0.0;
    for (vector<tr1::shared_ptr<WeakClassifier> >::iterator wc_it = m_picked_classifiers.begin();
        wc_it != m_picked_classifiers.end();
        wc_it++)
    {
      tr1::shared_ptr<WeakClassifier> wc = *wc_it;

      uint64 feature_type;
      wc->feature_type(feature_type);

      uint64 fv_id = *test_inst_it << 16 | m_region_id << 4 | feature_type;
      map<uint64, tr1::shared_ptr<Sample> >::iterator sample_it = m_all_test_feature_vectors.find(fv_id);

      if (sample_it == m_all_test_feature_vectors.end())
      {
        continue;
      }

      double alpha, prediction;
      wc->alpha(alpha);
      wc->test(sample_it->second, prediction);
      if (prediction > 0.0)
      {
        pos_weight += alpha;
      }
      else
      {
        neg_weight += alpha;
      }
      //result += (alpha * prediction > 0.0);
    }


    if (pos_weight > neg_weight)
    {
      misclassification++;
    }
  }

  prediction = (double)misclassification / (m_test_pos_instance_id.size() + m_test_neg_instance_id.size());

  cout << "******************** Test for Attribute Classifier ****************" << endl
      << ref_person_names[m_ref_person_id] << " - region " << region_name[m_region_id] << " error rate: "
      << (double)misclassification / (m_test_pos_instance_id.size() + m_test_neg_instance_id.size()) << endl
      << "test set size: " << (m_test_pos_instance_id.size() + m_test_neg_instance_id.size()) << endl;

  return 0;
}

int SimileClassifier::blind_predict(const map<uint64, tr1::shared_ptr<Sample> > &all_fv, const vector<uint64> &test_set, vector<double> &all_prediction)
{

  all_prediction.reserve(test_set.size());
  all_prediction.resize(test_set.size());


  for (int i = 0; i < (int)test_set.size(); i++)
  {
    double pos_weight = 0.0, neg_weight = 0.0;
    for (vector<tr1::shared_ptr<WeakClassifier> >::iterator wc_it = m_picked_classifiers.begin();
        wc_it != m_picked_classifiers.end();
        wc_it++)
    {
      tr1::shared_ptr<WeakClassifier> wc = *wc_it;

      uint64 feature_type;
      wc->feature_type(feature_type);

      uint64 fv_id = test_set[i] << 16 | m_region_id << 4| feature_type;
      map<uint64, tr1::shared_ptr<Sample> >::const_iterator sample_it = all_fv.find(fv_id);

      if (sample_it == all_fv.end())
      {
        continue;
      }

      double alpha, prediction;
      wc->alpha(alpha);
      wc->test(sample_it->second, prediction);
      if (prediction > 0.0)
      {
        pos_weight += alpha;
      }
      else
      {
        neg_weight += alpha;
      }
    }

    if (pos_weight > neg_weight)
    {
      all_prediction[i] = abs(pos_weight);
    }
    else
    {
      all_prediction[i] = -abs(neg_weight);
    }
  }

  return 0;
}

int SimileClassifier::cross_validate()
{
  double err_rate = 0.0;
  for (int iter = 0; iter < 3; iter++)
  {
    reset();

    vector<uint64> pos_training_set, pos_test_set;
    vector<uint64> neg_training_set, neg_test_set;

    split_fold(3, iter, m_all_pos_instance_id, pos_training_set, pos_test_set);
    split_fold(3, iter, m_all_neg_instance_id, neg_training_set, neg_test_set);

    assign_this_time_training_instance_id(pos_training_set, neg_training_set, pos_test_set, neg_test_set);

    learn(vector<tr1::shared_ptr<Sample> >());

    tr1::shared_ptr<Sample> dummy1;
    double this_err_rate;
    test(dummy1, this_err_rate);

    err_rate += this_err_rate;
    // test TO DO
  }

  err_rate /= 3.0;
  cout << "Ref person-"<<ref_person_names[m_ref_person_id] <<", region-" << region_name[m_region_id] << ", Cross Validation - 3 folds: " << endl;
  cout << "  Average Error Rate: " << err_rate << endl;


  return 0;
}

void SimileClassifier::set_ref_person_id_and_region_id(uint64 person_id, uint64 region_id)
{
  m_ref_person_id = person_id;
  m_region_id = region_id;
}

void SimileClassifier::assign_total_training_instance_id(const vector<uint64> &all_pos_instance_id, const vector<uint64> &all_neg_instance_id)
{
  m_all_pos_instance_id = all_pos_instance_id;
  m_all_neg_instance_id = all_neg_instance_id;
}

void SimileClassifier::assign_this_time_training_instance_id(const vector<uint64> &training_pos_instance_id,
                                            const vector<uint64> &training_neg_instance_id,
                                            const vector<uint64> &test_pos_instance_id,
                                            const vector<uint64> &test_neg_instance_id)
{
  m_training_pos_instance_id = training_pos_instance_id;
  m_training_neg_instance_id = training_neg_instance_id;

  m_test_pos_instance_id = test_pos_instance_id;
  m_test_neg_instance_id = test_neg_instance_id;
}

void SimileClassifier::assign_training_data(const map<uint64, tr1::shared_ptr<Sample> > &training_feature_vectors, const map<uint64, tr1::shared_ptr<Sample> > &test_feature_vectors)
{
  m_all_training_feature_vectors = training_feature_vectors;
  m_all_test_feature_vectors = test_feature_vectors;
}



int SimileClassifier::split_fold(int folds, int iter, const vector<uint64> &whole_set, vector<uint64> &training_set, vector<uint64> &test_set)
{
  int size = whole_set.size();
  int step = size / folds;
  int start = iter * step;
  int stop = start + step;

  training_set = whole_set;

  vector<uint64>::const_iterator it_start = training_set.begin() + start,
                                 it_stop = training_set.begin() + stop;

  vector<uint64>::iterator it_start2 = training_set.begin() + start,
                                 it_stop2 = training_set.begin() + stop;
  test_set.assign(it_start, it_stop);

  training_set.erase(it_start2, it_stop2);
  return 0;
}

int SimileClassifier::reset()
{
  AdaBoost2::reset();
  return 0;
}

//ostream &operator<<(ostream &f, SimileClassifier &classifier)
//{
//  if (!f.good())
//  {
//    return f;
//  }
//
//  f << classifier.m_ref_person_id << "\t" << classifier.m_region_id << endl;
//
//  f << static_cast<AdaBoost2&>(classifier) << endl;
//
//  return f;
//}
//
//istream &operator>>(istream &f, SimileClassifier &classifier)
//{
//  if (!f.good())
//  {
//    return f;
//  }
//
//  f >> classifier.m_ref_person_id >> classifier.m_region_id;
//  f >> static_cast<AdaBoost2&>(classifier);
//
//  return f;
//}

ostream &operator<<(ostream &f, SimileClassifier &classifier)
{
  if (!f.good())
  {
    cerr << "Openning file failed *****: " <<__FUNCTION__ << endl;
    return f;
  }

  f << classifier.m_ref_person_id << "\t" << classifier.m_region_id << endl;

  f << classifier.m_picked_classifiers.size() << endl;

   for (vector<tr1::shared_ptr<WeakClassifier> >::iterator it = classifier.m_picked_classifiers.begin();
        it != classifier.m_picked_classifiers.end();
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
//     f << string(buf) << endl;
     f << string(buf);
     {
       (*it)->feature_type(feature_type);
        sprintf(buf, "%lu", feature_type);

        string fn = FLAGS_simile_classifier_model_dir + "/" + ref_person_names[classifier.m_ref_person_id] + "-" + region_name[classifier.m_region_id] + "-" + string(buf) + ".txt";

        QpSvm *svm = (QpSvm*)(it->get());
        svm_save_model(fn.c_str(), svm->m_model);
     }

     if ((it + 1) != classifier.m_picked_classifiers.end())
     {
       f << endl;
     }
   }


  return f;
}

istream &operator>>(istream &f, SimileClassifier &classifier)
{
  if (!f.good())
  {
    cerr << "Openning file failed *****: " <<__FUNCTION__ << endl;
    return f;
  }

  f >> classifier.m_ref_person_id >> classifier.m_region_id;

  int num = 0;

  f >> num;

  classifier.m_picked_classifiers.clear();

  for (int i = 0; i < num; i++)
  {

    QpSvm *svm = new QpSvm;

    string str;
    uint64 feature_type;
    double d = 0.0;

    f >> str;
    sscanf(str.c_str(), "%lu", &feature_type);
    svm->set_feature_type(feature_type);

    f >> str;
    sscanf(str.c_str(), "%lf", &d);
    svm->set_alpha(d);

    {
      char buf[128];
      sprintf(buf, "%lu", feature_type);
      string fn = FLAGS_simile_classifier_model_dir + "/" + ref_person_names[classifier.m_ref_person_id] + "-" + region_name[classifier.m_region_id] + "-" + string(buf) + ".txt";
      if (svm->m_model != NULL)
      {
        svm_free_and_destroy_model(&(svm->m_model));
        svm->m_model = NULL;
      }
      svm->m_model = svm_load_model(fn.c_str());

      if (svm->m_model == NULL)
      {
        cerr << "Error in load libsvm model..." << endl;
      }
    }

//    f >> *svm;
//
//    if (!f.good())
//    {
//      cerr << "Error in loading Strong Classifier: " << __FILE__ << ", " << __LINE__ << endl;
//      break;
//    }

    classifier.m_picked_classifiers.push_back(tr1::shared_ptr<WeakClassifier>(svm));
  }

  if ((int)classifier.m_picked_classifiers.size() != num)
  {
    cerr << "Error in loading Strong Classifier: " << __FILE__ << ", " << __LINE__ << endl;
  }

  return f;
}

void SimileClassifier::detach_training_data()
{
  m_all_training_feature_vectors.clear();
  m_all_test_feature_vectors.clear();
  map<uint64, tr1::shared_ptr<Sample> >().swap(m_all_training_feature_vectors);
  map<uint64, tr1::shared_ptr<Sample> >().swap(m_all_test_feature_vectors);

  vector<uint64>().swap(m_all_pos_instance_id);
  vector<uint64>().swap(m_all_neg_instance_id);
  vector<uint64>().swap(m_training_pos_instance_id);
  vector<uint64>().swap(m_training_neg_instance_id);
  vector<uint64>().swap(m_test_pos_instance_id);
  vector<uint64>().swap(m_test_neg_instance_id);

  m_all_weak_classifiers.clear();
  map<uint64, tr1::shared_ptr<WeakClassifier> >().swap(m_all_weak_classifiers);

  vector<double>().swap(m_instance_weights);

  for (vector<tr1::shared_ptr<WeakClassifier> >::iterator it = m_picked_classifiers.begin();
      it != m_picked_classifiers.end();
      it++)
  {
//    ((SgdSvm*)(*it).get())->detach_temp_variables();
    ((QpSvm*)(*it).get())->detach_temp_variables();
  }
}
