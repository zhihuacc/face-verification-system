/*
 * strong_classifier.cpp
 *
 *  Created on: 13 Jul, 2013
 *      Author: harvey
 */

#include "attr_classifier.h"
#include "feature_extraction_public.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include "sgd_svm.h"
#include "qp_svm.h"
#include "flags.h"


//int StrongClassifier::assign_training_data(
//    const map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
//    const vector<uint> &pos_instance_id, const vector<uint> &neg_instance_id)
//{
////  m_all_feature_vectors = all_feature_vectors;
////  m_pos_instance_id = pos_instance_id;
////  m_neg_instance_id = neg_instance_id;
//
//  return 0;
//}

//int AttrClassifier::create_one_weak_classifier(uint64 feature_type)
//{
//  tr1::shared_ptr<WeakClassifier> wc(new WeakClassifier(feature_type));
//
////  wc->collect_training_samples(m_all_feature_vectors, m_pos_instance_id, m_neg_instance_id);
//  m_all_weak_classifiers.insert(pair<uint64, tr1::shared_ptr<WeakClassifier> >(feature_type, wc));
//  return 0;
//}
//
//int AttrClassifier::learn_all_weak_classifiers(const map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
//                                 const vector<uint> &pos_instance_id,
//                                 const vector<uint> &neg_instance_id)
//{
//
//  for (map<uint64, tr1::shared_ptr<WeakClassifier> >::const_iterator wc_it = m_all_weak_classifiers.begin();
//      wc_it != m_all_weak_classifiers.end();
//      wc_it++)
//  {
//    vector<tr1::shared_ptr<Sample> > training_samples;
//    uint64 feature_type = wc_it->first;
//    tr1::shared_ptr<WeakClassifier> wc = wc_it->second;
//
//    map<uint64, tr1::shared_ptr<Sample> >::const_iterator fv_it = all_feature_vectors.end();
//    for (vector<unsigned long long>::const_iterator inst_it = pos_instance_id.begin();
//        inst_it != pos_instance_id.end(); inst_it++)
//    {
//      unsigned long long fv_id = *inst_it << 16 | feature_type;
//
//      fv_it = all_feature_vectors.find(fv_id);
//      if (fv_it == all_feature_vectors.end())
//      {
//        cerr << "Cannot find feature vector for this image: " << *inst_it << endl;
//        continue;
//      }
//      fv_it->second->set_label(+1);
//      training_samples.push_back(fv_it->second);
//    }
//
//
//    fv_it = all_feature_vectors.end();
//    for (vector<unsigned long long>::const_iterator inst_it = neg_instance_id.begin();
//        inst_it != neg_instance_id.end(); inst_it++)
//    {
//      unsigned long long fv_id = *inst_it << 16 | feature_type;
//      fv_it = all_feature_vectors.find(fv_id);
//      if (fv_it == all_feature_vectors.end())
//      {
//        cerr << "Cannot find feature vector for this image: " << *inst_it << endl;
//        continue;
//      }
//      fv_it->second->set_label(-1);
//      training_samples.push_back(fv_it->second);
//    }
//
//    wc->learn(training_samples);
//
//    tr1::shared_ptr<vector<bool> > test_result(new vector<bool>);
//    double total_err_rate, false_pos_rate, false_neg_rate;
//    wc->test(training_samples, *test_result, total_err_rate, false_pos_rate, false_neg_rate);
//
//    m_all_weak_classifier_test_result.insert(pair<uint64, tr1::shared_ptr<vector<bool> > >(feature_type, test_result));
//  }
//
//  return 0;
//}
//
//int AttrClassifier::boosting()
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

AttrClassifier::~AttrClassifier()
{

}

int AttrClassifier::learn(const vector<tr1::shared_ptr<Sample> > &training_samples)
{

  learn_all_weak_classifiers(m_all_training_feature_vectors, m_training_pos_instance_id, m_training_neg_instance_id);
  boosting();
  cout << "Attr-"<<attribute_names[m_attr_id] << ", end" << endl << endl;
  return 0;
}
int AttrClassifier::learn(tr1::shared_ptr<Sample> training_sample)
{
  learn_all_weak_classifiers(m_all_training_feature_vectors, m_training_pos_instance_id, m_training_neg_instance_id);
  boosting();
  cout << "Attr-"<<attribute_names[m_attr_id] << ", end" << endl << endl;
  return 0;
}
int AttrClassifier::test(const vector<tr1::shared_ptr<Sample> > &test_samples,
                 vector<bool> &test_result, double &total_error_rate,
                 double &false_pos_rate, double &false_neg_rate,
                 double &true_pos_rate, double &true_neg_rate)
{
  tr1::shared_ptr<Sample> sample;
  double prediction;
  test(sample, prediction);
  return 0;
}
int AttrClassifier::test(tr1::shared_ptr<Sample> test_sample, double &prediction)
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

        uint64 fv_id = *test_inst_it << 16 | feature_type;
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

        uint64 fv_id = *test_inst_it << 16 | feature_type;
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
        << attribute_names[m_attr_id] << " error rate: "
        << (double)misclassification / (m_test_pos_instance_id.size() + m_test_neg_instance_id.size()) << endl
        << "test set size: " << (m_test_pos_instance_id.size() + m_test_neg_instance_id.size()) << endl;
    return 0;
}

//int AttrClassifier::load_all_training_samples(const string &labels_dir_name, const string &fv_set_dir)
//{
//  string full_name = labels_dir_name + "/" + attribute_names[m_attr_id] + "/positive.txt";
//  ifstream infile(full_name.c_str());
//
//
//  while (infile.good())
//  {
//    uint64 inst_id;
//    infile >> inst_id;
//
//    if (!infile.good())
//    {
//      break;
//    }
//
//    m_training_pos_instance_id.push_back(inst_id);
//  }
//
//  infile.close();
//
//  full_name = labels_dir_name + "/" + attribute_names[m_attr_id] + "/negative.txt";
//  infile.open(full_name.c_str());
//  while (infile.good())
//  {
//    uint64 inst_id;
//    infile >> inst_id;
//
//    if (!infile.good())
//    {
//      break;
//    }
//
//    m_training_neg_instance_id.push_back(inst_id);
//  }
//
//  infile.close();
//
//
//  for (vector<uint64>::iterator it = m_training_pos_instance_id.begin();
//      it != m_training_pos_instance_id.end();)
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
//      cerr << "File " << ss.str() << " not good. Next one" << endl;
//      it = m_training_pos_instance_id.erase(it);
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
//
//      m_all_training_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->label(), fv));
//      fv->set_label(1.0);
//    }
//    it++;
//  }
//
//  for (vector<uint64>::iterator it = m_training_neg_instance_id.begin();
//      it != m_training_neg_instance_id.end();)
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
//      cerr << "File " << ss.str() << " not good. Next one" << endl;
//      it = m_training_neg_instance_id.erase(it);
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
//
//      m_all_training_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->label(), fv));
//      fv->set_label(-1.0);
//    }
//    it++;
//  }
//
//  return 0;
//}
//
//int AttrClassifier::load_all_test_samples(const string &labels_dir_name, const string &fv_set_dir)
//{
//  string full_name = labels_dir_name + "/" + attribute_names[m_attr_id] + "/positive.txt";
//  ifstream infile(full_name.c_str());
//
//
//  while (infile.good())
//  {
//    uint64 inst_id;
//    infile >> inst_id;
//
//    if (!infile.good())
//    {
//      break;
//    }
//
//    m_test_pos_instance_id.push_back(inst_id);
//  }
//
//  infile.close();
//
//  full_name = labels_dir_name + "/" + attribute_names[m_attr_id] + "/negative.txt";
//  infile.open(full_name.c_str());
//  while (infile.good())
//  {
//    uint64 inst_id;
//    infile >> inst_id;
//
//    if (!infile.good())
//    {
//      break;
//    }
//
//    m_test_neg_instance_id.push_back(inst_id);
//  }
//
//  infile.close();
//
//  for (vector<uint64>::iterator it = m_test_pos_instance_id.begin();
//      it != m_test_pos_instance_id.end();)
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
//      cerr << "File " << ss.str() << " not good. Next one" << endl;
//      it = m_test_pos_instance_id.erase(it);
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
//
//      m_all_test_feature_vectors.insert(pair<unsigned long long, tr1::shared_ptr<Sample> >(fv->label(), fv));
//      fv->set_label(1.0);
//    }
//
//    it++;
//  }
//
//  for (vector<uint64>::iterator it = m_test_neg_instance_id.begin();
//      it != m_test_neg_instance_id.end();)
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
//      cerr << "File " << ss.str() << " not good. Next one" << endl;
//      it = m_test_neg_instance_id.erase(it);
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
//
//      m_all_test_feature_vectors.insert(pair<unsigned long long, tr1::shared_ptr<Sample> >(fv->label(), fv));
//      fv->set_label(-1.0);
//    }
//
//    it++;
//  }
//
//  return 0;
//}

int AttrClassifier::detach_training_data()
{
//  m_all_training_feature_vectors.clear();
//  m_training_pos_instance_id.clear();
//  m_training_neg_instance_id.clear();
//  m_all_weak_classifiers.clear();
////  map<uint64, tr1::shared_ptr<WeakClassifier> >::iterator it = m_picked_classifiers.begin();
//  vector<tr1::shared_ptr<WeakClassifier> >::iterator it = m_picked_classifiers.begin();
//  for (; it != m_picked_classifiers.end(); it++)
//  {
//    SgdSvm *svm = dynamic_cast<SgdSvm*>((*it).get());
//    if (svm == NULL)
//    {
//      continue;
//    }
//    svm->detach_last_test_result();
//  }
//  return 0;

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
  return 0;
}

//int AttrClassifier::detach_test_data()
//{
//  m_all_test_feature_vectors.clear();
//  m_test_pos_instance_id.clear();
//  m_test_neg_instance_id.clear();
//  return 0;
//}

int AttrClassifier::cross_validate()
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
  cout << "Attribute - " << attribute_names[m_attr_id]  << " - Cross Validation - 3 folds: " << endl;
  cout << "  Average Error Rate: " << err_rate << endl;


  return 0;
}



void AttrClassifier::assign_total_training_instance_id(const vector<uint64> &all_pos_instance_id, const vector<uint64> &all_neg_instance_id)
{
  m_all_pos_instance_id = all_pos_instance_id;
  m_all_neg_instance_id = all_neg_instance_id;
}

void AttrClassifier::assign_this_time_training_instance_id(const vector<uint64> &training_pos_instance_id,
                                            const vector<uint64> &training_neg_instance_id,
                                            const vector<uint64> &test_pos_instance_id,
                                            const vector<uint64> &test_neg_instance_id)
{
  m_training_pos_instance_id = training_pos_instance_id;
  m_training_neg_instance_id = training_neg_instance_id;

  m_test_pos_instance_id = test_pos_instance_id;
  m_test_neg_instance_id = test_neg_instance_id;
}

void AttrClassifier::assign_training_data(const map<uint64, tr1::shared_ptr<Sample> > &training_feature_vectors, const map<uint64, tr1::shared_ptr<Sample> > &test_feature_vectors)
{
  m_all_training_feature_vectors = training_feature_vectors;
  m_all_test_feature_vectors = test_feature_vectors;
}


int AttrClassifier::split_fold(int folds, int iter, const vector<uint64> &whole_set, vector<uint64> &training_set, vector<uint64> &test_set)
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

int AttrClassifier::blind_predict(const map<uint64, tr1::shared_ptr<Sample> > &all_fv, const vector<uint64> &test_set, vector<double> &all_prediction)
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

       uint64 fv_id = test_set[i] << 16 | feature_type;
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

//ostream &operator<<(ostream &f, AttrClassifier &classifier)
//{
//  if (!f.good())
//  {
//    cerr << "Openning file failed *****: " <<__FUNCTION__ << endl;
//    return f;
//  }
//
//  f << classifier.m_attr_id << endl;
//  f << static_cast<AdaBoost2&>(classifier) << endl;
//
//  return f;
//}
//
//istream &operator>>(istream &f, AttrClassifier &classifier)
//{
//  if (!f.good())
//  {
//    cerr << "Openning file failed *****: " <<__FUNCTION__ << endl;
//    return f;
//  }
//
//  f >> classifier.m_attr_id;
//
//  f >> static_cast<AdaBoost2&>(classifier);
//
//  return f;
//}

ostream &operator<<(ostream &f, AttrClassifier &classifier)
{
  if (!f.good())
  {
    cerr << "Openning file failed *****: " <<__FUNCTION__ << endl;
    return f;
  }

  f << classifier.m_attr_id << endl;

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

        string fn = FLAGS_attr_classifier_model_dir + "/" + attribute_names[classifier.m_attr_id] + "-" + string(buf) + ".txt";

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

istream &operator>>(istream &f, AttrClassifier &classifier)
{
  if (!f.good())
  {
    cerr << "Openning file failed *****: " <<__FUNCTION__ << endl;
    return f;
  }

  f >> classifier.m_attr_id;

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
      string fn = FLAGS_attr_classifier_model_dir + "/" + attribute_names[classifier.m_attr_id] + "-" + string(buf) + ".txt";
      if (svm->m_model != NULL)
      {
        svm_free_and_destroy_model(&(svm->m_model));
        svm->m_model = NULL;
      }
      svm->m_model = svm_load_model(fn.c_str());

      if (svm->m_model == NULL)
      {
        cerr << "Err in load libsvm model.." << endl;
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


