/*
 * verifier_producer.cpp
 *
 *  Created on: 8 Aug, 2013
 *      Author: harvey
 */

#include "verifier_feature_producer.h"
#include "attr_classifier.h"

#include "simile_classifier.h"
#include "flags.h"

#include <cstdlib>
#include <cmath>

using namespace std;

extern map<string, uint64> pubfig_name_id_mapping;
extern map<uint64, string> pubfig_id_name_mapping;
extern SimileClassifier simile_classifiers[60][REGION_NUM];
extern AttrClassifier attr_classifiers[ATTR_NUM];
extern vector<Fold> all_folds;
extern set<uint64> all_benchmark_instance_id;
extern string ref_person_names[REF_PERSON_NUM];

extern int load_feature_vectors_by_instance_id(map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors, vector<uint64> &instance_id);


int load_pubfig_verification_benchmark(const string &benchmark)
{
  ifstream benchmark_file(benchmark.c_str());

  if (!benchmark_file.good())
  {
    return -1;
  }

  string line;
  int folds = 0;
  benchmark_file >> folds;

  all_folds.reserve(folds);
  all_folds.resize(folds);

  for (int i = 0; i <  folds; i++)
  {
    int pos_pairs = 0, neg_pairs = 0;
    benchmark_file >> pos_pairs >> neg_pairs;
    if (!benchmark_file.good())
    {
      break;
    }

    for (int pos = 0; pos < pos_pairs; )
    {
      string line;
      string name;
      string id;
      getline(benchmark_file, line);
      if (!benchmark_file.good())
      {
        break;
      }

      if (line.empty())
      {
        continue;
      }

      pos++;

      stringstream ss(line);
      getline(ss, name, '\t');
      getline(ss, id, '\t');

      string full = name+"-"+id;

      map<string, uint64>::iterator it= pubfig_name_id_mapping.find(full);
      if (it == pubfig_name_id_mapping.end())
      {
        continue;
      }

      VerificationPair pair;
      pair.first = it->second;

      getline(ss, name, '\t');
      getline(ss, id, '\t');
      full = name + "-" + id;

      it= pubfig_name_id_mapping.find(full);
      if (it == pubfig_name_id_mapping.end())
      {
        continue;
      }
      pair.second = it->second;

      all_folds[i].pos_pairs.push_back(pair);
      all_benchmark_instance_id.insert(pair.first);
      all_benchmark_instance_id.insert(pair.second);
    }

    cout << "Available Pos Pairs in Fold : " << i <<", " << all_folds[i].pos_pairs.size() << endl;

    for (int neg = 0; neg < neg_pairs; )
    {
      string line;
      string name;
      string id;

      getline(benchmark_file, line);
      if (!benchmark_file.good())
      {
        break;
      }

      if (line.empty())
      {
        continue;
      }

      neg++;

      stringstream ss(line);
      getline(ss, name, '\t');
      getline(ss, id, '\t');

      string full = name+"-"+id;

      map<string, uint64>::iterator it= pubfig_name_id_mapping.find(full);
      if (it == pubfig_name_id_mapping.end())
      {
        continue;
      }

      VerificationPair pair;
      pair.first = it->second;

      getline(ss, name, '\t');
      getline(ss, id, '\t');
      full = name + "-" + id;

      it= pubfig_name_id_mapping.find(full);
      if (it == pubfig_name_id_mapping.end())
      {
        continue;
      }
      pair.second = it->second;

      all_folds[i].neg_pairs.push_back(pair);
      all_benchmark_instance_id.insert(pair.first);
      all_benchmark_instance_id.insert(pair.second);
    }

    cout << "Available Neg Pairs in Fold : " << i <<", " << all_folds[i].neg_pairs.size() << endl;
  }

  return 0;
}

static string compose_feature_line(tr1::shared_ptr<Sample> fv)
{
  stringstream ss;
  char buf[128];

  sprintf(buf, "%lu", fv->feature_vector_id());

  ss << string(buf) << " ";

  for (int i = 1; i <= fv->x()->size(); i++)
  {
    sprintf(buf, "%lf", fv->x()->valueAt(i));
    ss << i << ":" << string(buf) << " ";
  }

  return ss.str();
}

static string compose_combined_fv_line(tr1::shared_ptr<Sample> fv1, tr1::shared_ptr<Sample> fv2)
{
  if (fv1->feature_vector_id() != fv2->feature_vector_id())
  {
    cerr << "Error in concatenating two features **********" << endl;
    exit(-1);
  }
  stringstream ss;
  char buf[128];

  sprintf(buf, "%lu", fv1->feature_vector_id());

  ss << string(buf) << " ";

  int fv1_length = fv1->x()->size();
  int fv2_length = fv2->x()->size();
  for (int i = 1; i <= fv1_length; i++)
  {
    sprintf(buf, "%lf", fv1->x()->valueAt(i));
    ss << i << ":" <<  string(buf) << " ";
  }


  for (int i = 1; i <= fv2_length; i++)
  {
    sprintf(buf, "%lf", fv2->x()->valueAt(i));
    ss << fv1_length + i << ":" <<  string(buf) << " ";
  }

  return ss.str();
}

static string compose_combined_fv_line2(tr1::shared_ptr<Sample> fv1, tr1::shared_ptr<Sample> fv2)
{

  stringstream ss;
  char buf[128];
  int fv1_length = fv1->x()->size();
  int fv2_length = fv2->x()->size();

  if (fv1_length != fv2_length)
  {
    cerr << "Error in two fv length " <<  __FUNCTION__ << endl;
    exit(-1);
  }

  uint64 two = fv1->feature_vector_id() << 32 | fv2->feature_vector_id();

  sprintf(buf, "%lu", two);

  ss << string(buf) << " ";

  for (int i = 1; i <= fv1_length; i++)
  {
    double d = abs(fv1->x()->valueAt(i) - fv2->x()->valueAt(i));
    sprintf(buf, "%lf", d);
    ss << 2 * i - 1 << ":" << string(buf) << " ";

    d = fv1->x()->valueAt(i) * fv2->x()->valueAt(i);
    sprintf(buf, "%lf", d);
    ss << 2 * i << ":" << string(buf) << " ";
  }


//  for (int i = 1; i <= fv1_length; i++)
//  {
//    sprintf(buf, "%lf", fv1->x()->valueAt(i));
//    ss << i << ":" <<  string(buf) << " ";
//  }
//
//
//  for (int i = 1; i <= fv2_length; i++)
//  {
//    sprintf(buf, "%lf", fv2->x()->valueAt(i));
//    ss << fv1_length + i << ":" <<  string(buf) << " ";
//  }

  return ss.str();
}

int produce_trait_vectors()
{
  vector<uint64> all_test_instance_id(all_benchmark_instance_id.begin(), all_benchmark_instance_id.end());

//while (false)
//{
  string fname = "/home/harvey/Dataset/qp-verifier/simile-verifier.txt";
  ofstream simile_fv_file(fname.c_str());

  fname = "/home/harvey/Dataset/qp-verifier/attr-verifier.txt";
  ofstream attr_fv_file(fname.c_str());

  fname = "/home/harvey/Dataset/qp-verifier/combined-verifier.txt";
  ofstream combined_fv_file(fname.c_str());

  vector<uint64> subset_test_instance_id;
  int step = 3000;
  int iteration = all_test_instance_id.size() / step + 1;
  for (int i = 0; i < iteration; i++)
  {
    cout << "Iter " << i << " to produce trait vectors." << endl;
    int start = i * step;
    int stop = i * step + step;

    start = min((int)all_test_instance_id.size(), start);
    stop = min((int)all_test_instance_id.size(), stop);

    if (start == stop)
    {
      break;
    }

    map<uint64, tr1::shared_ptr<Sample> > all_feature_set;

    subset_test_instance_id.assign(all_test_instance_id.begin() + start, all_test_instance_id.begin() + stop);

    load_feature_vectors_by_instance_id(all_feature_set, subset_test_instance_id);

    vector<tr1::shared_ptr<Sample> > subset_simile_prediction;
    vector<tr1::shared_ptr<Sample> > subset_attr_prediction;


    subset_simile_prediction.reserve(subset_test_instance_id.size());
    subset_simile_prediction.resize(subset_test_instance_id.size());

    subset_attr_prediction.reserve(subset_test_instance_id.size());
    subset_attr_prediction.resize(subset_test_instance_id.size());

    for (int j = 0; j < (int)subset_test_instance_id.size(); j++)
    {
      subset_simile_prediction[j] = tr1::shared_ptr<Sample>(new Sample);
      subset_attr_prediction[j] = tr1::shared_ptr<Sample>(new Sample);
      subset_simile_prediction[j]->set_feature_vector_id(subset_test_instance_id[j]);
      subset_attr_prediction[j]->set_feature_vector_id(subset_test_instance_id[j]);
    }

    for (int person = 0; person < REF_PERSON_NUM; person++)
    {
      cout << "Producing trait vectors for " << person << " " << ref_person_names[person] << endl;
      for (int region = REGION_START; region < REGION_NUM; region++)
      {
        string model_name = FLAGS_simile_classifier_model_dir +"/" + ref_person_names[person] + "-" + region_name[region] + ".txt";
        ifstream this_model(model_name.c_str());

        this_model >> simile_classifiers[person][region];

        if (!this_model.eof())
        {
          cerr << "Error in loading simile model " << endl;
          exit(-1);
        }

        vector<double> this_prediction;
//        for (int i = 0; i < (int)subset_test_instance_id.size(); i++)
//        {
//          this_prediction.push_back(i);
//        }
        simile_classifiers[person][region].blind_predict(all_feature_set, subset_test_instance_id, this_prediction);

        for (int i = 0; i < (int)subset_test_instance_id.size(); i++)
        {
          subset_simile_prediction[i]->x()->setValue(1 + person * REGION_NUM + region, this_prediction[i]);
        }

        simile_classifiers[person][region].reset();
      }
    }

    for (int attr = 0; attr < ATTR_NUM; attr++)
    {
      string model_name = FLAGS_attr_classifier_model_dir +"/" + attribute_names[attr] + ".txt";
      ifstream this_model(model_name.c_str());

      this_model >> attr_classifiers[attr];
      if (!this_model.eof())
      {
        cerr << "Error in loading attr model " << endl;
        exit(-1);
      }

      vector<double> this_prediction;
//      for (int i = 0; i < (int)subset_test_instance_id.size(); i++)
//      {
//        this_prediction.push_back(i);
//      }
      attr_classifiers[attr].blind_predict(all_feature_set, subset_test_instance_id, this_prediction);

      for (int i = 0; i < (int)subset_test_instance_id.size(); i++)
      {
        subset_attr_prediction[i]->x()->setValue(1 + attr, this_prediction[i]);
      }

      attr_classifiers[attr].reset();
    }

    for (int i = 0; i < (int)subset_test_instance_id.size(); i++)
    {
      string line;
      line = compose_feature_line(subset_simile_prediction[i]);
      simile_fv_file << line << endl;

      line = compose_feature_line(subset_attr_prediction[i]);
      attr_fv_file << line << endl;

      line = compose_combined_fv_line(subset_simile_prediction[i], subset_attr_prediction[i]);
      combined_fv_file << line << endl;
    }

  }

//}

  cout << "Produce Raw Trait Vectors, Done ..." << endl;
  cout << "Scaling Simile FV ..." << endl;


  map<uint64, tr1::shared_ptr<Sample> > all_fvs;
  tr1::shared_ptr<Sample> min_simile_fv, max_simile_fv, min_attr_fv, max_attr_fv, min_combined_fv, max_combined_fv;
  ifstream in_simile_file("/home/harvey/Dataset/qp-verifier/simile-verifier.txt");
  ofstream out_simile_file("/home/harvey/Dataset/qp-verifier/simile-verifier-scaled.txt");

  bool first = true;
  while (!in_simile_file.eof())
  {
    tr1::shared_ptr<Sample> fv(new Sample);
    in_simile_file >> *fv;
    if (!in_simile_file.good())
    {
      break;
    }

    all_fvs.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->feature_vector_id(), fv));

    if (first)
    {
      min_simile_fv.reset(new Sample);
      max_simile_fv.reset(new Sample);

      for (int i = 1; i <= fv->x()->size(); i++)
      {
        min_simile_fv->x()->setValue(i, fv->x()->valueAt(i));
        max_simile_fv->x()->setValue(i, fv->x()->valueAt(i));
      }

      first = false;
    }

    if (min_simile_fv->x()->size() != fv->x()->size())
    {
      cerr << "Different Length of MIN Simile FV *********" << endl;
      exit(-1);
    }

    for (int j = 1; j <= min_simile_fv->x()->size(); j++)
    {
      double d = fv->x()->valueAt(j);
      if (min_simile_fv->x()->valueAt(j) > d)
      {
        min_simile_fv->x()->setValue(j, d);
      }
      else if (max_simile_fv->x()->valueAt(j) < d)
      {
        max_simile_fv->x()->setValue(j, d);
      }
    }

  }

  for (map<uint64, tr1::shared_ptr<Sample> >::iterator it = all_fvs.begin();
       it != all_fvs.end();
       it++)
  {
    tr1::shared_ptr<Sample> fv = it->second;
    for (int j = 1; j <= min_simile_fv->x()->size(); j++)
    {
      double scaled = 0.0;
      if ((max_simile_fv->x()->valueAt(j) - min_simile_fv->x()->valueAt(j)) > 0.00001)
      {
        scaled = -1 + ((fv->x()->valueAt(j) - min_simile_fv->x()->valueAt(j)) * (2)) / (max_simile_fv->x()->valueAt(j) - min_simile_fv->x()->valueAt(j));
      }


      fv->x()->setValue(j, scaled);
    }
//    string line = compose_feature_line(fv);
//    out_simile_file << line << endl;
  }

  for (vector<Fold>::iterator it = all_folds.begin();
       it != all_folds.end();
       it++)
  {
    Fold &this_fold = *it;

    for (vector<VerificationPair>::iterator it2 = this_fold.pos_pairs.begin();
         it2 != this_fold.pos_pairs.end();
         it2++)
    {
      tr1::shared_ptr<Sample> fv1, fv2;

      map<uint64, tr1::shared_ptr<Sample> >::iterator it3 = all_fvs.find(it2->first);
      if (it3 == all_fvs.end())
      {
        continue;
      }

      fv1 = it3->second;

      it3 = all_fvs.find(it2->second);
      if (it3 == all_fvs.end())
      {
        continue;
      }

      fv2 = it3->second;

      string line = compose_combined_fv_line2(fv1, fv2);
      out_simile_file << line << endl;
    }

    for (vector<VerificationPair>::iterator it2 = this_fold.neg_pairs.begin();
         it2 != this_fold.neg_pairs.end();
         it2++)
    {
      tr1::shared_ptr<Sample> fv1, fv2;

      map<uint64, tr1::shared_ptr<Sample> >::iterator it3 = all_fvs.find(it2->first);
      if (it3 == all_fvs.end())
      {
        continue;
      }

      fv1 = it3->second;

      it3 = all_fvs.find(it2->second);
      if (it3 == all_fvs.end())
      {
        continue;
      }

      fv2 = it3->second;

      string line = compose_combined_fv_line2(fv1, fv2);
      out_simile_file << line << endl;
    }
  }

  all_fvs.clear();
  map<uint64, tr1::shared_ptr<Sample> >().swap(all_fvs);
  first = true;

  cout << "scaling Simile FV, done" << endl;
  cout << "Scaling Attr FV ...." << endl;

  ifstream in_attr_file("/home/harvey/Dataset/qp-verifier/attr-verifier.txt");
  ofstream out_attr_file("/home/harvey/Dataset/qp-verifier/attr-verifier-scaled.txt");

  while (!in_attr_file.eof())
  {
    tr1::shared_ptr<Sample> fv(new Sample);
    in_attr_file >> *fv;
    if (!in_attr_file.good())
    {
      break;
    }

    all_fvs.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->feature_vector_id(), fv));

    if (first)
    {
      min_attr_fv.reset(new Sample);
      max_attr_fv.reset(new Sample);

      for (int i = 1; i <= fv->x()->size(); i++)
      {
        min_attr_fv->x()->setValue(i, fv->x()->valueAt(i));
        max_attr_fv->x()->setValue(i, fv->x()->valueAt(i));
      }

      first = false;
    }

    if (min_attr_fv->x()->size() != fv->x()->size())
    {
      cerr << "Different Length of MIN Attr FV *********" << endl;
      exit(-1);
    }

    for (int j = 1; j <= min_attr_fv->x()->size(); j++)
    {
      double d = fv->x()->valueAt(j);
      if (min_attr_fv->x()->valueAt(j) > d)
      {
        min_attr_fv->x()->setValue(j, d);
      }
      else if (max_attr_fv->x()->valueAt(j) < d)
      {
        max_attr_fv->x()->setValue(j, d);
      }
    }

  }

  for (map<uint64, tr1::shared_ptr<Sample> >::iterator it = all_fvs.begin();
       it != all_fvs.end();
       it++)
  {
    tr1::shared_ptr<Sample> fv = it->second;
    for (int j = 1; j <= min_attr_fv->x()->size(); j++)
    {
      double scaled = 0.0;
      if ((max_attr_fv->x()->valueAt(j) - min_attr_fv->x()->valueAt(j)) > 0.00001)
      {
        scaled = -1 + ((fv->x()->valueAt(j) - min_attr_fv->x()->valueAt(j)) * (2)) / (max_attr_fv->x()->valueAt(j) - min_attr_fv->x()->valueAt(j));
      }


      fv->x()->setValue(j, scaled);
    }
//    string line = compose_feature_line(fv);
//    out_attr_file << line << endl;
  }

  for (vector<Fold>::iterator it = all_folds.begin();
        it != all_folds.end();
        it++)
   {
     Fold &this_fold = *it;

     for (vector<VerificationPair>::iterator it2 = this_fold.pos_pairs.begin();
          it2 != this_fold.pos_pairs.end();
          it2++)
     {
       tr1::shared_ptr<Sample> fv1, fv2;

       map<uint64, tr1::shared_ptr<Sample> >::iterator it3 = all_fvs.find(it2->first);
       if (it3 == all_fvs.end())
       {
         continue;
       }

       fv1 = it3->second;

       it3 = all_fvs.find(it2->second);
       if (it3 == all_fvs.end())
       {
         continue;
       }

       fv2 = it3->second;

       string line = compose_combined_fv_line2(fv1, fv2);
       out_attr_file << line << endl;
     }

     for (vector<VerificationPair>::iterator it2 = this_fold.neg_pairs.begin();
          it2 != this_fold.neg_pairs.end();
          it2++)
     {
       tr1::shared_ptr<Sample> fv1, fv2;

       map<uint64, tr1::shared_ptr<Sample> >::iterator it3 = all_fvs.find(it2->first);
       if (it3 == all_fvs.end())
       {
         continue;
       }

       fv1 = it3->second;

       it3 = all_fvs.find(it2->second);
       if (it3 == all_fvs.end())
       {
         continue;
       }

       fv2 = it3->second;

       string line = compose_combined_fv_line2(fv1, fv2);
       out_attr_file << line << endl;
     }
   }

  all_fvs.clear();
  map<uint64, tr1::shared_ptr<Sample> >().swap(all_fvs);
  first = true;

  cout << "Scaling Attr FV done." << endl;
  cout << "Scaling Combined FV ..." << endl;


  ifstream in_combined_file("/home/harvey/Dataset/qp-verifier/combined-verifier.txt");
  ofstream out_combined_file("/home/harvey/Dataset/qp-verifier/combined-verifier-scaled.txt");

  while (!in_combined_file.eof())
  {
    tr1::shared_ptr<Sample> fv(new Sample);
    in_combined_file >> *fv;
    if (!in_combined_file.good())
    {
      break;
    }

    all_fvs.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->feature_vector_id(), fv));

    if (first)
    {
      min_combined_fv.reset(new Sample);
      max_combined_fv.reset(new Sample);

      for (int i = 1; i <= fv->x()->size(); i++)
      {
        min_combined_fv->x()->setValue(i, fv->x()->valueAt(i));
        max_combined_fv->x()->setValue(i, fv->x()->valueAt(i));
      }

      first = false;
    }

    if (min_combined_fv->x()->size() != fv->x()->size())
    {
      cerr << "Different Length of MIN Combined FV *********" << endl;
      exit(-1);
    }

    for (int j = 1; j <= min_combined_fv->x()->size(); j++)
    {
      double d = fv->x()->valueAt(j);
      if (min_combined_fv->x()->valueAt(j) > d)
      {
        min_combined_fv->x()->setValue(j, d);
      }
      else if (max_combined_fv->x()->valueAt(j) < d)
      {
        max_combined_fv->x()->setValue(j, d);
      }
    }

  }

  for (map<uint64, tr1::shared_ptr<Sample> >::iterator it = all_fvs.begin();
       it != all_fvs.end();
       it++)
  {
    tr1::shared_ptr<Sample> fv = it->second;
    for (int j = 1; j <= min_combined_fv->x()->size(); j++)
    {
      double scaled = 0.0;
      if ((max_combined_fv->x()->valueAt(j) - min_combined_fv->x()->valueAt(j)) > 0.00001)
      {
        scaled = -1 + ((fv->x()->valueAt(j) - min_combined_fv->x()->valueAt(j)) * (2)) / (max_combined_fv->x()->valueAt(j) - min_combined_fv->x()->valueAt(j));
      }


      fv->x()->setValue(j, scaled);
    }
//    string line = compose_feature_line(fv);
//    out_combined_file << line;
  }

  for (vector<Fold>::iterator it = all_folds.begin();
        it != all_folds.end();
        it++)
   {
     Fold &this_fold = *it;

     for (vector<VerificationPair>::iterator it2 = this_fold.pos_pairs.begin();
          it2 != this_fold.pos_pairs.end();
          it2++)
     {
       tr1::shared_ptr<Sample> fv1, fv2;

       map<uint64, tr1::shared_ptr<Sample> >::iterator it3 = all_fvs.find(it2->first);
       if (it3 == all_fvs.end())
       {
         continue;
       }

       fv1 = it3->second;

       it3 = all_fvs.find(it2->second);
       if (it3 == all_fvs.end())
       {
         continue;
       }

       fv2 = it3->second;

       string line = compose_combined_fv_line2(fv1, fv2);
       out_combined_file << line << endl;
     }

     for (vector<VerificationPair>::iterator it2 = this_fold.neg_pairs.begin();
          it2 != this_fold.neg_pairs.end();
          it2++)
     {
       tr1::shared_ptr<Sample> fv1, fv2;

       map<uint64, tr1::shared_ptr<Sample> >::iterator it3 = all_fvs.find(it2->first);
       if (it3 == all_fvs.end())
       {
         continue;
       }

       fv1 = it3->second;

       it3 = all_fvs.find(it2->second);
       if (it3 == all_fvs.end())
       {
         continue;
       }

       fv2 = it3->second;

       string line = compose_combined_fv_line2(fv1, fv2);
       out_combined_file << line << endl;
     }
   }

  cout << "Scaling combined FV, done" << endl;
  return 0;
}

int reload_all_models()
{
  for (int attr = 0; attr < ATTR_NUM; attr++)
  {
    string model_name = FLAGS_attr_classifier_model_dir +"/" + attribute_names[attr] + ".txt";
    ifstream this_model(model_name.c_str());

    this_model >> attr_classifiers[attr];
    if (!this_model.eof())
    {
      cerr << "Error in loading attr model " << endl;
      exit(-1);
    }
  }

  for (int ref_person_id = 0; ref_person_id < REF_PERSON_NUM; ref_person_id++)
  {
    for (int region_id = 0; region_id < REGION_NUM; region_id++)
    {
      string model_name = FLAGS_simile_classifier_model_dir +"/" + ref_person_names[ref_person_id] + "-" + region_name[region_id] + ".txt";
      ifstream this_model(model_name.c_str());

      this_model >> simile_classifiers[ref_person_id][region_id];

      if (!this_model.eof())
      {
        cerr << "Error in loading simile model " << endl;
        exit(-1);
      }
    }
  }

  return 0;
}
