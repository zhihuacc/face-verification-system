/*
 * main.cpp
 *
 *  Created on: 23 Jun, 2013
 *      Author: harvey
 */

#include "flags.h"
#include "assert.h"
#include "sample.h"
#include "feature_extraction_public.h"
#include "sgd_svm.h"
#include <iostream>
#include <fstream>
#include <tr1/memory>
#include <map>
#include "adaboost.h"
#include "verifier_feature_producer.h"
#include "verifier.h"

#include "qp_svm.h"


using namespace std;


int load_feature_set(const string &file_name, int max_rows, map<unsigned long long, std::tr1::shared_ptr<Sample> > &sample_map, int &max_dim)
{
  cout << "Loading sample file: " << file_name << endl;
  ifstream sample_file(file_name.c_str());

  max_dim = 0;
  if (!sample_file.good())
      assertfail("This file is not good.");

    int rows = 0;
    while (sample_file.good() && max_rows--)
    {
      tr1::shared_ptr<Sample> fv(new Sample);
      sample_file >> *fv;
      if (!sample_file.good()) {
        break;
      }

      sample_map.insert(pair<unsigned long long, std::tr1::shared_ptr<Sample> >(
              static_cast<unsigned long long>(fv->label()), fv));

      if (fv->x()->size() > max_dim) {
        max_dim = fv->x()->size();
      }
      rows++;
    }

    cout << "Total samples: " << rows << endl;
    return 0;
}



//AdaBoost attribute_classifiers[ATTR_NUM] = {
//    AdaBoost(ATTR_MALE),
//    AdaBoost(ATTR_WHITE),
//    AdaBoost(ATTR_ASIAN),
//    AdaBoost(ATTR_BLACK),
//    AdaBoost(ATTR_BLACK_HAIR),
//    AdaBoost(ATTR_BLOND_HAIR),
//    AdaBoost(ATTR_BROWN_HAIR),
//    AdaBoost(ATTR_EYEGLASS),
//    AdaBoost(ATTR_INDIAN),
//    AdaBoost(ATTR_SUNGLASS)
//};



#include <highgui/highgui.hpp>

//int main(int argc, char **argv)
//{
//  google::ParseCommandLineFlags(&argc, &argv, true);
//  setbuf(stdout,NULL);
//
//  // Load training samples
//  for (int attr = ATTR_START; attr < ATTR_NUM; attr++)
//  {
//    cout << "Load training samples ..." << endl;
//    attribute_classifiers[attr].load_all_training_samples(FLAGS_training_labeling_dir + "/" + attribute_names[attr],
//                                                           FLAGS_feature_set_dir);
//
//    attribute_classifiers[attr].initialize_weights();
//
//    cout << "Learn weak classifiers ..." << endl;
//    attribute_classifiers[attr].learn_weak_classifiers();
//
//
//
//    /********boost***********/
//    cout << "Boosting ..." << endl;
//
//    attribute_classifiers[attr].boosting();
//
//
//    attribute_classifiers[attr].shrink_memory();
//
//
//    /********** Test *****************/
//
//    attribute_classifiers[attr].load_all_test_samples(FLAGS_test_labeling_dir + "/" + attribute_names[attr],
//                                                               FLAGS_feature_set_dir);
//
//
//
//    attribute_classifiers[attr].test();
//
//    attribute_classifiers[attr].shrink_memory();
//
//  }
//}

#include "attr_classifier.h"

AttrClassifier attr_classifiers[ATTR_NUM] = {
    AttrClassifier(ATTR_ASIAN),
    AttrClassifier(ATTR_BLACK),
    AttrClassifier(ATTR_BLACK_HAIR),
    AttrClassifier(ATTR_BLOND_HAIR),
    AttrClassifier(ATTR_BOLD),
    AttrClassifier(ATTR_BROWN_HAIR),
    AttrClassifier(ATTR_CHILD),
    AttrClassifier(ATTR_COLORFUL),
    AttrClassifier(ATTR_ENVIRONMENT),
    AttrClassifier(ATTR_EYEGLASSES),
    AttrClassifier(ATTR_EYE_OPEN),
    AttrClassifier(ATTR_EYE_WEAR),
    AttrClassifier(ATTR_INDIAN),
    AttrClassifier(ATTR_JAW_BONES),
    AttrClassifier(ATTR_MALE),
    AttrClassifier(ATTR_MIDDLE_AGED),
    AttrClassifier(ATTR_MOUTH_CLOSED),
    AttrClassifier(ATTR_MOUTH_OPEN),
    AttrClassifier(ATTR_NOSE_MOUTH_LINE),
    AttrClassifier(ATTR_SENIOR),
    AttrClassifier(ATTR_SHARP_JAW),
    AttrClassifier(ATTR_SIMILING),
    AttrClassifier(ATTR_STRAIGHT_HAIR),
    AttrClassifier(ATTR_SUNGLASSES),
    AttrClassifier(ATTR_WEARING_HAT),
    AttrClassifier(ATTR_WHITE),
    AttrClassifier(ATTR_YOUTH)
};

#include "simile_classifier.h"


SimileClassifier simile_classifiers[REF_PERSON_NUM][REGION_NUM];

map<string, uint64> pubfig_name_id_mapping;
map<uint64, string> pubfig_id_name_mapping;

Verifier simile_verifier;
Verifier attr_verifier;
Verifier combined_verifier;

int load_feature_vectors_by_instance_id(map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
                                        vector<uint64> &instance_id)
{
  for (vector<uint64>::iterator it = instance_id.begin();
        it != instance_id.end();)
    {
      stringstream ss;
      ss << FLAGS_feature_set_dir;
      ss << "/";
      ss.width(5);
      ss.fill('0');
      ss << *it << ".txt";

      ifstream fv_file(ss.str().c_str());
      if (!fv_file.good())
      {
        //cerr << "File " << ss.str() << " not good. Next one" << endl;
        it = instance_id.erase(it);
        continue;
      }

      int fv_num = 0;
      while (fv_file.good())
      {
        tr1::shared_ptr<Sample> fv(new Sample);

        fv_file >> *fv;
        if (!fv_file.good())
        {
          break;
        }
        fv_num++;

        all_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->feature_vector_id(), fv));
        fv->set_label(0.0);
      }
      if (fv_num != 288)
      {
        cerr << "Feature File not compleete: " << ss.str() << endl;
      }
      it++;
    }


    cout << "Available&Effective Test Negative Feature Set Files: " << instance_id.size() << endl;
  return 0;
}

int load_all_pubfig_training_samples(const string &labels_dir_name, const string &fv_set_dir,
                                     map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
                                     vector<uint64> &pos_instance_id, vector<uint64> &neg_instance_id)
{
  string full_name = labels_dir_name + "/positive.txt";
  ifstream infile(full_name.c_str());


  while (infile.good())
  {
    string name;
    getline(infile, name);
    if (!infile.good())
    {
      break;
    }
    map<string, uint64>::iterator it = pubfig_name_id_mapping.find(name);
    if (it == pubfig_name_id_mapping.end())
    {
      continue;
    }

    pos_instance_id.push_back(it->second);
  }

  infile.close();

  full_name = labels_dir_name + "/negative.txt";
  infile.open(full_name.c_str());
  while (infile.good())
  {
    string name;
    getline(infile, name);
    if (!infile.good())
    {
      break;
    }
    map<string, uint64>::iterator it = pubfig_name_id_mapping.find(name);
    if (it == pubfig_name_id_mapping.end())
    {
      continue;
    }

    neg_instance_id.push_back(it->second);
  }

  infile.close();

  int pos_inst_num = pos_instance_id.size(), neg_inst_num = neg_instance_id.size();
  random_shuffle(pos_instance_id.begin(), pos_instance_id.end());
  random_shuffle(neg_instance_id.begin(), neg_instance_id.end());
//  if ((double)pos_inst_num / neg_inst_num > 1.2)
//  {
//    vector<uint64>::iterator start = pos_instance_id.begin() + (int)(neg_inst_num * 1.2);
//    pos_instance_id.erase(start, pos_instance_id.end());
//  }

  if ((double)pos_inst_num / neg_inst_num < 1.5)
  {
    vector<uint64>::iterator start = neg_instance_id.begin() + (int)(pos_inst_num / 1.5);
    neg_instance_id.erase(start, neg_instance_id.end());
  }

  for (vector<uint64>::iterator it = pos_instance_id.begin();
      it != pos_instance_id.end();)
  {
    stringstream ss;
    ss << fv_set_dir;
    ss << "/";
    ss.width(5);
    ss.fill('0');
    ss << *it << ".txt";

    ifstream fv_file(ss.str().c_str());
    if (!fv_file.good())
    {
      //cerr << "File " << ss.str() << " not good. Next one" << endl;
      it = pos_instance_id.erase(it);
      continue;
    }

    int fv_num = 0;
    while (fv_file.good())
    {
      tr1::shared_ptr<Sample> fv(new Sample);

      fv_file >> *fv;
      if (!fv_file.good())
      {
        break;
      }
      fv_num++;

      all_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->feature_vector_id(), fv));
      fv->set_label(1.0);
    }
    if (fv_num != 288)
    {
      cerr << "Feature File not complete: " << ss.str() << endl;
    }
    it++;
  }

  for (vector<uint64>::iterator it = neg_instance_id.begin();
      it != neg_instance_id.end();)
  {
    stringstream ss;
    ss << fv_set_dir;
    ss << "/";
    ss.width(5);
    ss.fill('0');
    ss << *it << ".txt";

    ifstream fv_file(ss.str().c_str());
    if (!fv_file.good())
    {
      //cerr << "File " << ss.str() << " not good. Next one" << endl;
      it = neg_instance_id.erase(it);
      continue;
    }

    int fv_num = 0;
    while (fv_file.good())
    {
      tr1::shared_ptr<Sample> fv(new Sample);

      fv_file >> *fv;
      if (!fv_file.good())
      {
        break;
      }

      fv_num++;

      all_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->feature_vector_id(), fv));
      fv->set_label(-1.0);
    }

    if (fv_num != 288)
    {
      cerr << "Feature File not compleete: " << ss.str() << endl;
    }
    it++;
  }

  cout << "Available&Effective Test Positive Feature Set Files: " << pos_instance_id.size() << endl;

  cout << "Available&Effective Test Negative Feature Set Files: " << neg_instance_id.size() << endl;


  return 0;
}

int load_all_training_samples(const string &labels_dir_name, const string &fv_set_dir,
                                     map<uint64, tr1::shared_ptr<Sample> > &all_feature_vectors,
                                     vector<uint64> &pos_instance_id, vector<uint64> &neg_instance_id)
{
  string full_name = labels_dir_name + "/positive.txt";
  ifstream infile(full_name.c_str());


  while (infile.good())
  {
    uint64 inst_id;
    infile >> inst_id;
    if (!infile.good())
    {
      break;
    }

    pos_instance_id.push_back(inst_id);
  }

  infile.close();

  full_name = labels_dir_name + "/negative.txt";
  infile.open(full_name.c_str());
  while (infile.good())
  {
    uint64 inst_id;
    infile >> inst_id;
    if (!infile.good())
    {
      break;
    }
    neg_instance_id.push_back(inst_id);
  }

  infile.close();

  int pos_inst_num = pos_instance_id.size(), neg_inst_num = neg_instance_id.size();
  random_shuffle(pos_instance_id.begin(), pos_instance_id.end());
  random_shuffle(neg_instance_id.begin(), neg_instance_id.end());
  if ((double)pos_inst_num / neg_inst_num > 1.2)
  {
    vector<uint64>::iterator start = pos_instance_id.begin() + (int)(neg_inst_num * 1.2);
    pos_instance_id.erase(start, pos_instance_id.end());
  }
  else if ((double)pos_inst_num / neg_inst_num < 1.0 / 1.2)
  {
    vector<uint64>::iterator start = neg_instance_id.begin() + (int)(pos_inst_num * 1.2);
    neg_instance_id.erase(start, neg_instance_id.end());
  }

  for (vector<uint64>::iterator it = pos_instance_id.begin();
      it != pos_instance_id.end();)
  {
    stringstream ss;
    ss << fv_set_dir;
    ss << "/";
    ss.width(5);
    ss.fill('0');
    ss << *it << ".txt";

    ifstream fv_file(ss.str().c_str());
    if (!fv_file.good())
    {
      //cerr << "File " << ss.str() << " not good. Next one" << endl;
      it = pos_instance_id.erase(it);
      continue;
    }

    int fv_num = 0;
    while (fv_file.good())
    {
      tr1::shared_ptr<Sample> fv(new Sample);

      fv_file >> *fv;
      if (!fv_file.good())
      {
        break;
      }
      fv_num++;

      all_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->feature_vector_id(), fv));
      fv->set_label(1.0);
    }
    if (fv_num != 288)
    {
      cerr << "Feature File not compleete: " << ss.str() << endl;
    }
    it++;
  }

  for (vector<uint64>::iterator it = neg_instance_id.begin();
      it != neg_instance_id.end();)
  {
    stringstream ss;
    ss << fv_set_dir;
    ss << "/";
    ss.width(5);
    ss.fill('0');
    ss << *it << ".txt";

    ifstream fv_file(ss.str().c_str());
    if (!fv_file.good())
    {
      //cerr << "File " << ss.str() << " not good. Next one" << endl;
      it = neg_instance_id.erase(it);
      continue;
    }

    int fv_num = 0;
    while (fv_file.good())
    {
      tr1::shared_ptr<Sample> fv(new Sample);

      fv_file >> *fv;
      if (!fv_file.good())
      {
        break;
      }

      fv_num++;

      all_feature_vectors.insert(pair<uint64, tr1::shared_ptr<Sample> >(fv->feature_vector_id(), fv));
      fv->set_label(-1.0);
    }

    if (fv_num != 288)
    {
      cerr << "Feature File not compleete: " << ss.str() << endl;
    }
    it++;
  }

  cout << "Available&Effective Test Positive Feature Set Files: " << pos_instance_id.size() << endl;

  cout << "Available&Effective Test Negative Feature Set Files: " << neg_instance_id.size() << endl;


  return 0;
}


extern string ref_person_names[60];


int main(int argc, char ** argv)
{

  int option = 3;

//while(false) {
  ifstream names_mapping_file("/home/harvey/Dataset/pubfig-mapping.txt");

  while (!names_mapping_file.eof())
  {
    string ids;
    string name;

    string line;
    getline(names_mapping_file, line);
    if (!names_mapping_file.good())
    {
      break;
    }

    stringstream ss(line);
    getline(ss, name, '\t');
    getline(ss, ids, '\t');

    int id = 0;
    sscanf(ids.c_str(), "%d", &id);
    pubfig_id_name_mapping[static_cast<uint64>(id)] = name;
    pubfig_name_id_mapping[name] = static_cast<uint64>(id);
  }

  load_pubfig_verification_benchmark("/home/harvey/Dataset/pubfig_full.txt");

  // Produce trait vectors for verifiers
  if (option == 2)
  {

//    reload_all_models();
    produce_trait_vectors();
    return 0;
  }

if (option == 1)
{

//  while (false)
//  {
  for (int ref_person_id = 0; ref_person_id < REF_PERSON_NUM; ref_person_id++)
  {

    for (int region_id = REGION_START; region_id < REGION_NUM; region_id++)
    {



     map<uint64, tr1::shared_ptr<Sample> > all_feature_vectors;
      vector<uint64> pos_instance_id, neg_instance_id;

      load_all_pubfig_training_samples(FLAGS_training_simile_labels_dir + "/" + ref_person_names[ref_person_id] + "/", FLAGS_feature_set_dir, all_feature_vectors, pos_instance_id, neg_instance_id);

      simile_classifiers[ref_person_id][region_id].set_ref_person_id_and_region_id(ref_person_id, region_id);
      simile_classifiers[ref_person_id][region_id].assign_training_data(all_feature_vectors, all_feature_vectors);
      simile_classifiers[ref_person_id][region_id].assign_total_training_instance_id(pos_instance_id, neg_instance_id);


      for (int nor_type = NOR_TYPE_START; nor_type < NOR_TYPE_NUM; nor_type++)
      {
        for (int pixel_type = PIXEL_TYPE_START; pixel_type < PIXEL_TYPE_NUM; pixel_type++)
        {
          for (int agg_type = AGG_TYPE_START; agg_type < AGG_TYPE_NUM; agg_type++)
          {
//        	  for (uint64 lambda_type = 0; lambda_type < LAMBDA_OPTIONS; lambda_type++)
//        	  {
//              uint64 feature_type = map_feature_type_id(nor_type, pixel_type, region_id, agg_type) | static_cast<uint64>(lambda_type) << 16;
//              tr1::shared_ptr<WeakClassifier> wc(new SgdSvm(feature_type));
//              simile_classifiers[ref_person_id][region_id].create_one_weak_classifier(wc);
//        	  }
              uint64 feature_type = map_feature_type_id(nor_type, pixel_type, region_id, agg_type);
              tr1::shared_ptr<WeakClassifier> wc(new QpSvm(feature_type));
              simile_classifiers[ref_person_id][region_id].create_one_weak_classifier(wc);
          }
        }
      }

//      simile_classifiers[ref_person_id][region_id].cross_validate();
//
//      simile_classifiers[ref_person_id][region_id].reset();

      simile_classifiers[ref_person_id][region_id].assign_this_time_training_instance_id(pos_instance_id, neg_instance_id, vector<uint64>(), vector<uint64>());

      simile_classifiers[ref_person_id][region_id].learn(tr1::shared_ptr<Sample>());
      simile_classifiers[ref_person_id][region_id].detach_training_data();

      string model_name = FLAGS_simile_classifier_model_dir + "/" + ref_person_names[ref_person_id] + "-" + region_name[region_id] + ".txt";

      ofstream omodel_file(model_name.c_str());

      omodel_file << simile_classifiers[ref_person_id][region_id];


      simile_classifiers[ref_person_id][region_id].reset();
    }

    return 0;
  }
//  }
//}





  for (int attr = 0; attr < ATTR_NUM; attr++)
  {

//    cout << "Adaboost for " << attribute_names[attr] << endl;
    /*
     * Load all feature vectors for this strong classifier all_feature_vectors
     *
     * pos_img, neg_img
     */
    map<uint64, tr1::shared_ptr<Sample> > all_feature_vectors;
    vector<uint64> pos_instance_id, neg_instance_id;

    load_all_training_samples(FLAGS_training_attr_labels_dir + "/" +attribute_names[attr], FLAGS_feature_set_dir, all_feature_vectors, pos_instance_id, neg_instance_id);

//    attr_classifiers[attr].load_all_training_samples(FLAGS_training_attr_labels_dir, FLAGS_feature_set_dir);
    attr_classifiers[attr].assign_training_data(all_feature_vectors, all_feature_vectors);
    attr_classifiers[attr].assign_total_training_instance_id(pos_instance_id, neg_instance_id);

    for (int i = NOR_TYPE_START; i < NOR_TYPE_NUM; i++)
    {
      for (int j = PIXEL_TYPE_START; j < PIXEL_TYPE_NUM; j++)
      {
        for (int k = REGION_START; k < REGION_NUM; k++)
        {
          for (int l = AGG_TYPE_START; l < AGG_TYPE_NUM; l++)
          {
//            for (uint64 lambda_type = 0; lambda_type < LAMBDA_OPTIONS; lambda_type++)
//            {
//              uint64 feature_type = map_feature_type_id(i, j, k, l) | static_cast<uint64>(lambda_type) << 16;
//              tr1::shared_ptr<WeakClassifier> wc(new SgdSvm(feature_type));
//              attr_classifiers[attr].create_one_weak_classifier(wc);
//            }
              uint64 feature_type = map_feature_type_id(i, j, k, l);
              tr1::shared_ptr<WeakClassifier> wc(new QpSvm(feature_type));
              attr_classifiers[attr].create_one_weak_classifier(wc);
          }
        }
      }
    }



//    attr_classifiers[attr].cross_validate();
//
//
//    attr_classifiers[attr].reset();

    attr_classifiers[attr].assign_this_time_training_instance_id(pos_instance_id, neg_instance_id, vector<uint64>(), vector<uint64>());
    attr_classifiers[attr].learn(tr1::shared_ptr<Sample>());

    attr_classifiers[attr].detach_training_data();

    string model_file_name = FLAGS_attr_classifier_model_dir + "/" + attribute_names[attr] + ".txt";
    ofstream model(model_file_name.c_str());

    model << attr_classifiers[attr];

    attr_classifiers[attr].reset();

  }
}

if (option == 3)
{
//    simile_verifier.load_verification_benchmark_feature_vectors(FLAGS_verifier_feature_set_dir + "/simile-verifier-scaled.txt");
//    simile_verifier.produce_cross_validation_files();
	simile_verifier.load_verification_benchmark_feature_vectors(FLAGS_verifier_feature_set_dir + "/simile-verifier-scaled.txt");
	simile_verifier.cross_validate();
	simile_verifier.detach_training_data();

	attr_verifier.load_verification_benchmark_feature_vectors(FLAGS_verifier_feature_set_dir + "/attr-verifier-scaled.txt");
	attr_verifier.cross_validate();
	attr_verifier.detach_training_data();

	combined_verifier.load_verification_benchmark_feature_vectors(FLAGS_verifier_feature_set_dir + "/combined-verifier-scaled.txt");
	combined_verifier.cross_validate();
	combined_verifier.detach_training_data();
}

}



int LoadSampleFile(const string &file_name, int max_rows, vector<std::tr1::shared_ptr<Sample> > &sample_set, int &max_dim)
{
	cout << "Loading sample file: " << file_name << endl;
	ifstream sample_file(file_name.c_str());

	max_dim = 0;
	if (!sample_file.good())
	    assertfail("This file is not good.");

	  int ncount = 0;
	  int pcount = 0;
	  int unknown = 0;
	  while (sample_file.good() && max_rows--)
	  {
          tr1::shared_ptr<Sample> fv(new Sample);

          sample_file >> *fv;
	      if (!sample_file.good())
	      {
	  	    break;
	      }

	      if (fv->label() == 1.0) {
	    	  pcount++;
	      } else if (fv->label() == -1.0) {
	    	  ncount++;
	      } else {
	    	  unknown++;
	      }

	    if (fv->x()->size() > max_dim) {
	    	max_dim = fv->x()->size();
	    }
	    sample_set.push_back(fv);
    }

	  cout << "    Max feature vector dimension: " << max_dim <<endl;
    cout << "    Positve samples: " << pcount << endl;
    cout << "    Negative samples: " << ncount << endl;
    cout << "    Unknown samples: " << unknown << endl;
    cout << "    Total samples: " << pcount + ncount + unknown << endl;
    return 0;
}

//int main(int argc, const char **argv) {
//    SgdSvm svm(-1);
//    vector<tr1::shared_ptr<Sample> > all_training_samples, all_test_samples;
//    int max_dim = 0;
//    LoadSampleFile("/home/harvey/factory/sgd/sgd-2.0/data/pascal/alpha03.txt", -1, all_training_samples, max_dim);
//    LoadSampleFile("/home/harvey/factory/sgd/sgd-2.0/data/pascal/alpha04.txt", -1, all_test_samples, max_dim);
//    svm.set_lambda(0.0001);
//    svm.set_batch_size(1);
//    svm.set_num_iter_to_average(100);
////    svm.set_T(20000);
//
////    svm.learn(all_training_samples);
////
//
////	double t_err = 0, fp_err = 0, fn_err = 0;
////	vector<bool> test_result;
////	svm.test(all_test_samples, test_result, t_err, fp_err, fn_err);
//    for (int i = 1; i <= 5; i++) {
//    	svm.set_T(i * 100000);
//        svm.reset();
//        cout << endl << "Experiment " << i << " ********************************* " << endl;
//        svm.learn(all_training_samples);
//
//
//        double t_err = 0, fp_err = 0, fn_err = 0;
//        vector<bool> test_result;
//        svm.test(all_test_samples, test_result, t_err, fp_err, fn_err);
//
//    }
//}


