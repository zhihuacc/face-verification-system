/*
 * sgd_svm.cpp
 *
 *  Created on: 27 Apr, 2013
 *      Author: harvey
 */

#include "sgd_svm.h"
#include "loss.h"
#include "assert.h"
#include "my_timer.h"

#include <fstream>
#include <algorithm>

#include <set>

SgdSvm::SgdSvm(uint64 feature_type, double lambda, int t, int batch_size, int num_iter_to_avg):
WeakClassifier(feature_type), m_lambda(lambda), m_T(t), m_batch_size(batch_size),
m_num_iter_to_average(num_iter_to_avg), m_learned_bias(0), m_learned_average_bias(0), m_last_test_error_rate(1)
{
  m_num_iter_to_average = min(m_T, m_num_iter_to_average);
}

SgdSvm::SgdSvm(uint64 feature_type): WeakClassifier(feature_type), m_lambda(0.0001), m_T(0), m_batch_size(1),
m_num_iter_to_average(1), m_learned_bias(0), m_learned_average_bias(0), m_last_test_error_rate(1)
{
}

SgdSvm::SgdSvm(): WeakClassifier(0xFFFFFFFF), m_lambda(0.0001), m_T(0), m_batch_size(1),
m_num_iter_to_average(1), m_learned_bias(0), m_learned_average_bias(0), m_last_test_error_rate(1)
{
}

//int SgdSvm::LoadTrainingSampleFile(const string &file_name, int max_rows) {
//	return LoadSampleFile(file_name, max_rows, m_training_sample_set, m_max_feature_vector_dim);
//}
//
//int SgdSvm::LoadTestSampleFile(const string &file_name, int max_rows) {
//	return LoadSampleFile(file_name, max_rows, m_test_sample_set, m_max_feature_vector_dim);
//}
//
//int SgdSvm::load_training_samples(vector<std::tr1::shared_ptr<Sample> > &samples)
//{
//  m_training_sample_set.assign(samples.begin(), samples.end());
//  return 0;
//}
//
//int SgdSvm::load_test_samples(vector<std::tr1::shared_ptr<Sample> > &samples)
//{
//  m_test_sample_set.assign(samples.begin(), samples.end());
//  return 0;
//}


//int SgdSvm::LoadSampleFile(const string &file_name, int max_rows, vector<std::tr1::shared_ptr<Sample> > &sample_set, int &max_dim)
//{
//	cout << "Loading sample file: " << file_name << endl;
//	ifstream sample_file(file_name.c_str());
//
//	max_dim = 0;
//	if (!sample_file.good())
//	    assertfail("This file is not good.");
//
//	  int ncount = 0;
//	  int pcount = 0;
//	  int unknown = 0;
//	  while (sample_file.good() && max_rows--)
//	  {
//          tr1::shared_ptr<Sample> fv(new Sample);
//
//          sample_file >> *fv;
//	      if (!sample_file.good())
//	      {
//	  	    break;
//	      }
//
//	      if (fv->label() == +1.0) {
//	    	  pcount++;
//	      } else if (fv->label() == -1.0) {
//	    	  ncount++;
//	      } else {
//	    	  unknown++;
//	  }
//
//	    if (fv->x()->size() > max_dim) {
//	    	max_dim = fv->x()->size();
//	    }
//	    sample_set.push_back(fv);
//    }
//
//	  cout << "    Max feature vector dimension: " << max_dim <<endl;
//    cout << "    Positve samples: " << pcount << endl;
//    cout << "    Negative samples: " << ncount << endl;
//    cout << "    Unknown samples: " << unknown << endl;
//    cout << "    Total samples: " << pcount + ncount + unknown << endl;
//    return 0;
//}

void SgdSvm::set_T(int t) {
	m_T = t;
}
void SgdSvm::set_lambda(double lambda) {
	m_lambda = lambda;
}

void SgdSvm::set_batch_size(int n)
{
  m_batch_size = n;
}

void SgdSvm::set_num_iter_to_average(int n)
{
  m_num_iter_to_average = n;
}

//void SgdSvm::set_max_feature_vector_dim(int dim)
//{
//  m_max_feature_vector_dim = dim;
//}

void SgdSvm::reset() {
    m_learned_weights.reset();
    m_learned_bias = 0;
    m_learned_average_weights.reset();
    m_learned_average_bias = 0;
}

void SgdSvm::PrintLearningParameters() {
    cout << "Learning Parameters:" << endl;
    cout << "    Lamda: " << m_lambda << endl;
    cout << "    Iterations: " << m_T << endl;
    cout << "    Num iter to average: " << m_num_iter_to_average << endl;
    cout << "    Batch size: " << m_batch_size << endl;
}

int SgdSvm::learn(tr1::shared_ptr<Sample> training_sample)
{
  return -1;
}

int SgdSvm::learn(const vector<tr1::shared_ptr<Sample> > &training_samples) {

  double etan = 0;
  MyTimer timer;
  timer.start();

  PrintLearningParameters();

  if (training_samples.size() == 0)
  {
    return -1;
  }

//  for (int i = 0; i < (int)training_samples.size(); i++)
//  {
//    training_samples[i]->norm();
//  }

	m_learned_weights.reset();
	m_learned_bias = 0;
	m_learned_average_weights.reset();
	m_learned_average_bias = 0;
  std::set<int> random_number;

  srandom(time(NULL));
  for (int t = 1; t <= m_T; t++) {

    etan = 1.0 / (m_lambda * t);

    /*
     * Randomly select a subset of training set At, where |At| = batch_size
     */
    double dloss_sum = 0;
    DensityVector gradient_of_loss(0);

    srandom(time(NULL) + random());
    for (int r = 0; r < m_batch_size; r++) {
      int rn = random() % training_samples.size();
      std::tr1::shared_ptr<Sample> one_training_sample = training_samples.at(rn);
      random_number.insert(rn);

      double wxb = m_learned_weights.dot(*(one_training_sample->x())) + m_learned_bias;

      double dloss = HingeLoss::dloss(wxb, one_training_sample->label());
      gradient_of_loss.add(*(one_training_sample->x()), dloss);
      dloss_sum += dloss;
    }


    double avg_loss_scale = 1.0 / m_batch_size;

    m_learned_weights.mul(1.0 - etan * m_lambda);
    gradient_of_loss.mul(avg_loss_scale);
    m_learned_weights.add(gradient_of_loss, -etan);

    m_learned_bias += -etan * (dloss_sum * avg_loss_scale);

    if (m_T <= m_num_iter_to_average + t - 1)
    {
      double avg_scale = 1.0 / m_num_iter_to_average;
      m_learned_average_weights.add(m_learned_weights, avg_scale);
      m_learned_average_bias += (m_learned_bias * avg_scale);
    }

  }
  timer.stop();
  struct timespec diff = timer.diff();
  cout << "Learned Results: " << endl;
  cout << "    Norm of weights: " << m_learned_average_weights.dot(m_learned_average_weights) << endl;
  cout << "    Bias: " << m_learned_average_bias << endl;
  cout << "    Time consumed: " << diff.tv_sec << "sec, " << diff.tv_nsec << "nsec" << endl;
  cout << "    Coverage: " << (double)random_number.size() / training_samples.size() << endl;
  return 0;
}

int SgdSvm::test(tr1::shared_ptr<Sample> x, double &prediction)
{
  prediction = m_learned_average_weights.dot(*(x->x())) + m_learned_average_bias;
  return 0;
}

int SgdSvm::test(const vector<tr1::shared_ptr<Sample> > &test_samples, vector<bool> &test_result,
    double &total_error_rate, double &false_pos_rate, double &false_neg_rate,
    double &true_pos_rate, double &true_neg_rate)
{
	double loss = 0;
	int    misclassification = 0;
	int    false_pos = 0;
	int    true_pos = 0;
	int    false_neg = 0;
	int    true_neg = 0;
	int    pos_num = 0;
	int    neg_num = 0;

	if (test_samples.size() == 0)
	{
	  return -1;
	}

	m_last_test_result.clear();
  for (vector<std::tr1::shared_ptr<Sample> >::const_iterator it = test_samples.begin();
      it != test_samples.end(); it++) {

      std::tr1::shared_ptr<Sample> ps = *it;

      double wxb = m_learned_average_weights.dot(*(ps->x())) + m_learned_average_bias;

      double y = ps->label();
      loss += HingeLoss::loss(wxb, ps->label());

      if (y > 0)
      {
        pos_num++;
      }
      else
      {
        neg_num++;
      }

      double threshold = 0;
      if (y > 0 && wxb <= threshold)
      {
        false_neg++;
      }
      else if (y < 0 && wxb >= threshold)
      {
        false_pos++;
      }
      else if (y > 0 && wxb >= threshold)
      {
        true_pos++;
      }
      else if (y < 0 && wxb <= threshold)
      {
        true_neg++;
      }

      m_last_test_result.push_back((wxb * y > 0));
  }

  test_result = m_last_test_result;

  misclassification = false_pos + false_neg;
  total_error_rate = (double)misclassification / test_samples.size();
  false_pos_rate = (double)false_pos / neg_num;
  false_neg_rate = (double)false_neg / pos_num;


  true_pos_rate = (double)true_pos / pos_num;
  true_neg_rate = (double)true_neg / neg_num;

  cout << "Test Result: " << endl;
  cout << "    Test set size: " << test_samples.size() << endl;
  cout << "    Total loss: " << loss << endl;
  cout << "    Average loss: " << loss / test_samples.size() << endl;
  cout << "    Total misclassification: " << misclassification << endl;
  cout << "    False Positive Rate: " << false_pos_rate << endl;
  cout << "    False Negative Rate: " << false_neg_rate << endl;
  cout << "    True Positive Rate: " << true_pos_rate << endl;
  cout << "    True Negative Rate: " << true_neg_rate << endl;
  cout << "    Total error rate: " << total_error_rate << endl;
  return 0;
}

int SgdSvm::weighted_error(double &error)
{
  error = m_weighted_error;
  return 0;
}

int SgdSvm::calc_weighted_error(const vector<double> &weights, double &weighted_error)
{
  m_weighted_error = 1;
//  m_last_test_error_rate = 1;

  if (m_last_test_result.size() != weights.size())
  {
    return -1;
  }

  m_weighted_error = 0;
//  m_last_test_error_rate = 0;
  for (int i = 0; i < (int)m_last_test_result.size(); i++)
  {
    if (!m_last_test_result[i])
    {
      m_weighted_error += weights[i];
    }
  }

//  m_error_rate = m_svm->get_test_error_rate();
  weighted_error = m_weighted_error;
  return 0;
}

int SgdSvm::detach_temp_variables()
{
  m_last_test_result.clear();
  vector<bool>().swap(m_last_test_result);
  return 0;
}

int SgdSvm::shuffle() {
	//random_shuffle(m_training_sample_set.begin(), m_training_sample_set.end());
	return 0;
}



int SgdSvm::set_alpha(double a)
{
  m_alpha = a;
  return 0;
}

int SgdSvm::alpha(double &alpha)
{
  alpha = m_alpha;
  return 0;
}

int SgdSvm::set_feature_type(uint64 feature_type)
{
  m_feature_type_id = feature_type;
  return 0;
}

int SgdSvm::feature_type(uint64 &feature_type)
{
  feature_type = m_feature_type_id;
  return 0;
}

int SgdSvm::get_last_test_result(vector<bool> &last_test_result)
{
  last_test_result = m_last_test_result;
  return 0;
}

ostream &operator<<(ostream &f, SgdSvm &svm)
{
  if (!f.good())
  {
    return f;
  }

  char buf[128];

  sprintf(buf, "%lf", svm.m_learned_average_bias);
  f << string(buf) << endl;

  f << svm.m_learned_average_weights;

  return f;
}

istream &operator>>(istream &f, SgdSvm &svm)
{
  if (!f.good())
  {
    return f;
  }

  string str;
  double d = 0.0;

  f >> str;
  sscanf(str.c_str(), "%lf", &d);
  svm.m_learned_average_bias = d;

  int c = f.get();
  while (c == '\n' || c == '\t' || isspace(c))
  {
    c = f.get();
  }

  f.unget();

  f >> svm.m_learned_average_weights;

  return f;
}

//double SgdSvm::get_test_error_rate()
//{
//  return m_test_error_rate;
//}

//void SgdSvm::detach_training_samples()
//{
//  m_training_sample_set.clear();
//  m_positive_sample_num = m_negative_sample_num = 0;
//}
//
//void SgdSvm::detach_test_samples()
//{
//  m_test_sample_set.clear();
//}
//
//void SgdSvm::shrink_test_result_set()
//{
//  m_test_result.clear();
//}
//
//int SgdSvm::get_positive_sample_num()
//{
//  return m_positive_sample_num;
//}
//
//int SgdSvm::get_negative_sample_num()
//{
//  return m_negative_sample_num;
//}

