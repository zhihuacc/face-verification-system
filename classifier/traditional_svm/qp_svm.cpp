/*
 * QpSvm.cpp
 *
 *  Created on: 21 Aug, 2013
 *      Author: harvey
 */

#include "qp_svm.h"
#include <cstring>
#include <cmath>

QpSvm::QpSvm(uint64 ft):WeakClassifier(ft), m_model(NULL), m_fv_length(0)
{
  memset((void*)&m_param, 0, sizeof(m_param));
  memset((void*)&m_problem_set, 0, sizeof(m_problem_set));
}

QpSvm::QpSvm(): WeakClassifier(0xFFFFFFFFFFFFFFFF), m_model(NULL), m_fv_length(0)
{
  memset((void*)&m_param, 0, sizeof(m_param));
  memset((void*)&m_problem_set, 0, sizeof(m_problem_set));
}

QpSvm::~QpSvm()
{
  reset();
}

int QpSvm::learn(const vector<tr1::shared_ptr<Sample> > &training_samples)
{


  convert_to_internal_vectors(training_samples, m_problem_set);

  m_model = svm_train(&m_problem_set, &m_param);

  return 0;
}

//int QpSvm::learn(const svm_problem &problem_set)
//{
//  m_model = svm_train(&problem_set, &m_param);
//  return 0;
//}

int QpSvm::convert_to_internal_vectors(const vector<tr1::shared_ptr<Sample> > &training_set, svm_problem &prolem_set)
{
  if (training_set.size() == 0)
  {
    return -1;
  }

  m_fv_length = training_set[0]->x()->size();
  if (m_param.gamma == 0)
  {
    m_param.gamma = 1.0 / m_fv_length;
  }

  prolem_set.l = training_set.size();
  prolem_set.y = new double[prolem_set.l];
  prolem_set.x = new svm_node*[prolem_set.l];

//  m_space = new svm_node[prolem_set.l  * (feature_num + 1)];

  svm_node *x_space = new svm_node[prolem_set.l  * (m_fv_length + 1)];


  int i = 0, j = 0;
  for (vector<tr1::shared_ptr<Sample> >::const_iterator it = training_set.begin();
       it != training_set.end();
       it++, i++, j++)
  {
    tr1::shared_ptr<Sample> ps = *it;

    prolem_set.y[i] = ps->label();
    prolem_set.x[i] = &x_space[j];

    for (int k = 1; k <= ps->x()->size(); k++, j++)
    {
      x_space[j].index = k;
      x_space[j].value = ps->x()->valueAt(k);
    }

    x_space[j].index = -1;
  }

  return 0;
}

int QpSvm::free_problem_set(svm_problem &problem_set)
{
  if (problem_set.y != NULL)
  {
    delete [] problem_set.y;
    problem_set.y = NULL;
  }


  if (problem_set.x != NULL)
  {
    delete [] problem_set.x[0];
    delete [] problem_set.x;
    problem_set.x = NULL;
  }

  problem_set.l = 0;

  return 0;
}

int QpSvm::learn(tr1::shared_ptr<Sample> training_sample)
{
  return -1;
}

int QpSvm::test(const vector<tr1::shared_ptr<Sample> > &test_samples, vector<bool> &test_result,
    double &total_error_rate, double &false_pos_rate, double &false_neg_rate,
    double &true_pos_rate, double &true_neg_rate)
{
  int pos_num = 0, neg_num = 0, false_pos = 0, false_neg = 0, true_pos = 0, true_neg = 0, misclassification = 0;
  int nr = svm_get_nr_class(m_model);
  double *prob_estimates = new double[nr];
  double *dec_values = new double[nr*(nr-1)/2];
  for (vector<tr1::shared_ptr<Sample> >::const_iterator it = test_samples.begin();
       it != test_samples.end();
       it++)
  {
    double y = (*it)->label();

    if (y > 0)
    {
      pos_num++;
    }
    else
    {
      neg_num++;
    }

    svm_node *x = NULL;
    convert_to_internal_vector(*it, &x);

    double predict_label = svm_predict_probability(m_model, x, prob_estimates);
    svm_predict_values(m_model, x, dec_values);
    double prediction = dec_values[0];
    if (predict_label > 0)
    {
      prediction = abs(prediction);
    }
    else
    {
      prediction = -abs(prediction);
    }

//    double prediction = svm_predict(m_model, x);
    double threshold = -3.0;
    if (y > 0 && prediction <= threshold)
    {
      false_neg++;
    }
    else if (y < 0 && prediction >= threshold)
    {
      false_pos++;
    }
    else if (y > 0 && prediction >= threshold)
    {
      true_pos++;
    }
    else if (y < 0 && prediction <= threshold)
    {
      true_neg++;
    }

    m_last_test_result.push_back((prediction * y > 0));

    free_internal_vector(x);
  }

  delete [] prob_estimates;
  delete [] dec_values;

  test_result = m_last_test_result;

  misclassification = false_pos + false_neg;
  total_error_rate = (double)misclassification / test_samples.size();
  false_pos_rate = (double)false_pos / neg_num;
  false_neg_rate = (double)false_neg / pos_num;


  true_pos_rate = (double)true_pos / pos_num;
  true_neg_rate = (double)true_neg / neg_num;

  cout << "Test Result: " << endl;
  cout << "    Test set size: " << test_samples.size() << endl;
  cout << "    Total misclassification: " << misclassification << endl;
  cout << "    False Positive Rate: " << false_pos_rate << endl;
  cout << "    False Negative Rate: " << false_neg_rate << endl;
  cout << "    True Positive Rate: " << true_pos_rate << endl;
  cout << "    True Negative Rate: " << true_neg_rate << endl;
  cout << "    Total error rate: " << total_error_rate << endl;
  return 0;
}

int QpSvm::convert_to_internal_vector(tr1::shared_ptr<Sample> this_sample, svm_node **internal_vector)
{
  int size = this_sample->x()->size();
  if (size == 0)
  {
    return -1;
  }

  svm_node *nodes = new svm_node[size + 1];
  for (int i = 1; i <= size; i++)
  {
    nodes[i - 1].index = i;
    nodes[i - 1].value = this_sample->x()->valueAt(i);
  }
  nodes[size].index = -1;

  *internal_vector = nodes;
  return 0;
}

int QpSvm::free_internal_vector(svm_node *v)
{
  if (v == NULL)
  {
    return -1;
  }

  delete [] v;

  return 0;
}

int QpSvm::test(tr1::shared_ptr<Sample> test_sample, double &prediction)
{
  svm_node *x = NULL;
  convert_to_internal_vector(test_sample, &x);

  int nr = svm_get_nr_class(m_model);
  double *dec_values = new double[nr*(nr-1)/2];

  svm_predict_values(m_model, x, dec_values);

  prediction = dec_values[0];

  free_internal_vector(x);
  delete [] dec_values;

  return 0;
}

void QpSvm::reset()
{
  svm_free_and_destroy_model(&m_model);
  svm_destroy_param(&m_param);
  free_problem_set(m_problem_set);
}

int QpSvm::detach_temp_variables()
{
  m_last_test_result.clear();
  vector<bool>().swap(m_last_test_result);
  return 0;
}

int QpSvm::set_alpha(double a)
{
  m_alpha = a;
  return 0;
}

int QpSvm::alpha(double &alpha)
{
  alpha = m_alpha;
  return 0;
}

int QpSvm::set_feature_type(uint64 feature_type)
{
  m_feature_type_id = feature_type;
  return 0;
}

int QpSvm::feature_type(uint64 &feature_type)
{
  feature_type = m_feature_type_id;
  return 0;
}

int QpSvm::get_last_test_result(vector<bool> &last_test_result)
{
  last_test_result = m_last_test_result;
  return 0;
}

int QpSvm::weighted_error(double &error)
{
  error = m_weighted_error;
  return 0;
}

int QpSvm::calc_weighted_error(const vector<double> &weights, double &weighted_error)
{
  m_weighted_error = 1;

  if (m_last_test_result.size() != weights.size())
  {
    return -1;
  }

  m_weighted_error = 0;
  for (int i = 0; i < (int)m_last_test_result.size(); i++)
  {
    if (!m_last_test_result[i])
    {
      m_weighted_error += weights[i];
    }
  }

  weighted_error = m_weighted_error;
  return 0;
}



