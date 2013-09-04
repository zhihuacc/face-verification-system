/*
 * sample.cpp
 *
 *  Created on: 27 Apr, 2013
 *      Author: harvey
 */



#include "sample.h"
#include "assert.h"
#include <cstdio>

Sample::Sample():m_label(0), m_feature_vector_id(-1) {
	m_x.reset(new DensityVector(8));
}

double Sample::label() {
	return m_label;
}

uint64 Sample::feature_vector_id()
{
  return m_feature_vector_id;
}

istream &operator>>(istream &f, Sample &fv) {
  string label_str;
  f >> std::skipws >> label_str;


  sscanf(label_str.c_str(), "%lu", &fv.m_feature_vector_id);
//  fv.m_label = static_cast<double>(label);
//  if (f.fail()) {
//	  assertfail("This feature vector is in wrong format.");
//  }
  f >> *fv.m_x;
  return f;
}

ostream &operator<<(ostream &f, Sample &fv)
{
  char buf[128];
  sprintf(buf, "%lf", fv.label());

  f << string(buf) << " ";
  for (int index = 1; index <= fv.x()->size(); index++)
  {
    char buf[128];
    sprintf(buf, "%lf", fv.x()->valueAt(index));
    f << index << ":" << string(buf) << " ";
  }
  f << endl;
  return f;
}

std::tr1::shared_ptr<DensityVector> Sample::x() const {
	return m_x;
}


