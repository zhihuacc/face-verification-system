/*
 * sample.cpp
 *
 *  Created on: 27 Apr, 2013
 *      Author: harvey
 */



#include "sample.h"
#include "assert.h"
#include <cstdio>

Sample::Sample():m_label(0), m_instance_id(-1), m_feature_type_id(-1) {
	m_x.reset(new DensityVector(8));
}

uint64 Sample::label() {
	return m_label;
}

istream &operator>>(istream &f, Sample &fv) {
  string label_str;
  f >> std::skipws >> label_str;

  sscanf(label_str.c_str(), "%lu", &fv.m_label);
//  if (f.fail()) {
//	  assertfail("This feature vector is in wrong format.");
//  }
  f >> *fv.m_x;
  return f;
}

ostream &operator<<(ostream &f, Sample &fv)
{
  char buf[128];
  sprintf(buf, "%lu", fv.label());

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


