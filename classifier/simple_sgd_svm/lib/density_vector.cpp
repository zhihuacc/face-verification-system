/*
 * density_vector.cpp
 *
 *  Created on: 27 Apr, 2013
 *      Author: harvey
 */

#include "density_vector.h"
#include "assert.h"
#include <iostream>
#include <cmath>
#include <cstdio>

using namespace std;

DensityVector::DensityVector(int capacity):m_capacity(capacity)
{
  if (m_capacity <= 0) {
    m_values = NULL;
    reset();
  } else {
    m_values = new double[m_capacity];
    m_size = 0;
  }
}
DensityVector::DensityVector() {
	m_values = NULL;
	reset();
}

void DensityVector::reset() {
	if (m_values != NULL) {
		delete [] m_values;
	}

	m_capacity = 8;
	m_values = new double[m_capacity];
	m_size = 0;
}

DensityVector::~DensityVector() {
	delete [] m_values;
}

/*
 * index starts from 1 instead of 0
 */
int DensityVector::setValue(int index, double value)
{
	assert(index > 0);

  if (index > m_capacity) {
    int properSize = m_capacity;
      for (int i = 0; i < (MAX_VECTOR_LENGTH_SHIFT - 3); i++) {
        properSize *= 2;
        if (index <= properSize) {
          break;
        }
      }

      if (index > properSize) {
          assertfail("This feature is too long.");
      }

      double *largerArea = new double[properSize];
      int i = 0;
      for (i = 0; i < m_size; i++) {
        largerArea[i] = m_values[i];
      }
      delete [] m_values;
      m_values = largerArea;

      m_capacity = properSize;
  }

  if (index > m_size) {
    for (int i = m_size; i < index; i++) {
      m_values[i] = 0;
    }
    m_size = index;
  }

  m_values[index - 1] = value;

  return 0;
}

double DensityVector::valueAt(int index) const {

	assert(index > 0 && index <= m_size);
	return m_values[index - 1];
}

int DensityVector::size() const{
	return m_size;
}

ostream &operator<<(ostream &f, DensityVector &v)
{
  for (int i = 1; i <= v.size(); i++)
  {
    char buf[128];

    sprintf(buf, "%lf", v.valueAt(i));

    f << i << ":" << string(buf) << " ";
  }
  return f;
}

istream &operator>>(istream &f, DensityVector &v)
{
  for(;;) {
    int c = f.get();
    if (!f.good() || (c=='\n' || c=='\r'))
    {
      break;
    }

    if (::isspace(c))
    {
      continue;
    }
    else if (c == '#')
    {
      string comment;
      getline(f, comment);
      f.unget();
      continue;
    }


    int i;
    f.unget();
    f >> std::skipws >> i >> std::ws;
    if (i <= 0) {
      assertfail("feature number should be positive.");
    }
    if (f.get() != ':')
    {
//		    f.unget();
//		    f.setstate(std::ios::badbit);
//		    break;
      assertfail("colon \":\" is required.");
    }
    double x;
    string val_str;
    f >> std::skipws >> val_str;
    sscanf(val_str.c_str(), "%lf", &x);
    //f >> std::skipws >> x;
    if (!f.good()) {
      cerr << "Error at feature " << i << ", " << x << endl;
      assertfail("This feature vector is in wrong format.");
//		  break;
    }
    v.setValue(i,x);
  }
  return f;
}

double DensityVector::dot(const DensityVector &v) const{
    int n = min(size(), v.size());
    double sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += (valueAt(i) * v.valueAt(i));
    }
    return sum;
}

int DensityVector::add(const DensityVector &v) {
    int l = min(size(), v.size());

    for (int i = 1; i <= l; i++) {
        setValue(i, valueAt(i) + v.valueAt(i));
    }

    if (v.size() > size()) {
      int h = max(size(), v.size());
      for (int i = l + 1; i <= h; i++) {
        setValue(i, v.valueAt(i));
      }
    }

    return 0;
}

int DensityVector::add(const DensityVector &v, double c) {
    int l = min(size(), v.size());

    for (int i = 1; i <= l; i++) {
        setValue(i, valueAt(i) + v.valueAt(i) * c);
    }

    if (v.size() > size()) {
      int h = max(size(), v.size());
      for (int i = l + 1; i <= h; i++) {
        setValue(i, v.valueAt(i) * c);
      }
    }

    return 0;
}

int DensityVector::sub(const DensityVector &v) {
    int l = min(size(), v.size());

    for (int i = 1; i <= l; i++) {
        setValue(i, valueAt(i) - v.valueAt(i));
    }

    if (v.size() > size()) {
      int h = max(size(), v.size());
      for (int i = l + 1; i <= h; i++) {
        setValue(i, -v.valueAt(i));
      }
    }

    return 0;
}

int DensityVector::mul(double multiplier) {
    int n = size();
    for (int i = 1; i <= n; i++) {
    	setValue(i, multiplier * valueAt(i));
    }

    return 0;
}

int DensityVector::norm()
{
  int n = size();
  double norm2 = sqrt(this->dot(*this));
  if (norm2 == 0)
  {
    return 0;
  }

  for (int i = 1; i <= n; i++)
  {
    setValue(i, valueAt(i) / norm2);
  }

  return 0;
}


