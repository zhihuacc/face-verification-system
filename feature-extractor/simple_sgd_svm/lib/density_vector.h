/*
 * density_vector.h
 *
 *  Created on: 26 Apr, 2013
 *      Author: harvey
 */

#ifndef DENSITY_VECTOR_H_
#define DENSITY_VECTOR_H_

#include <iostream>

using namespace std;


#define MAX_VECTOR_LENGTH_SHIFT 11
#define MAX_VECTOR_LENGTH (1 << MAX_VECTOR_LENGTH_SHIFT)

class DensityVector {
public:
	DensityVector(int capcity);
	DensityVector();
	~DensityVector();
	void reset();
	int setValue(int index, double value);
	double valueAt(int index) const;
	int size() const;
	double dot(const DensityVector &v) const;
	int add(const DensityVector &v);
	int add(const DensityVector &v, double c);
	int sub(const DensityVector &v);
	int mul(double multiplier);
	int norm();

	friend istream &operator>>(istream &f, DensityVector &v);
private:
	int    m_capacity;
	int    m_size;
	double *m_values;
	double m_label;


};

//double dot(const DensityVector &v1, const DensityVector &v2);


#endif /* DENSITY_VECTOR_H_ */
