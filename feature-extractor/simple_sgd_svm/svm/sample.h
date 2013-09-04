/*
 * sample.h
 *
 *  Created on: 27 Apr, 2013
 *      Author: harvey
 */

#ifndef SAMPLE_H_
#define SAMPLE_H_

#include "../../types.h"
#include "density_vector.h"
#include <tr1/memory>


class Sample {
public:
	Sample();
	uint64 label();
	uint64 instance_id();
	uint64 feature_type();
	std::tr1::shared_ptr<DensityVector> x() const;

	void set_label(double label) { m_label = label; }
	void set_instance_id(uint64 id) { m_instance_id = id; }
	void set_feature_type(uint64 ft) { m_feature_type_id = ft; }

	int norm() { return m_x->norm();}

	friend istream &operator>>(istream &f, Sample &v);
	friend ostream &operator<<(ostream &f, Sample &v);
private:

	std::tr1::shared_ptr<DensityVector> m_x;
	uint64  m_label;
	uint64  m_instance_id;
	uint64  m_feature_type_id;
};


#endif
