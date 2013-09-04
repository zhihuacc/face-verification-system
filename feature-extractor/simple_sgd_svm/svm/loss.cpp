/*
 * loss.cpp
 *
 *  Created on: 29 Apr, 2013
 *      Author: harvey
 */

#include "loss.h"

/*
 * a = w * x + b
 */
double HingeLoss::loss(double a, double y) {
	double z = a * y;
	// z > 1
  if (z >= 1) {
    return 0;
  } else {
    return 1 - z;
  }
}

double HingeLoss::dloss(double a, double y) {
	double z = a * y;
	// z > 1
  if (z >= 1) {
    return 0;
  } else {
    return -y;
  }
}
