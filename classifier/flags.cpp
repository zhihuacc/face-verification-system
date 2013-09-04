/*
 * flags.cpp
 *
 *  Created on: 23 Jun, 2013
 *      Author: harvey
 */

#include "flags.h"

DEFINE_string(feature_set_dir, "/var/all-low-level-features-scaled/", "");
DEFINE_string(training_attr_labels_dir, "/home/harvey/Dataset/attr-labels/", "");

DEFINE_string(training_simile_labels_dir, "/home/harvey/Dataset/simile/", "");
DEFINE_string(attr_classifier_model_dir, "/home/harvey/Dataset/qp-model/attr-model/", "");
DEFINE_string(simile_classifier_model_dir, "/home/harvey/Dataset/qp-model/simile-model/", "");

DEFINE_string(verifier_feature_set_dir, "/home/harvey/Dataset/qp-verifier/", "");

//DEFINE_double(lambda, 0.0001, "");
//DEFINE_double(batch_size_ratio, 1.0 / 3000, "");
//DEFINE_int32(T, 3000, "");

