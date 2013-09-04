/*
 * flags.h
 *
 *  Created on: 23 Jun, 2013
 *      Author: harvey
 */

#ifndef FLAGS_H_
#define FLAGS_H_

#include <gflags/gflags.h>

DECLARE_string(feature_set_dir);
DECLARE_string(training_attr_labels_dir);
DECLARE_string(training_simile_labels_dir);
DECLARE_string(attr_classifier_model_dir);
DECLARE_string(simile_classifier_model_dir);

DECLARE_string(verifier_feature_set_dir);

DECLARE_double(lambda);
DECLARE_double(batch_size_ratio);
DECLARE_int32(T);

#endif /* FLAGS_H_ */
