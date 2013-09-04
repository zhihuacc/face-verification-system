/*
 * verifier_feature_producer.h
 *
 *  Created on: 8 Aug, 2013
 *      Author: harvey
 */

#ifndef VERIFIER_FEATURE_PRODUCER_H_
#define VERIFIER_FEATURE_PRODUCER_H_


#include <fstream>
#include "verifier.h"
#include "feature_extraction_public.h"
#include "sample.h"

#include <vector>
#include <map>
#include <iostream>
#include <sstream>


int load_pubfig_verification_benchmark(const string &benchmark);
int produce_trait_vectors();
int reload_all_models();


#endif /* VERIFIER_FEATURE_PRODUCER_H_ */
