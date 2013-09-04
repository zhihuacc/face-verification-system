/*
 * feature_extraction_public.h
 *
 *  Created on: 24 Jun, 2013
 *      Author: harvey
 */

#ifndef FEATURE_EXTRACTION_PUBLIC_H_
#define FEATURE_EXTRACTION_PUBLIC_H_

#include <string>
#include <fstream>
#include "types.h"

using namespace std;

enum {
  AGG_TYPE_START = 0,
  AGG_TYPE_NONE = 0,
  AGG_TYPE_STAT = 1,
  AGG_TYPE_HIST = 2,
  AGG_TYPE_NUM
};

enum
{
  PIXEL_TYPE_START = 0,
  PIXEL_TYPE_RGB = 0,
  PIXEL_TYPE_HSV,
  PIXEL_TYPE_INTENSITY,
  PIXEL_TYPE_EDGE_MAGNITUDE,
  PIXEL_TYPE_EDGE_ORIENTATION,
  PIXEL_TYPE_LBP,
  PIXEL_TYPE_NUM
};

enum
{
  REGION_START = 0,
  REGION_EYES = 0,
  REGION_MOUTH,
  REGION_NOSE,
  REGION_CHEEKS,
  REGION_CHIN,
  REGION_EYEBROWS,
  REGION_HAIR,
  REGION_WHOLE_FACE,
  REGION_NUM

};

enum
{
  NOR_TYPE_START = 0,
  NOR_TYPE_NONE = 0,
  NOR_TYPE_EQHIST,
  NOR_TYPE_NUM
};

extern string pixel_type_name[PIXEL_TYPE_NUM];
extern string region_name[REGION_NUM];
extern string agg_type_name[AGG_TYPE_NUM];
extern string nor_type_name[NOR_TYPE_NUM];

extern ofstream fv_set_files[NOR_TYPE_NUM][PIXEL_TYPE_NUM][REGION_NUM][AGG_TYPE_NUM];


string map_feature_vector_name(uint64 img_nid, uint64 nor_type, uint64 pixel_type, uint64 region, uint64 agg_type);

string map_feature_type_name(uint64 nor_type, uint64 pixel_type, uint64 region, uint64 agg_type);
uint64 map_feature_type_id(uint64 nor_type, uint64 pixel_type, uint64 region, uint64 agg_type);
uint64 map_feature_vector_id(uint64 img_id, uint64 nor_type, uint64 pixel_type, uint64 region, uint64 agg_type);

void init_fv_set_files();

#endif /* FEATURE_EXTRACTION_PUBLIC_H_ */
