/*
 * feature_extraction_public.cpp
 *
 *  Created on: 24 Jun, 2013
 *      Author: harvey
 */


#include "feature_extraction_public.h"
#include "flags.h"
#include <sstream>

string pixel_type_name[PIXEL_TYPE_NUM] = {"rgb", "hsv", "intensity", "edgemag", "edgeori", "lbp"};
string region_name[REGION_NUM] = {"eyes", "mouth", "nose", "cheeks", "chin", "eyebrows", "hair", "wface"};
string agg_type_name[AGG_TYPE_NUM] = {"none", "stat", "hist"};
//string nor_type_name[NOR_TYPE_NUM] = {"none", "mean", "energy"};
string nor_type_name[NOR_TYPE_NUM] = {"none", "eqhist"};

ofstream fv_set_files[NOR_TYPE_NUM][PIXEL_TYPE_NUM][REGION_NUM][AGG_TYPE_NUM];

string map_feature_vector_name(uint64 img_nid, uint64 nor_type, uint64 pixel_type, uint64 region, uint64 agg_type)
{
  stringstream ss;
  ss.width(5);
  ss.fill('0');
  ss << img_nid;
  return ss.str() + "-" + map_feature_type_name(nor_type, pixel_type, region, agg_type);
}

string map_feature_type_name(uint64 nor_type, uint64 pixel_type, uint64 region, uint64 agg_type)
{
  return nor_type_name[nor_type] + "-" +  pixel_type_name[pixel_type] + "-" + region_name[region] + "-" + agg_type_name[agg_type];
}

uint64 map_feature_type_id(uint64 nor_type, uint64 pixel_type, uint64 region, uint64 agg_type)
{
  return map_feature_vector_id(0, nor_type, pixel_type, region, agg_type);
}

uint64 map_feature_vector_id(uint64 img_id, uint64 nor_type, uint64 pixel_type, uint64 region, uint64 agg_type)
{
  uint64 id = img_id << 16 | nor_type << 12 | pixel_type << 8 | region << 4 | agg_type;
  return id;
}

void init_fv_set_files()
{
  for (int i = 0; i < NOR_TYPE_NUM; i++)
  {
    for (int j = 0; j < PIXEL_TYPE_NUM; j++)
    {
      for (int k = 0; k < REGION_NUM; k++)
      {
        for (int l = 0; l < AGG_TYPE_NUM; l++)
        {
          string fn = FLAGS_feature_set_dir;
          fn += map_feature_type_name(i, j, k, l);
          fn += ".txt";
          fv_set_files[i][j][k][l].open(fn.c_str());
        }
      }
    }
  }
}

