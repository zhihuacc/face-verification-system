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
string nor_type_name[NOR_TYPE_NUM] = {"none", "eqhist"};

ifstream fv_set_files[NOR_TYPE_NUM][PIXEL_TYPE_NUM][REGION_NUM][AGG_TYPE_NUM];

//string attribute_names[ATTR_NUM] = {
//    "male",
//    "white",
//    "asian",
//    "black",
//    "black-hair",
//    "blond-hair",
//    "brown-hair",
//    "eyeglass",
//    "indian",
//    "sunglass"
//};

string attribute_names[ATTR_NUM] = {
    "asian",
    "black",
    "black hair",
    "blond hair",
    "bold",
    "brown hair",
    "child",
    "colorful",
    "environment",
    "eyeglasses",
    "eye open",
    "eye wear",
    "indian",
    "jaw bones",
    "male",
    "middle aged",
    "mouth closed",
    "mouth open",
    "nose mouth line",
    "senior",
    "sharp jaw",
    "similing",
    "straight hair",
    "sunglasses",
    "wearing hat",
    "white",
    "youth"
};



string ref_person_names[REF_PERSON_NUM] =
{
    "Abhishek Bachan",
    "Alex Rodriguez",
    "Ali Landry",
    "Alyssa Milano",
    "Anderson Cooper",
    "Anna Paquin",
    "Audrey Tautou",
    "Barack Obama",
    "Ben Stiller",
    "Christina Ricci",
    "Clive Owen",
    "Cristiano Ronaldo",
    "Daniel Craig",
    "Danny Devito",
    "David Duchovny",
    "Denise Richards",
    "Diane Sawyer",
    "Donald Faison",
    "Ehud Olmert",
    "Faith Hill",
    "Famke Janssen",
    "Hugh Jackman",
    "Hugh Laurie",
    "James Spader",
    "Jared Leto",
    "Julia Roberts",
    "Julia Stiles",
    "Karl Rove",
    "Katherine Heigl",
    "Kevin Bacon",
    "Kiefer Sutherland",
    "Kim Basinger",
    "Mark Ruffalo",
    "Meg Ryan",
    "Michelle Trachtenberg",
    "Michelle Wie",
    "Mickey Rourke",
    "Miley Cyrus",
    "Milla Jovovich",
    "Nicole Richie",
    "Rachael Ray",
    "Robert Gates",
    "Ryan Seacrest",
    "Sania Mirza",
    "Sarah Chalke",
    "Sarah Palin",
    "Scarlett Johansson",
    "Seth Rogen",
    "Shahrukh Khan",
    "Shakira",
    "Stephen Colbert",
    "Stephen Fry",
    "Steve Carell",
    "Steve Martin",
    "Tracy Morgan",
    "Ty Pennington",
    "Viggo Mortensen",
    "Wilmer Valderrama",
    "Zac Efron",
    "Zach Braff"
};

string map_feature_vector_name(uint64 img_nid, uint64 nor_type, uint64 pixel_type, uint64 region, uint64 agg_type)
{
  stringstream ss;
  ss.fill('0');
  ss.width(5);
  ss << img_nid;
  return ss.str() + "-" + map_feature_type_name(nor_type, pixel_type, region, agg_type);
}

string map_feature_type_name(uint64 feature_type)
{
  feature_type &= 0xFFFF;
  int nor_type = (feature_type >> 12) & 0xF, pixel_type = (feature_type >> 8) & 0xF,
      region = (feature_type >> 4) & 0xF, agg_type = feature_type & 0xF;
  return map_feature_type_name(nor_type, pixel_type, region, agg_type);
}

string map_feature_type_name(uint64 nor_type, uint64 pixel_type, uint64 region, uint64 agg_type)
{
  return nor_type_name[nor_type] + "-" + pixel_type_name[pixel_type] + "-" + region_name[region] + "-" + agg_type_name[agg_type];
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
          string fn = FLAGS_feature_set_dir + "/";
          fn += map_feature_type_name(i, j, k, l);
          fn += ".txt";
          fv_set_files[i][j][k][l].open(fn.c_str());
        }
      }
    }
  }
}

