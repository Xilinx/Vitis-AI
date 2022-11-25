/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <xir/graph/graph.hpp>
#include <xir/graph/subgraph.hpp>
#include <xir/util/tool_function.hpp>
#include "tests.hpp"

jsonOrXirKeys::jsonOrXirKeys(std::string runner_dir)
{
  if(runner_dir.find(".json") != std::string::npos)
  { 
    runner_dir_ = runner_dir.substr(0, runner_dir.size()-9);
    std::ifstream f(runner_dir);
    std::stringstream metabuf;
    metabuf << f.rdbuf();
    json_object *jobj = json_tokener_parse(metabuf.str().c_str());     
    enable_xmodel_format_ = getBool("usexmodel", jobj);
    if(enable_xmodel_format_)
    {
      std::string xmodel_filename = runner_dir_+getFileNameIfExists("xmodelFile",jobj);
      loadFromXmodel(xmodel_filename);
    }
  else
    loadFromJson(jobj);
  }
  else
  {
    loadFromXmodel(runner_dir);
  }
}

uint32_t jsonOrXirKeys::getOutW(){return outW_;}
uint32_t jsonOrXirKeys::getOutH(){return outH_;}
uint32_t jsonOrXirKeys::getOutCh(){return outCh_;}
bool jsonOrXirKeys::getDebugMode(){return debugMode_;}
std::string jsonOrXirKeys::getGoldenFilename(){return golden_filename_;}
std::string jsonOrXirKeys::getSynsetFilename(){return synset_filename_;}
uint32_t jsonOrXirKeys::getInW(){return inW_;}
uint32_t jsonOrXirKeys::getInH(){return inH_;}
uint32_t jsonOrXirKeys::getInCh(){return inCh_;}

void jsonOrXirKeys::loadFromJson(json_object* jobj)
{
  debugMode_ = getBool("debugMode", jobj);
  golden_filename_ = runner_dir_+"gold.txt";
  synset_filename_ = runner_dir_+"synset_words.txt";

  bool multiFormat = false;
  json_object_object_foreach(jobj, key, val) 
  {
    std::string keyString(key);
    if(keyString.compare("inputs") == 0)
    {
      multiFormat = true;
      assert(val);
    }
  }

  if(not multiFormat)
  {
    outW_ = getValue("outW", jobj);
    outH_ = getValue("outH", jobj);
    outCh_ = getValue("outCh", jobj);
    inW_ = getValue("inW", jobj);
    inH_ = getValue("inH", jobj);
    inCh_ = getValue("inCh", jobj);
    
  }
  
  if(multiFormat)
  {
    json_object_object_foreach(jobj, key, val)
    {
      std::string keyString(key);
      if(keyString.compare("inputs") == 0)
      {
        json_object_object_foreach(val, inputkey, inputval)
        {
          assert(inputkey);
          json_object* obj = json_object_object_get(inputval, "shape");
          json_object* shapeVal;
          shapeVal = json_object_array_get_idx(obj, 1);
          inH_ = json_object_get_int(shapeVal);
          shapeVal = json_object_array_get_idx(obj, 2);
          inW_ = json_object_get_int(shapeVal);
          shapeVal = json_object_array_get_idx(obj, 3);
          inCh_ = json_object_get_int(shapeVal);
        }
      }
      if(keyString.compare("outputs") == 0)
      {
        json_object_object_foreach(val, outputkey, outputval)
        {
          assert(outputkey);
          json_object* obj = json_object_object_get(outputval, "shape");
          json_object* shapeVal;
          shapeVal = json_object_array_get_idx(obj, 1);
          outH_ = json_object_get_int(shapeVal);
          shapeVal = json_object_array_get_idx(obj, 2);
          outW_ = json_object_get_int(shapeVal);
          shapeVal = json_object_array_get_idx(obj, 3);
          outCh_ = json_object_get_int(shapeVal);
        }
      }
    }
  }

}

void jsonOrXirKeys::loadFromXmodel(std::string xmodelFname)
{

  std::unique_ptr<xir::Graph> graph = xir::Graph::deserialize(xmodelFname);

  std::vector<xir::Subgraph *> subgraphs = graph->get_root_subgraph()->children_topological_sort();
  auto subgraph = subgraphs[1];//TO_DO - replace 1 with automated value
  runner_dir_ = subgraph->get_attr<std::string>("runner_dir");
  bool multiFormat = true;
  
  auto attrs = subgraph->get_attrs();
  auto keys = attrs->get_keys();
  for (auto& key : keys) {
    std::string keyString(key);
    if(keyString.compare("inW")==0)
    {
      multiFormat = false;
      break;
    }
  }
  
  if(multiFormat)
  {
    std::string tensorInfo = subgraph->get_attr<std::string>("tensor_info");
    const char * c = tensorInfo.c_str();
    json_object* jobj = json_tokener_parse(c);
    json_object_object_foreach(jobj, key, val)
    {
      std::string keyString(key);
      if(keyString.compare("inputs") == 0)
      {
        json_object_object_foreach(val, inputkey, inputval)
        {
          assert(inputkey);
          json_object* obj = json_object_object_get(inputval, "shape");
          json_object* shapeVal;
          shapeVal = json_object_array_get_idx(obj, 1);
          inH_ = json_object_get_int(shapeVal);
          shapeVal = json_object_array_get_idx(obj, 2);
          inW_ = json_object_get_int(shapeVal);
          shapeVal = json_object_array_get_idx(obj, 3);
          inCh_ = json_object_get_int(shapeVal);
        }
      }
      if(keyString.compare("outputs") == 0)
      {
        json_object_object_foreach(val, outputkey, outputval)
        {
          assert(outputkey);
          json_object* obj = json_object_object_get(outputval, "shape");
          json_object* shapeVal;
          shapeVal = json_object_array_get_idx(obj, 1);
          outH_ = json_object_get_int(shapeVal);
          shapeVal = json_object_array_get_idx(obj, 2);
          outW_ = json_object_get_int(shapeVal);
          shapeVal = json_object_array_get_idx(obj, 3);
          outCh_ = json_object_get_int(shapeVal);
         
        }
      }
    }
  }
  else
  {
    outH_ = 1;//subgraph->get_attr<int>("outH");
    outW_ = 1;//subgraph->get_attr<int>("outW");
    outCh_ = 1000;//subgraph->get_attr<int>("outCh");
    inW_ = subgraph->get_attr<int>("inW");
    inH_ = subgraph->get_attr<int>("inH");
    inCh_ = subgraph->get_attr<int>("inCh");
 }
  
  debugMode_ = false;//subgraph->get_attr<int>("debugMode");
  golden_filename_ = runner_dir_+"gold.txt";//subgraph->get_attr<std::string>("goldenFile");
  synset_filename_ = runner_dir_+"synset_words.txt";//subgraph->get_attr<std::string>("synsetFile");

}

std::string jsonOrXirKeys::getFileNameIfExists(std::string name, json_object* jobj)
{
  json_object *obj = NULL;
  if (!json_object_object_get_ex(jobj, name.c_str(), &obj))
    throw std::runtime_error("Error: missing "+name+" field in meta.json");
  return json_object_get_string(obj);

}

uint32_t jsonOrXirKeys::getValue(std::string name, json_object* jobj)
{
  json_object *obj = NULL;
  if (!json_object_object_get_ex(jobj, name.c_str(), &obj))
    throw std::runtime_error("Error: missing "+name+" field in meta.json");
  return json_object_get_int(obj);

}

bool jsonOrXirKeys::getBool(std::string name, json_object* jobj)
{
  json_object *obj = NULL;
  if (!json_object_object_get_ex(jobj, name.c_str(), &obj))
  {  
    if(name=="usexmodel")
      return false;
    else
      throw std::runtime_error("Error: missing "+name+" field in meta.json");
  }
  return json_object_get_boolean(obj);

}


