/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "vitis/ai/json_object_visitor.hpp"
#include <iostream>
using namespace std;
#define S_(EXPR) (std::string(&((#EXPR)[1]), sizeof(#EXPR) - 3))

struct Point {
  int x;
  int y;
  void VisitAttrs(vitis::ai::JsonObjectVisitor visitor) {
    visitor["x"].visit(x);
    visitor["y"].visit(y);
  }
};
struct Home {
  Point location;
  string name;
  void VisitAttrs(vitis::ai::JsonObjectVisitor visitor) {
    visitor["location"].visit(location);
    visitor["name"].visit(name);
  }
};

struct PriorBoxParam {
  int layer_width;
  int layer_height;
  std::vector<float> variances;
  std::vector<float> min_sizes;
  std::vector<float> max_sizes;
  std::vector<float> aspect_ratios;
  float offset;
  float step_width;
  float step_height;
  bool flip;
  bool clip;
  void VisitAttrs(vitis::ai::JsonObjectVisitor visitor) {
    visitor["layer_width"].visit(layer_width);
    visitor["layer_height"].visit(layer_height);
    visitor["variances"].visit(variances);
    visitor["min_sizes"].visit(min_sizes);
    visitor["max_sizes"].visit(max_sizes);
    visitor["aspect_ratios"].visit(aspect_ratios);
    visitor["offset"].visit(offset);
    visitor["step_width"].visit(step_width);
    visitor["step_height"].visit(step_height);
    visitor["flip"].visit(flip);
    visitor["clip"].visit(clip);
  }
};

int main(int argc, char *argv[]) {
  auto test_string1 =
      S_(({"location" : {"x" : 1, "y" : 2}, "name" : "Charles"}));
  auto test_string2 = S_(({
    "prior_box_param" : [
      {
        "layer_width" : 60,
        "layer_height" : 45,
        "variances" : [ 0.1, 0.1, 0.2, 0.2 ],
        "min_sizes" : [21.0],
        "max_sizes" : [45.0],
        "aspect_ratios" : [2.0],
        "offset" : 0.5,
        "step_width" : 8.0,
        "step_height" : 8.0,
        "flip" : true,
        "clip" : false
      },
      {
        "layer_width" : 30,
        "layer_height" : 23,
        "variances" : [ 0.1, 0.1, 0.2, 0.2 ],
        "min_sizes" : [45.0],
        "max_sizes" : [99.0],
        "aspect_ratios" : [ 2.0, 3.0 ],
        "offset" : 0.5,
        "step_width" : 16.0,
        "step_height" : 16.0,
        "flip" : true,
        "clip" : false
      }
    ]
  }));

  // cout << test_string1 << endl;
  auto json_obj1 = json_tokener_parse(test_string1.c_str());
  auto home = Home();
  vitis::ai::JsonObjectVisitor(json_obj1).visit(home);

  cout << "json string sample1 :" << endl;
  cout << "location: (" << home.location.x << "," //
       << home.location.y << ")"                  //
       << "name=: " << home.name << ";" << endl;
  cout << endl;
  cout << endl;

  // cout << test_string2 << endl;
  auto json_obj2 = json_tokener_parse(test_string2.c_str());
  std::vector<PriorBoxParam> prior_boxes;
  vitis::ai::JsonObjectVisitor(json_obj2)["prior_box_param"].visit(prior_boxes);

  cout << "json string sample2 :" << endl;
  for (auto &prior_box : prior_boxes) {
    cout << "layer_width : " << prior_box.layer_width << "," << endl;
    cout << "layer_height : " << prior_box.layer_height << "," << endl;
    cout << "variances : [";
    for (auto variances : prior_box.variances) {
      cout << variances << ", ";
    }
    cout << "]" << endl;
    cout << "min_sizes : [";
    for (auto min_sizes : prior_box.min_sizes) {
      cout << min_sizes << ", ";
    }
    cout << "]" << endl;
    cout << "max_sizes : [";
    for (auto max_sizes : prior_box.max_sizes) {
      cout << max_sizes << ", ";
    }
    cout << "]" << endl;
    cout << "aspect_ratios : [";
    for (auto aspect_ratios : prior_box.aspect_ratios) {
      cout << aspect_ratios << ", ";
    }
    cout << "]" << endl;
    cout << "offset : " << prior_box.offset << "," << endl;
    cout << "step_width : " << prior_box.step_width << "," << endl;
    cout << "step_height : " << prior_box.step_height << "," << endl;
    cout << "flip : " << prior_box.flip << "," << endl;
    cout << "clip : " << prior_box.clip << "," << endl;
    cout << endl;
  }
  cout << endl;
  cout << endl;

  json_object_put(json_obj1);
  json_object_put(json_obj2);

  return 0;
}
