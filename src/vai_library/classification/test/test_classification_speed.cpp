/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>
#include <vitis/ai/classification.hpp>

struct Batch {
  std::vector<std::string> images;  // Array of file names of Images in a Batch,
                                    // array will be of length batchSize
  std::vector<cv::Mat> data;  // Array of Raw Bits of Images in a Batch, array
                              // will be of length batchSize
};

// This Singleton Class is used to load a directory of images
// Preprocess them, and store them in buffers
// The object is made threadsafe by use of a mutex
// Threads will have to obtain lock to get access to a buffer
// Then increment batchIdx, so the next thread will grab the next buffer
// Example Usage:
//   imageCache = ImageCache::getInstance(); // Repeated calls will only obtain
//   reference to a signle object imageCache.initialize(...)
class ImageCache {
 private:
  std::mutex lock_;
  unsigned int batchIdx_=0;
  std::vector<Batch> inputBatches_;

 public:
  // Return a singleton object
  static ImageCache& getInstance() {
    static ImageCache imageCache;
    return imageCache;
  }

  // Initialize singleton
  void initialize(std::string imageDir, const unsigned int width,
                  const unsigned int height, const unsigned int batchSize) {
    batchIdx_ = 0;
    inputBatches_ = prepareInputs(imageDir, width, height, batchSize);
  }

  // Fill ImageCache with batches of images from given directory
  std::vector<Batch> prepareInputs(std::string imageDir,
                                   const unsigned int width,
                                   const unsigned int height,
                                   const unsigned int batchSize) {
    /* Load all image names.*/
    std::vector<std::string> images;
    std::vector<cv::String> files;
    cv::glob(imageDir, files);
    for (auto& cvStr : files) images.push_back(std::string(cvStr));
    assert(images.size() > 0);

    unsigned int numImages = images.size();
    unsigned int remainder = numImages % batchSize;
    if (remainder != 0) numImages += batchSize - remainder;

    std::vector<Batch> batches;
    for (unsigned int i = 0; i < numImages; i++) {
      const int imgIdx = i % images.size();
      const int batchIdx = i % batchSize;
      if (batchIdx == 0)  // Allocate new batch
      {
        Batch in;
        batches.push_back(in);
      }

      // Load image into batch
      cv::Mat orig = cv::imread(images[imgIdx]);
      cv::Mat image = cv::Mat(height, width, CV_8SC3);
      cv::resize(orig, image, cv::Size(height, width), 0, 0, cv::INTER_NEAREST);
      batches.back().images.push_back(images[imgIdx]);
      batches.back().data.push_back(image);
    }
    return batches;
  }

  Batch& getNext() {
    std::unique_lock<std::mutex> lock(lock_);
    auto currIdx = batchIdx_;
    batchIdx_ = (batchIdx_ + 1) % inputBatches_.size();
    return inputBatches_[currIdx];
  }
};

// Thread function will create runner
// Then, run numQueries worth of inference jobs
void runThread(std::string runnerDir, const unsigned int batchSize,
               const unsigned int numQueries, bool supress) {
  auto det = vitis::ai::Classification::create(runnerDir);
  assert(det->get_input_batch() >= (unsigned int)batchSize);

  ImageCache& imageCache = ImageCache::getInstance();
  for (unsigned int n = 0; n < numQueries; n++) {
    Batch inData = imageCache.getNext();
    auto results = det->run(inData.data);
    if (!supress) {
      for (unsigned int i = 0; i < inData.images.size(); i++) {
        std::cout << "\nImage: " << inData.images[i] << std::endl;
        for (const auto& r : results[i].scores)
          std::cout << "index: " << r.index << " score " << r.score
                    << " text: " << results[i].lookup(r.index) << std::endl;
      }
    }
  }
}

// Initialize Image Cache
// Split numQueries amongst numThreads
void runTest(std::string runnerDir, std::string imageDir,
             unsigned int numThreads, unsigned int numQueries, bool supress) {
  const unsigned int inHeight = 224;  // TODO: How to get from model
  const unsigned int inWidth = 224;   // TODO: How to get from model
  const unsigned int batchSize = 3;   // TODO: How to get from system

  ImageCache& imageCache = ImageCache::getInstance();
  imageCache.initialize(imageDir, inWidth, inHeight, batchSize);

  std::vector<std::thread> threads(numThreads);
  for (unsigned ti = 0; ti < threads.size(); ti++)
    threads[ti] = std::thread(runThread, runnerDir, batchSize,
                              numQueries / numThreads, supress);

  auto tstart = std::chrono::high_resolution_clock::now();

  for (unsigned ti = 0; ti < threads.size(); ti++) threads[ti].join();

  auto tend = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart);

  std::cout << "Elapsed: " << elapsed.count() << "s" << std::endl;
  std::cout << "Queries: " << numQueries << std::endl;
  std::cout << "Queries/s: " << numQueries / elapsed.count() << std::endl;
  std::cout << "Images/s: " << numQueries * batchSize / elapsed.count()
            << std::endl;
}

int main(int argc, char* argv[]) {
  // TODO: Convert to CXXOPTS or better argparsing
  if (argc < 6) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_directory>"
              << " <num_threads>"
              << " <num_queries>"
              << " <suppress>" << std::endl;
    std::abort();
  }
  runTest(argv[1], argv[2], std::stoi(argv[3]), std::stoi(argv[4]),
          ((std::string)argv[5] == "1"));
  return 0;
}
