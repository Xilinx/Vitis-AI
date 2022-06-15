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
#ifndef __AKS_QUEUE_H_
#define __AKS_QUEUE_H_

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

//TODO : @abidk : do we need methods like size(), empty(), full() etc. ?

namespace AKS
{

  /// Any queue implementation should have following 3 methods.
  
  /// bool push(const T& val) {...}
  /// Push data to the queue.
  /// @param val Data to be pushed
  /// @return status true if push was successfull

  /// bool pop(T& val) {...}
  /// Pop data from the queue.
  /// @param val Reference to the popped data
  /// @return status true if pop was successfull

  /// bool close() {...}
  /// Close the queue immediately  and release all waiting threads
  /// It closes the queue even if there is pending data in the queue.
  /// So make sure all data is processed before closing the queue.
  /// @return status true if successfully closed

  /// Queue with limited size. 
  /// Push/Pop blocks the call if queue is full/empty
  template<typename T>
  class FixedSizeQueue {
    private:
      std::queue<T> _Q;
      std::mutex _mtx;
      std::condition_variable _cvPusher;
      std::condition_variable _cvPopper;
      const unsigned int _capacity;
      const std::string _name;
      bool _close=false;
    public:
      FixedSizeQueue(std::string name, unsigned int maxsize=128)
        : _name(name), _capacity(maxsize) {}

      ~FixedSizeQueue() {}

      bool push(const T& val) {
        bool status = false;
        std::unique_lock<std::mutex> lock(_mtx);
        _cvPusher.wait(lock, [this] { 
            bool dec = _Q.size() < _capacity; 
            return _close || dec; });

        if(!_close) {
          _Q.push(val);
          status = true;
          //std::cout << "Pushed data to Q -" << _name << ", #elems = " << _Q.size() << std::endl;
        }

        lock.unlock();
        _cvPopper.notify_all();
        return status;
      }

      bool pop(T& val) {
        bool status = false;
        std::unique_lock<std::mutex> lock(_mtx);
        _cvPopper.wait(lock, [this] {
            bool dec = !_Q.empty();
            return _close || dec; });

        if(!_close) {
          val = _Q.front();
          _Q.pop();
          status = true;
          //std::cout << "  Popped data from Q -" << _name << ", #elems = " << _Q.size() << std::endl;
        }

        lock.unlock();
        _cvPusher.notify_all();
        return status;
      }

      std::string getName() {
        return _name;
      }

      bool close() {
        bool status = false;

        _mtx.lock();
        _close = true;
        // std::cout << "Exit signal generated for FixedSizeQueue: " << _name << std::endl;
        _mtx.unlock();
        _cvPopper.notify_all();
        _cvPusher.notify_all();

        status = true;
        return status;
      }
  };

  /// Queue with no restriction on the size
  template<typename T>
  class UnlimitedQueue {
    private:
      std::queue<T> _Q;
      std::mutex _mtx;
      std::condition_variable _cvPusher;
      std::condition_variable _cvPopper;
      const unsigned int _capacity;
      const std::string _name;
      bool _close=false;
    public:
      UnlimitedQueue(std::string name, unsigned int maxsize=-1)
        : _name(name), _capacity(-1) {}

      ~UnlimitedQueue() {}

      bool push(const T& val) {
        bool status = false;
        _mtx.lock();
        if(_close == false) {
          _Q.push(val);
          status = true;
        }
        _mtx.unlock();
        _cvPopper.notify_all();
        return status;
      }

      bool pop(T& val) {
        bool status = false;
        std::unique_lock<std::mutex> lock(_mtx);
        _cvPopper.wait(lock, [this] {
            bool dec = !_Q.empty();
            return _close || dec; });

        if(!_close) {
          val = _Q.front();
          _Q.pop();
          status = true;
          //std::cout << "  Popped data from Q -" << _name << ", #elems = " << _Q.size() << std::endl;
        }

        lock.unlock();
        return status;
      }

      std::string getName() {
        return _name;
      }

      bool close() {
        bool status = false;

        _mtx.lock();
        _close = true;
        // std::cout << "Exit signal generated for BlockingQueue: " << _name << std::endl;
        _mtx.unlock();
        _cvPopper.notify_all();

        status = true;
        return status;
      }
  };

}  // namespace AKS
#endif  // __AKS_QUEUE_H_
