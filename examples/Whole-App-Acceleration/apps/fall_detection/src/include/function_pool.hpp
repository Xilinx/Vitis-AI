#pragma once
#include <queue>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>

class Function_pool
{

private:
    std::vector<std::thread> thread_pool = {};
    std::queue<std::string> m_arg_queue;
    std::function<void(std::string)> m_function;
    std::mutex m_lock;
    std::condition_variable m_data_condition;
    bool m_accept_functions;

public:
    Function_pool(int, std::function<void(std::string)>);
    ~Function_pool() = default;
    void push(std::string);
    void done();
    void infinite_loop_func();
    void wait_on_thread_pool();
};
