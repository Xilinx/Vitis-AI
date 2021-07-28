#include "function_pool.hpp"
#include <iostream>

Function_pool::Function_pool(int num_threads, std::function<void(std::string)> func) :
    m_function(func),
    m_arg_queue(),
    m_lock(),
    m_data_condition(),
    m_accept_functions(true)
{
    for (int i = 0; i < num_threads; i++)
    {
        thread_pool.push_back(std::thread(&Function_pool::infinite_loop_func, this));
    }
}

void Function_pool::wait_on_thread_pool() {
    for (auto &thread : thread_pool)
    {
        thread.join();
    }
}


void Function_pool::push(std::string arg)
{
    std::unique_lock<std::mutex> lock(m_lock);
    m_arg_queue.push(arg);
    // when we send the notification immediately,
    // the consumer will try to get the lock , so unlock asap
    lock.unlock();
    m_data_condition.notify_one();
}

void Function_pool::done()
{
    std::unique_lock<std::mutex> lock(m_lock);
    m_accept_functions = false;
    lock.unlock();
    // when we send the notification immediately,
    // the consumer will try to get the lock , so unlock asap
    m_data_condition.notify_all();
    //notify all waiting threads.
}

void Function_pool::infinite_loop_func()
{
    std::string arg;
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(m_lock);
            m_data_condition.wait(lock, [this]() {
                return !m_arg_queue.empty() || !m_accept_functions; });
            if (!m_accept_functions && m_arg_queue.empty())
            {
                //lock will be release automatically.
                //finish the thread loop and let it join in the main thread.
                return;
            }
            arg = m_arg_queue.front();
            m_arg_queue.pop();
            //release the lock
        }
        m_function(arg);
    }
}