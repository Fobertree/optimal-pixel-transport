//
// Created by Alexander Liu on 4/3/26.
//

#include "threadpool.h"

void ThreadPool::Start() {
    const uint32_t num_threads = std::thread::hardware_concurrency(); // Max # of threads the system supports
    for (uint32_t ii = 0; ii < num_threads; ++ii) {
        threads_.emplace_back(std::thread(&ThreadPool::ThreadLoop, this));
    }
}

void ThreadPool::ThreadLoop() {
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] {
                return !jobs_.empty() || should_terminate_;
            });
            if (should_terminate_) {
                return;
            }
            job = jobs_.front();
            jobs_.pop();
        }
        job();
    }
}

void ThreadPool::QueueJob(const std::function<void()> &job) {
    {
        std::unique_lock<std::mutex> lock(mtx_);
        jobs_.push(job);
    }
    cv_.notify_one();
}

// Source - https://stackoverflow.com/a/32593825
// Posted by PhD AP EcE, modified by community. See post 'Timeline' for change history
// Retrieved 2026-04-04, License - CC BY-SA 4.0

bool ThreadPool::busy() {
    bool poolbusy;
    {
        std::unique_lock<std::mutex> lock(mtx_);
        poolbusy = !jobs_.empty();
    }
    return poolbusy;
}

void ThreadPool::Sync() {
    // maybe combine this with Stop for one more arg
    {
        std::unique_lock<std::mutex> lock(mtx_);
        should_terminate_ = true;
    }
    cv_.notify_all();
    for (std::thread &active_thread: threads_) {
        active_thread.join();
    }
}


void ThreadPool::Stop() {
    {
        std::unique_lock<std::mutex> lock(mtx_);
        should_terminate_ = true;
    }
    cv_.notify_all();
    for (std::thread &active_thread: threads_) {
        active_thread.join();
    }
    threads_.clear();
}

