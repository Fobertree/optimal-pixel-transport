//
// Created by Alexander Liu on 4/3/26.
//

#ifndef OPTIMALPIXELTRANSPORT_THREADPOOL_H
#define OPTIMALPIXELTRANSPORT_THREADPOOL_H

#include <queue>
#include <thread>
#include <functional>

// std::thread on Emscripten requires enabling Emscripten pthread compile flag
// This is CPU-bound compute, so this is likely not the fastest way to do things
// Impl taken from Stack Overflow
class ThreadPool {
    // thread pool, but with promises
public:
    void Start();

    void QueueJob(const std::function<void()> &job);

    void Sync(); // join threads, but don't clear

    void Stop();

    bool busy();

private:
    void ThreadLoop();

    int MAX_THREADS = std::thread::hardware_concurrency();
    std::queue<std::function<void()>> jobs_;
    std::vector<std::thread> threads_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool should_terminate_ = false;
};

#endif //OPTIMALPIXELTRANSPORT_THREADPOOL_H
