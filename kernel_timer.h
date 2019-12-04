#ifndef KERNEL_TIMER_H
#define KERNEL_TIMER_H

#include <chrono>
#include <string>

using namespace std::chrono;

class kernel_timer {
private:
    high_resolution_clock::time_point current_starting_time;
    bool is_running;
public:
    kernel_timer();
    ~kernel_timer();
    
    void start(); // will record the times in order: setup, execution, and result
    void stop();

    // test_case_name is the name of input file, test_size is the size of the input to the kernel
    void write_to_file(const std::string& file_path, const std::string& test_case_name, int test_size); 

    // all times in microseconds
    int64_t kernel_setup_time; 
    int64_t kernel_execution_time;
    int64_t kernel_result_time;
};

#endif