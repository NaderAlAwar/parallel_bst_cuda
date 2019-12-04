#include <fstream>
#include <iomanip>

#include "kernel_timer.h"
#include "json.hpp"

kernel_timer::kernel_timer()
    : is_running(false), kernel_setup_time(0),
    kernel_execution_time(0),
    kernel_result_time(0) {
}

kernel_timer::~kernel_timer() {
}

void kernel_timer::start() {
    if (!is_running) {
        is_running = true;
        current_starting_time = high_resolution_clock::now();
    }
}

void kernel_timer::stop() {
    auto current_stopping_time = high_resolution_clock::now();
    int64_t current_duration = duration_cast<microseconds>(current_stopping_time - current_starting_time).count();

    if (is_running) {
        if (kernel_setup_time == 0) {
            kernel_setup_time = current_duration;
        } else if (kernel_execution_time == 0) {
            kernel_execution_time = current_duration;
        } else if (kernel_result_time == 0) {
            kernel_result_time = current_duration;
        }
    }
    is_running = false;
}

void kernel_timer::write_to_file(const std::string& file_path, const std::string& test_case_name, int test_size) {
    std::fstream ifs(file_path);
    nlohmann::json output_json = nlohmann::json::parse(ifs);
    ifs.close();

    nlohmann::json j = nlohmann::json{
        {"test_case_name", test_case_name},
        {"test_size", test_size},
        {"setup_time", kernel_setup_time},
        {"execution_time", kernel_execution_time},
        {"result_time", kernel_result_time},
        {"total_time", kernel_setup_time + kernel_execution_time + kernel_result_time}
        };
    output_json["gpu_results"].push_back(j);

    std::ofstream ofs(file_path);
    ofs << std::setw(4) << output_json << std::endl;
    ofs.close();
}