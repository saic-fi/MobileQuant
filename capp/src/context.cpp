#include "context.h"
#include "error.h"
#include "utils.h"

#include <cmath>
#include <array>
#include <mutex>
#include <thread>
#include <iostream>

#include <unistd.h>
#include <fcntl.h>


using namespace libllmod;
using namespace std::chrono_literals;

#define LIBLLMOD_USE_PROCESSES


void libllmod::time_in_ms(
    std::string name,
    std::chrono::high_resolution_clock::time_point const& t1,
    std::chrono::high_resolution_clock::time_point const& t2) {

    auto&& diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    info("{} took {}ms", name.c_str(), diff.count());
};


Context::Context(std::string const& models_dir, LogLevel log_level, std::string const& device_type, bool use_htp)
    : models_dir(models_dir), device_type(device_type), use_htp(use_htp) {
    _error_table = allocate_error_table();
    _logger.set_level(log_level);
    if (models_dir.empty())
        this->models_dir = '.';
    else if (models_dir.back() == '/')
        this->models_dir.pop_back();
}


Context::~Context() {
    inputs.clear();
    outputs.clear();
    graphs.clear();
    _qnn.reset();
}


void Context::init() {
    auto&& tick = std::chrono::high_resolution_clock::now();

    initialize_qnn();
    load_models();

    auto&& tock = std::chrono::high_resolution_clock::now();
    auto&& diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
    info("Context::init(): init QNN & graphs took {}ms", diff.count());

    prepare_buffers();
}


void Context::initialize_qnn() {
    if (_qnn_initialized)
        return;

    _qnn = std::shared_ptr<QnnBackend>(new QnnBackend(use_htp ? QnnBackendType::HTP : QnnBackendType::GPU, device_type));
    _qnn_initialized = true;
}


void Context::load_models() {
    if (!_qnn_initialized)
        return;

    auto&& suffix = use_htp ? ".bin" : ".qnn.so";

    std::string filename = format("{}/qnn_model{}", models_dir, suffix);
    auto&& tick = std::chrono::high_resolution_clock::now();
    auto&& g = _qnn->load_graphs(filename, use_htp);
    if (g.empty()) {
        throw libllmod_exception(ErrorCode::INVALID_ARGUMENT, format("Deserialized context {} does not contain any graphs!", filename), "load_models", __FILE__, STR(__LINE__));
    }
    auto&& tock = std::chrono::high_resolution_clock::now();
    auto&& diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
    graphs.splice(graphs.end(), std::move(g));
    info("Loading {} graphs took {} ms", graphs.size(), diff.count());

    // std::cout << "num_inputs: " << graphs.front().get_num_inputs() << std::endl;
    // std::cout << "num_outputs: " << graphs.front().get_num_outputs() << std::endl;

    for (int i = 0; i < graphs.front().get_num_inputs(); ++i) {
        inputs.emplace_back(graphs.front().allocate_input(i));
    }

    for (int i = 0; i < graphs.front().get_num_outputs(); ++i) {
        outputs.emplace_back(graphs.front().allocate_output(i));
    }

    // for (auto&& g : graphs) {
    //     inputs.emplace_back(g.allocate_input(0));
    //     outputs.emplace_back(g.allocate_output(0));
    // }

    // input_elements = inputs.front().get_num_elements(1); // 1 = batch size
    // output_elements = outputs.back().get_num_elements(1);

    input_elements = graphs.front().get_num_inputs();
    output_elements = graphs.back().get_num_outputs();

    // debug("Signalling readiness, input elements: {}, output_elements: {}", input_elements, output_elements);
}