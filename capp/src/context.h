#ifndef LIBLLMOD_CONTEXT_H
#define LIBLLMOD_CONTEXT_H

#include <string>
#include <random>
#include <chrono>

#include "errors.h"
#include "qnn_context.h"
#include "logging.h"


namespace libllmod {


void time_in_ms(
    std::string name,
    std::chrono::high_resolution_clock::time_point const& t1, 
    std::chrono::high_resolution_clock::time_point const& t2
);


class Context {
public:
    Context(std::string const& models_dir, LogLevel log_level, bool use_htp=true);
    virtual ~Context();

    void init();
    void initialize_qnn();
    void load_models();
    virtual void prepare_buffers() = 0;
    virtual void run() = 0;

    ErrorTable get_error_table() const { return _error_table; }

    Logger& get_logger() { return _logger; }
    Logger const& get_logger() const { return _logger; }
    ActiveLoggerScopeGuard activate_logger() { return ActiveLoggerScopeGuard(_logger); }

    virtual void generate(const char* prompt, int steps, char** text_out, int& last_token_position) = 0; // for TRY_RETRIEVE_CONTEXT

protected: // to allow child classes to access these members
    std::string models_dir;
    bool use_htp;

    bool _qnn_initialized = false;

    ErrorTable _error_table;
    Logger _logger;

    std::shared_ptr<QnnBackend> _qnn;
    // if a single graph, create the graph in the main process
    graph_list graphs;
    std::vector<QnnTensor> inputs;
    std::vector<QnnTensor> outputs;
    unsigned int input_elements = 0;
    unsigned int output_elements = 0;
};


} // end of namespace libllmod

#endif // LIBLLMOD_CONTEXT_H
