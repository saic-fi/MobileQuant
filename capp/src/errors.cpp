#include "errors.h"

#include <cstring>
#include <type_traits>

namespace libllmod {

static const char* _error_messages[LIBLLMOD_NUM_ERRORS] = {
    "No error",
    "Invalid context",
    "Invalid argument",
    "Failed to allocate memory or initialise an object",
    "Runtime error occurred",
    "Internal error occurred"
};

static constexpr unsigned int _num_error_messages = sizeof(_error_messages) / sizeof(decltype(_error_messages[0]));


ErrorTable allocate_error_table() {
    using _Table = ErrorTable::element_type;
    return ErrorTable(new _Table{});
}

static ErrorTable _contextless_error_table = allocate_error_table();

bool is_valid_error_code(int errorcode) {
    return errorcode >= 0 && errorcode < LIBLLMOD_NUM_ERRORS;
}

void record_error(ErrorTable tab, ErrorCode error) {
    if (!tab)
        tab = _contextless_error_table;
    tab->at(static_cast<unsigned int>(error)).reset();
}

void record_error(ErrorTable tab, ErrorCode error, std::string const& extra_info) {
    if (!tab)
        tab = _contextless_error_table;
    tab->at(static_cast<unsigned int>(error)).emplace(extra_info);
}

void record_error(ErrorTable tab, ErrorCode error, std::string&& extra_info) {
    if (!tab)
        tab = _contextless_error_table;
    tab->at(static_cast<unsigned int>(error)).emplace(std::move(extra_info));
}


const char* get_error_str(ErrorCode code) {
    unsigned int _i_code = static_cast<unsigned int>(code);
    if (_i_code < 0 || _i_code >= _num_error_messages)
        return nullptr;
    return _error_messages[_i_code];
}

const char* get_last_error_info(ErrorTable tab, ErrorCode error) {
    if (!tab)
        tab = _contextless_error_table;

    auto&& val = tab->at(static_cast<unsigned int>(error));
    if (val)
        return val->c_str();
    else
        return nullptr;
}


}

using namespace libllmod;

libllmod_exception::libllmod_exception(ErrorCode code, std::string msg, const char* func, const char* file, const char* line)
    : std::exception(), _code(code), _reason(std::move(msg)), _func(func), _file(file), _line(line)
{
    auto cfile = _file.c_str();
    auto last_sep = strrchr(cfile, '/');
    if (last_sep)
        cfile = last_sep + 1;
    _what = std::string("LibSD Error: ") + _reason + ", at: " + _func + " [" + cfile + ":" + _line + "]";
}
