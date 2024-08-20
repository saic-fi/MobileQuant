#include "libllmod.h"

#include "errors.h"
#include "context.h"
#include "utils.h"
#include "llm.h"

#include <string>
#include <cstring>

#define LIBLLMOD_VERSION_MAJOR 1
#define LIBLLMOD_VERSION_MINOR 0
#define LIBLLMOD_VERSION_PATCH 0

#define LIBLLMOD_VERSION_STR (#LIBLLMOD_VERSION_MAJOR "." #LIBLLMOD_VERSION_MINOR "." #LIBLLMOD_VERSION_PATCH)
#define LIBLLMOD_VERSION_INT (LIBLLMOD_VERSION_MAJOR*10000 + LIBLLMOD_VERSION_MINOR*100 + LIBLLMOD_VERSION_PATCH)
#define LIBLLMOD_CONTEXT_MAGIC_HEADER 0x00534443
#define LIBLLMOD_DEFAULT_CONTEXT_VERSION 1


namespace libllmod {

struct CAPI_Context_Handler {
    unsigned int magic_info = LIBLLMOD_CONTEXT_MAGIC_HEADER;
    unsigned int context_version = LIBLLMOD_DEFAULT_CONTEXT_VERSION;
    unsigned int ref_count = 0;
    Context* cptr = nullptr;
    std::string model_type;
};

template <class T>
ErrorCode _error(ErrorCode code, Context* c, T&& message, const char* func, const char* file, const char* line) {
    ErrorTable tab = nullptr;
    if (c)
        tab = c->get_error_table();

    auto last_sep = strrchr(file, '/');
    if (last_sep)
        file = last_sep + 1;

    std::string msg{ func };
    msg = msg + ": " + message + " [" + file + ":" + line + "]";
    record_error(tab, code, msg);
    return code;
}

#define ERROR(code, reason) _error(code, cptr, reason, __func__, __FILE__, STR(__LINE__))


#define TRY_RETRIEVE_CONTEXT \
    Context* cptr = nullptr; \
    if (context == nullptr) \
        return ERROR(ErrorCode::INVALID_CONTEXT, "context is nullptr"); \
    auto hnd = reinterpret_cast<CAPI_Context_Handler*>(context); \
    if (hnd->magic_info != LIBLLMOD_CONTEXT_MAGIC_HEADER) \
        return ERROR(ErrorCode::INVALID_CONTEXT, "context magic header mismatch! got: " + std::to_string(hnd->magic_info)); \
    if (hnd->context_version != LIBLLMOD_DEFAULT_CONTEXT_VERSION) \
        return ERROR(ErrorCode::INVALID_CONTEXT, "context version mismatch! got: " + std::to_string(hnd->context_version)); \
    if (hnd->ref_count == 0) \
        return ERROR(ErrorCode::INVALID_CONTEXT, "context has been released!"); \
    if (hnd->cptr == nullptr) \
        return ERROR(ErrorCode::INVALID_CONTEXT, "corrupted context, internal pointer is nullptr"); \
    cptr = hnd->cptr; \
    auto&& _logger_scope = cptr->activate_logger(); \
    (void)_logger_scope


static ErrorCode setup_impl(void** context, const char* models_dir, unsigned int log_level, bool use_htp,
                            const char* model_type, float temperature, float topp, unsigned long long rng_seed, int max_sequence_length) {
    Context* cptr = nullptr; // NOTE: ERROR(code, reason) requires cptr

    if (context == nullptr)
        return ERROR(ErrorCode::INVALID_ARGUMENT, "Context argument should not be nullptr!");

    if (*context != nullptr)
        return ERROR(ErrorCode::INVALID_ARGUMENT, "Context should point to a nullptr-initialized variable!");

    if (!is_valid_log_level(log_level))
        return ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid log_level");

    CAPI_Context_Handler* hnd = new (std::nothrow) CAPI_Context_Handler;
    if (hnd == nullptr)
        return ERROR(ErrorCode::FAILED_ALLOCATION, "Could not create a new CAPI_Context_Handler object");

    hnd->ref_count += 1;
    hnd->model_type = std::string(model_type);

    // model definition here
    if (hnd->model_type == "llama" || hnd->model_type == "gemma") {
        cptr = new LLM(
            models_dir, 
            hnd->model_type, 
            static_cast<LogLevel>(log_level), 
            use_htp, 
            temperature, 
            topp, 
            rng_seed, 
            max_sequence_length);
    } else {
        return ERROR(ErrorCode::FAILED_ALLOCATION, "Could not create a new Context object: unsupported model type");
    }

    hnd->cptr = cptr;
    *context = hnd; // return the context

    // now info() debug() enabled
    auto&& _logger_scope = cptr->activate_logger();
    (void)_logger_scope;

    // model init
    try {
        cptr->init();
    } catch (libllmod_exception const& e) {
        return _error(e.code(), cptr, e.reason(), e.func(), e.file(), e.line());
    } catch (std::exception const& e) {
        return ERROR(ErrorCode::INTERNAL_ERROR, e.what());
    } catch (...) {
        return ERROR(ErrorCode::INTERNAL_ERROR, "Unspecified error");
    }

    return ErrorCode::NO_ERROR;
}

static ErrorCode set_log_level_impl(void* context, unsigned int log_level) {
    TRY_RETRIEVE_CONTEXT;
    if (!is_valid_log_level(log_level))
        return ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid log_level");

    try {
        cptr->get_logger().set_level(static_cast<LogLevel>(log_level));
    } catch (libllmod_exception const& e) {
        return _error(e.code(), cptr, e.reason(), e.func(), e.file(), e.line());
    } catch (std::exception const& e) {
        return ERROR(ErrorCode::INTERNAL_ERROR, e.what());
    } catch (...) {
        return ERROR(ErrorCode::INTERNAL_ERROR, "Unspecified error");
    }

    return ErrorCode::NO_ERROR;
}

static ErrorCode ref_context_impl(void* context) {
    TRY_RETRIEVE_CONTEXT;
    ++hnd->ref_count;
    return ErrorCode::NO_ERROR;
}

static ErrorCode release_impl(void* context) {
    TRY_RETRIEVE_CONTEXT;
    if (--hnd->ref_count == 0) {
        delete cptr;
        cptr = nullptr;
        hnd->cptr = nullptr;
    }

    return ErrorCode::NO_ERROR;
}

static ErrorCode run_impl(void* context, const char* text, char** text_out, int steps, int& last_token_position) {
    // TODO: streaming

    TRY_RETRIEVE_CONTEXT;
    if (text_out == nullptr)
        return ERROR(ErrorCode::INVALID_ARGUMENT, "text_out is nullptr");

    try {
        // autoregressive generation
        cptr->generate(text, steps, text_out, last_token_position);

    } catch (libllmod_exception const& e) {
        return _error(e.code(), cptr, e.reason(), e.func(), e.file(), e.line());
    } catch (std::exception const& e) {
        return ERROR(ErrorCode::INTERNAL_ERROR, e.what());
    } catch (...) {
        return ERROR(ErrorCode::INTERNAL_ERROR, "Unspecified error");
    }

    return ErrorCode::NO_ERROR;
}

static const char* get_error_description_impl(int errorcode) {
    if (!is_valid_error_code(errorcode))
        return nullptr;

    return get_error_str(static_cast<ErrorCode>(errorcode));
}

static const char* get_last_error_extra_info_impl(int errorcode, void* context) {
    if (!is_valid_error_code(errorcode))
        return nullptr;

    ErrorTable tab = nullptr;
    if (context && errorcode != std::underlying_type_t<ErrorCode>(ErrorCode::INVALID_CONTEXT)) {
        auto hnd = reinterpret_cast<CAPI_Context_Handler*>(context);
        if (hnd->magic_info == LIBLLMOD_CONTEXT_MAGIC_HEADER
            && hnd->context_version == LIBLLMOD_DEFAULT_CONTEXT_VERSION
            && hnd->ref_count > 0
            && hnd->cptr != nullptr)
            tab = hnd->cptr->get_error_table();
    }

    return get_last_error_info(tab, static_cast<ErrorCode>(errorcode));
}

}

extern "C" {

LIBLLMOD_API int libllmod_setup(void** context, const char* models_dir, unsigned int log_level, int use_htp,
                                const char* model_type, float temperature, float topp, unsigned long long rng_seed, int max_sequence_length) {
    return static_cast<int>(libllmod::setup_impl(context, models_dir, log_level, static_cast<bool>(use_htp),
                                                 model_type, temperature, topp, rng_seed, max_sequence_length));
}

LIBLLMOD_API int libllmod_set_log_level(void* context, unsigned int log_level) {
    return static_cast<int>(libllmod::set_log_level_impl(context, log_level));
}

LIBLLMOD_API int libllmod_ref_context(void* context) {
    return static_cast<int>(libllmod::ref_context_impl(context));
}

LIBLLMOD_API int libllmod_release(void* context) {
    return static_cast<int>(libllmod::release_impl(context));
}

LIBLLMOD_API int libllmod_run(void* context, const char* text, char** text_out, int steps, int& last_token_position) {
    return static_cast<int>(libllmod::run_impl(context, text, text_out, steps, last_token_position));
}

LIBLLMOD_API const char* libllmod_get_error_description(int errorcode) {
    return libllmod::get_error_description_impl(errorcode);
}

LIBLLMOD_API const char* libllmod_get_last_error_extra_info(int errorcode, void* context) {
    return libllmod::get_last_error_extra_info_impl(errorcode, context);
}

}
