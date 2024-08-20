#include "logging.h"
#include "errors.h"

#include <iostream>
#include <utility>
#include <cstring>
#include <unistd.h>

#ifdef __ANDROID__
#include <android/log.h>
#else
// dummy values, since they won't be used anyway
#define ANDROID_LOG_DEBUG 0
#define ANDROID_LOG_INFO 0
#define ANDROID_LOG_ERROR 0
#endif

namespace libllmod {

namespace {

thread_local Logger* active_logger = nullptr;

void dispatch_message(std::ostream& out, uint64_t timestamp, const char* level_tag, int android_prio, const char* message, bool devmsg) {
    auto&& len = strlen(message);
    while (len && message[len-1] == '\n')
        --len;

    if (!devmsg) {
        out.write("[", 1);
        out << getpid();
        out.write("][+", 3);
        out << timestamp;
        out.write("]:", 2);
        out.write(level_tag, strlen(level_tag));
    } else {
        out << "DEVMSG";
    }
    out.write(" ", 1);
    out.write(message, len);
    if (len && message[len-1] != '\n')
        out.write("\n", 1);
    out.flush();

#ifdef __ANDROID__
    __android_log_write(android_prio, "libllmod", message);
#endif
}

} // end local

bool is_valid_log_level(int log_level) {
    return log_level >= 0 && log_level < LIBLLMOD_NUM_LOG_LEVELS;
}

bool is_enabled(LogLevel level) {
    if (!active_logger)
        return false;
    return active_logger->get_level() >= level;
}

void message(LogLevel level, std::string const& str) {
    if (active_logger)
        active_logger->message(level, str);
}

void message(uint64_t timestamp, LogLevel level, std::string const& str) {
    if (active_logger)
        active_logger->message(timestamp, level, str);
}

} // end libsd functions

using namespace libllmod;

Logger::Logger() : current_level(LogLevel::NOTHING) {
    created = std::time(nullptr);
}

Logger::~Logger() {
    if (active_logger == this)
        active_logger = nullptr;
}

void Logger::set_level(LogLevel level) {
    current_level = level;
}

void Logger::message(uint64_t timestamp, LogLevel level, std::string const& str) {
    if (current_level < level || current_level <= LogLevel::NOTHING)
        return;

    switch (level) {
    case LogLevel::ABUSIVE:
        dispatch_message(std::cout, timestamp > created ? timestamp - created : 0, "[ABUSIVE]", ANDROID_LOG_DEBUG, str.c_str(), false);
        break;
    case LogLevel::DEBUG:
        dispatch_message(std::cout, timestamp > created ? timestamp - created : 0, "[DEBUG]", ANDROID_LOG_DEBUG, str.c_str(), false);
        break;
    case LogLevel::INFO:
        dispatch_message(std::cout, timestamp > created ? timestamp - created : 0, "[INFO]", ANDROID_LOG_INFO, str.c_str(), false);
        break;
    case LogLevel::ERROR:
        dispatch_message(std::cerr, timestamp > created ? timestamp - created : 0, "[ERROR]", ANDROID_LOG_ERROR, str.c_str(), false);
        break;
#ifdef LIBLLMOD_DEBUG
    case LogLevel::DEVELOPER:
        dispatch_message(std::cout, 0, nullptr, ANDROID_LOG_DEBUG, str.c_str(), true);
        break;
#endif
    case LogLevel::NOTHING:
        throw libllmod_exception(ErrorCode::INTERNAL_ERROR, "Unreachable", __func__, __FILE__, STR(__LINE__));
    }
}

ActiveLoggerScopeGuard::ActiveLoggerScopeGuard(Logger& logger) : prev(active_logger) {
    active_logger = &logger;
}

ActiveLoggerScopeGuard::ActiveLoggerScopeGuard(ActiveLoggerScopeGuard&& other) : prev(other.prev) {
    other.active = false;
}

ActiveLoggerScopeGuard::~ActiveLoggerScopeGuard() {
    if (active)
        active_logger = prev;
}
