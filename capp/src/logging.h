#ifndef LIBLLMOD_LOGGING_H
#define LIBLLMOD_LOGGING_H

#include <string>
#include <ctime>

#include "utils.h"


namespace libllmod {

enum class LogLevel : int {
#ifdef LIBLLMOD_DEBUG
    NOTHING = -1,
    DEVELOPER,
#else
    NOTHING = 0,
#endif
    ERROR,
    INFO,
    DEBUG,
    ABUSIVE
};

constexpr unsigned int LIBLLMOD_NUM_LOG_LEVELS = 5;

bool is_valid_log_level(int loglevel);
bool is_enabled(LogLevel level);
void set_level(LogLevel level);
void message(LogLevel level, std::string const& str);
void message(uint64_t timestamp, LogLevel level, std::string const& str);

template <class... T>
void info(std::string const& fmt, T&&... args) {
    if (!is_enabled(LogLevel::INFO))
        return;
    return message(LogLevel::INFO, format(fmt, std::forward<T>(args)...));
}

template <class... T>
void debug(std::string const& fmt, T&&... args) {
    if (!is_enabled(LogLevel::DEBUG))
        return;
    return message(LogLevel::DEBUG, format(fmt, std::forward<T>(args)...));
}

template <class... T>
void error(std::string const& fmt, T&&... args) {
    if (!is_enabled(LogLevel::ERROR))
        return;
    return message(LogLevel::ERROR, format(fmt, std::forward<T>(args)...));
}

template <class... T>
void abusive(std::string const& fmt, T&&... args) {
    if (!is_enabled(LogLevel::ABUSIVE))
        return;
    return message(LogLevel::ABUSIVE, format(fmt, std::forward<T>(args)...));
}

#ifdef LIBLLMOD_DEBUG
template <class... T>
void devmsg(std::string const& fmt, T&&... args) {
    return message(LogLevel::DEVELOPER, format(fmt, std::forward<T>(args)...));
}
#endif


class Logger {
public:
    Logger();
    virtual ~Logger();

    void set_level(LogLevel level);
    LogLevel get_level() const { return current_level; }

    void message(LogLevel level, std::string const& str) { return message(std::time(nullptr), level, str); }
    void message(uint64_t timestamp, LogLevel level, std::string const& str);

private:
    LogLevel current_level;
    uint64_t created;
};


class ActiveLoggerScopeGuard {
public:
    ActiveLoggerScopeGuard(Logger& logger);
    ActiveLoggerScopeGuard(ActiveLoggerScopeGuard&& other);
    ActiveLoggerScopeGuard(ActiveLoggerScopeGuard const&) = delete;
    ~ActiveLoggerScopeGuard();

    ActiveLoggerScopeGuard& operator =(ActiveLoggerScopeGuard const&) = delete;
    ActiveLoggerScopeGuard& operator =(ActiveLoggerScopeGuard&&) = delete;

private:
    bool active = true;
    Logger* prev = nullptr;
};


}

#endif // LIBLLMOD_LOGGING_H
