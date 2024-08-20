#include "utils.h"
#include "logging.h"

#include <string>
#include <fstream>
#include <cstddef>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>


namespace libllmod {

namespace details {

std::size_t get_next_insertion_point(std::string const& str, std::size_t offset) {
    for (std::size_t i=offset; i<str.length()-1; ++i) {
        if (str[i] == '{' && str[i+1] == '}') {
            return i;
        }
    }

    return str.length();
}

std::string format(std::size_t pos, std::string const& fmt) {
    return fmt;
}

std::string format(std::size_t pos, std::string&& fmt) {
    return std::move(fmt);
}

}

std::size_t get_file_size(std::string const& path) {
    auto fp = std::fopen(path.c_str(), "rb");
    if (!fp)
        return 0;

    auto&& _guard = scope_guard([fp](){ std::fclose(fp); });
    (void)_guard;

    if (!std::fseek(fp, 0, SEEK_END))
        return 0;

    auto length = std::ftell(fp);
    return length;
}

bool read_file_content(std::string const& path, std::vector<unsigned char>& buffer) {
    auto fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        debug("Could not open file: {}", path);
        return false;
    }

    auto&& _guard = scope_guard([fp](){ std::fclose(fp); });
    (void)_guard;

    if (std::fseek(fp, 0, SEEK_END))
        return 0;

    auto length = std::ftell(fp);
    std::rewind(fp);

    buffer.resize(length);
    if (buffer.size() != length)
        return false;

    auto read = std::fread(buffer.data(), 1, buffer.size(), fp);
    if (read != buffer.size())
        return false;

    return true;
}

bool write_to_file(const void* ptr, std::size_t len, std::string const& path) {
    auto fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        debug("Could not open file: {}", path);
        return false;
    }

    auto&& _guard = scope_guard([fp](){ std::fclose(fp); });
    (void)_guard;

    auto written = std::fwrite(ptr, len, 1, fp);
    return (written == len);
}


scope_guard::scope_guard(std::function<void()> const& init, std::function<void()> deinit) {
    if (init)
        init();
    this->deinit.swap(deinit);
}


scope_guard::scope_guard(std::function<void()> deinit) : deinit(std::move(deinit)) {
}


scope_guard::scope_guard(scope_guard&& other) : deinit(std::move(other.deinit)) {

}


scope_guard::~scope_guard() {
    if (deinit)
        deinit();
}


mmap_t::mmap_t(std::string const& path) : data(nullptr), size(0), fd(-1) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd == -1)
        return;

    struct stat sb;
    if (fstat(fd, &sb)) {
        close(fd);
        fd = -1;
        debug("Could not stat: {}", path.c_str());
        return;
    }

    size = sb.st_size;
    data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        fd = -1;
        data = nullptr;
        size = 0;
        debug("mmap failed");
        return;
    }
}


mmap_t::mmap_t(mmap_t&& other) : data(other.data), size(other.size), fd(other.fd) {
    other.data = nullptr;
    other.size = 0;
    other.fd = -1;
}


mmap_t::~mmap_t() {
    if (data) {
        munmap(data, size);
        data = nullptr;
        size = 0;
    }

    if (fd != -1)
        close(fd);
}


}
