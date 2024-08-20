#include "buffer.h"
#include "errors.h"
#include "utils.h"

#include <cstdlib>

using namespace libllmod;


Buffer<void>::Buffer() {}

Buffer<void>::Buffer(std::size_t len){
    _ptr = malloc(len);
    if (!_ptr)
        throw libllmod_exception(ErrorCode::FAILED_ALLOCATION, "Could not allocate a buffer of size: " + std::to_string(len), __func__, __FILE__, STR(__LINE__));
    _len = len;
    _own = true;
}

Buffer<void>::Buffer(void* ptr, std::size_t len) {
    _ptr = ptr;
    _len = len;
    _own = false;
}

Buffer<void>::Buffer(Buffer&& other) {
    _ptr = other._ptr;
    _len = other._len;
    _own = other._own;
    other._own = false;
}

Buffer<void>::~Buffer() {
    if (_own && _ptr) {
        free(_ptr);
    }
    _ptr = nullptr;
    _len = 0;
}
