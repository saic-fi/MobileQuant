#ifndef LIBLLMOD_BUFFER_H
#define LIBLLMOD_BUFFER_H

#include <cstddef>
#include <memory>


namespace libllmod {

template <class T>
class Buffer;

template <>
class Buffer<void> {
public:
    Buffer();
    Buffer(std::size_t len);
    Buffer(void* ptr, size_t len);

    Buffer(Buffer const&) = delete;
    Buffer(Buffer&& other);

    virtual ~Buffer();

    void* data_ptr() const { return _ptr; }
    size_t data_len() const { return _len; }
    void own(bool own) { _own = own; }

private:
    void* _ptr = nullptr;
    std::size_t _len = 0;
    bool _own = true;
};

template <class T>
class Buffer : public Buffer<void> {
public:
    Buffer(size_t len) : Buffer<void>(len * sizeof(T)) {}
    Buffer(T* ptr, size_t len) : Buffer<void>(static_cast<void*>(ptr), len * sizeof(T)) {}
    Buffer(Buffer const&) = delete;
    Buffer(Buffer&& other) : Buffer<void>(std::move(other)) {}
    T* data_ptr() const { return reinterpret_cast<T*>(Buffer<void>::data_ptr()); }
    size_t data_len() const { return Buffer<void>::data_len() / sizeof(T); }
};

}

#endif // LIBLLMOD_BUFFER_H
