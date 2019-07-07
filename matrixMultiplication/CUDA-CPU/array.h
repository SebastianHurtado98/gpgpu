#ifndef _DEV_ARRAY_H_
#define _DEV_ARRAY_H_

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

template <class T>
class array
{
public:
    explicit array()
        : start_(0),
          end_(0)
    {}

    explicit array(size_t size)
    {
        allocate(size);
    }

    ~array()
    {
        free();
    }

    void resize(size_t size)
    {
        free();
        allocate(size);
    }

    size_t getSize() const
    {
        return end_ - start_;
    }

    const T* getData() const
    {
        return start_;
    }

    T* getData()
    {
        return start_;
    }

    void set(const T* src, size_t size)
    {
        size_t min = std::min(size, getSize());
        cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
        if (!result)
        {
            throw std::runtime_error("failed to copy to device memory");
        }
    }

    void get(T* dest, size_t size)
    {
        size_t min = std::min(size, getSize());
        cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
        if (!result)
        {
            throw std::runtime_error("failed to copy to host memory");
        }
    }


private:
    void allocate(size_t size)
    {
        cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
        if (!result)
        {
            start_ = end_ = 0;
            throw std::runtime_error("failed to allocate device memory");
        }
        end_ = start_ + size;
    }

    void free()
    {
        if (start_ != 0)
        {
            cudaFree(start_);
            start_ = end_ = 0;
        }
    }

    T* start_;
    T* end_;
};

#endif