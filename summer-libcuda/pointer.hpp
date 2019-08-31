#pragma once


#include "stdafx.hpp"


namespace Summer { namespace CUDA {


template<typename T> class Pointer final {
	public:
		union {
			T*          ptr;
			CUdeviceptr ptr_integral;
		};

	public:
		Pointer() = default;
		Pointer(T* ptr) {
			this->ptr = ptr;
		}
		template<typename T2> Pointer(Pointer<T2> const& other) {
			ptr_integral = other.ptr_integral;
		}
		~Pointer() = default;

		__device_host__ operator T*() const { return ptr; }

		__device_host__ bool operator==(T* other) const { return ptr==other; }
		__device_host__ bool operator!=(T* other) const { return ptr!=other; }

		__device_host__ T const& operator[](size_t index) const { return ptr[index]; }
		__device_host__ T&       operator[](size_t index)       { return ptr[index]; }
};


}}
