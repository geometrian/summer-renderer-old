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
};


}}
