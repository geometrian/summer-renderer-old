#pragma once


#include "stdafx.hpp"


namespace Summer { namespace CUDA {


class Device final {
	public:
		CUdevice device;

		cudaDeviceProp properties;

	public:
		explicit Device(size_t index);
		~Device() = default;

		void set_current() const;

		static size_t get_count();
};


}}
