#pragma once


#include "stdafx.hpp"


namespace Summer { namespace CUDA {


class Device;


class Context final {
	public:
		CUcontext context;

		CUstream stream;

	public:
		explicit Context(Device const* device_cuda);
		~Context();
};


}}
