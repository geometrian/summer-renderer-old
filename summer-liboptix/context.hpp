#pragma once


#include "stdafx.hpp"


namespace Summer { namespace OptiX {


class Context final {
	public:
		CUDA::Context const*const context_cuda;

		OptixDeviceContext context;

	public:
		explicit Context(CUDA::Context const* context_cuda);
		~Context();
};


}}
