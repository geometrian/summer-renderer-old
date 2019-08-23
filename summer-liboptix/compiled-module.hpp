#pragma once


#include "stdafx.hpp"

#include "pipeline.hpp"


namespace Summer { namespace OptiX {


//Encapsulates a set of programs that are compiled for use in a particular pipeline.  Analogous to
//	an object file in the C++ compilation model.
class CompiledModule final {
	public:
		Context const*const context_optix;

		OptixModule module;

	public:
		CompiledModule(Context const* context_optix, Pipeline::Options const& pipeline_opts, std::string const& ptx_str);
		~CompiledModule();
};


}}
