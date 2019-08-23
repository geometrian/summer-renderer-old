#pragma once


#include "stdafx.hpp"


namespace Summer { namespace OptiX {


class Context;
class ShaderBindingTable;


class Pipeline final {
	public:
		Context const*const context_optix;

		class Options final {
			public:
				OptixPipelineCompileOptions comp;
				OptixPipelineLinkOptions    link;

			public:
				Options();
				~Options() = default;
		};
		Options const options;

		ShaderBindingTable const*const sbt;

	private:
		OptixPipeline _pipeline;

	public:
		explicit Pipeline(Context const* context_optix, Options const& options, ShaderBindingTable const* sbt);
		~Pipeline();

		void launch(CUDA::BufferGPUManaged* launch_interface, size_t const res[3]) const;
};


}}
