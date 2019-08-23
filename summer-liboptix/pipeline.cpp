#include "pipeline.hpp"

#include "context.hpp"
#include "shader-binding-table.hpp"


namespace Summer { namespace OptiX {


Pipeline::Options::Options() {
	comp.usesMotionBlur = 1;
	comp.traversableGraphFlags = OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	comp.numPayloadValues = 2;
	comp.numAttributeValues = 2;
	#ifdef BUILD_DEBUG
	comp.exceptionFlags =
		OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
		OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
		OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_USER |
		OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_DEBUG
	;
	#else
	comp.exceptionFlags = OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE;
	#endif
	comp.pipelineLaunchParamsVariableName = "interface";

	link.maxTraceDepth = 31u;
	#ifdef BUILD_DEBUG
	link.debugLevel = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
	//link.debugLevel = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_FULL;
	#else
	link.debugLevel = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE;
	#endif
	link.overrideUsesMotionBlur = 0;
}


Pipeline::Pipeline(Context const* context_optix, Options const& options, ShaderBindingTable const* sbt) :
	context_optix(context_optix),
	options(options),
	sbt(sbt)
{
	assert_optix(optixPipelineCreate(
		context_optix->context,
		&options.comp, &options.link,
		sbt->_program_sets.data(), static_cast<unsigned int>(sbt->_program_sets.size()),
		nullptr, nullptr,
		&_pipeline
	));

	assert_optix(optixPipelineSetStackSize(
		_pipeline,
		//Stack sizes for callables, which we're not using.
		0u, 0u, 0u,
		//Maximum number of traversables, including transforms, touched during trace.
		3u
	));
}
Pipeline::~Pipeline() {
	assert_optix(optixPipelineDestroy(_pipeline));
}

void Pipeline::launch(CUDA::BufferGPUManaged* launch_interface, size_t const res[3]) const {
	assert_optix(optixLaunch(
		_pipeline,
		context_optix->context_cuda->stream,
		launch_interface->ptr_integral, launch_interface->size,
		&sbt->_sbt,
		static_cast<unsigned int>(res[0]), static_cast<unsigned int>(res[1]), static_cast<unsigned int>(res[2])
	));

	assert_cuda(cudaDeviceSynchronize());
}


}}
