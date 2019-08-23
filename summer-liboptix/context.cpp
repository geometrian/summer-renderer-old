#include "context.hpp"

//This `#include` should occur in exactly one translation unit.
#include <optix_function_table_definition.h>


namespace Summer { namespace OptiX {


static bool _inited = false;


#ifdef BUILD_DEBUG
inline static void _optix_callback(unsigned int level, char const* tag, char const* message, void* /*cbdata*/) {
	printf("OptiX Callback (level %u (%s)):\n  \"%s\"\n",level,tag,message);

	if (level>2u);
	else throw;
}
#endif


Context::Context(CUDA::Context const* context_cuda) :
	context_cuda(context_cuda)
{
	if (!_inited) {
		assert_optix(optixInit());
		_inited = true;
	}

	OptixDeviceContextOptions options;
	#ifdef BUILD_DEBUG
		options.logCallbackFunction = _optix_callback;
		//options.logCallbackLevel = 2; //errors
		options.logCallbackLevel = 3; //warnings
		//options.logCallbackLevel = 4; //print everything
	#else
		options.logCallbackFunction = nullptr;
		options.logCallbackLevel = 0;
	#endif
	options.logCallbackData = nullptr;
	assert_optix(optixDeviceContextCreate(context_cuda->context, &options, &context));
}
Context::~Context() {
	assert_optix(optixDeviceContextDestroy(context));
}


}}
