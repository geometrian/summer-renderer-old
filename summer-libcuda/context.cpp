#include "context.hpp"

#include "device.hpp"


namespace Summer { namespace CUDA {


Context::Context(Device const* device_cuda) {
	/*assert_cuda(static_cast<cudaError>(
		cuCtxCreate(&context,CU_CTX_SCHED_YIELD|CU_CTX_MAP_HOST,device_cuda->device)
	));*/

	device_cuda->set_current();
	assert_cuda(static_cast<cudaError>(
		cuCtxGetCurrent(&context)
	));

	assert_cuda(cudaStreamCreate(&stream));
}
Context::~Context() {
	assert_cuda(cudaStreamDestroy(stream));

	/*assert_cuda(static_cast<cudaError>(
		cuCtxDestroy(context)
	));*/
}


}}
