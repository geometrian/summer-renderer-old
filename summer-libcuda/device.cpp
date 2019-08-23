#include "device.hpp"


namespace Summer { namespace CUDA {


Device::Device(size_t index) {
	//Forces CUDA to initialize itself, if it hasn't been.  When CUDA initializes, it creates a CUDA
	//	context for every device.
	assert_cuda(cudaFree(0));

	assert_term(index<get_count(),"Invalid device!");
	device = static_cast<CUdevice>(index);

	set_current();
	assert_cuda(cudaGetDeviceProperties(&properties,device));
}

void Device::set_current() const {
	assert_cuda(cudaSetDevice(device));
}

size_t Device::get_count() {
	int num;
	assert_cuda(cudaGetDeviceCount(&num));
	assert_term(num>=0,"Implementation error!");

	return static_cast<size_t>(num);
}


}}
