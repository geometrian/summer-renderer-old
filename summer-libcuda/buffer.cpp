#include "buffer.hpp"


namespace Summer { namespace CUDA {


void BufferCPUBase::_set_data(BufferBase const& other) /*override*/ {
	assert_term(size==other.size,"Buffer size mismatch!");
	switch (other.type) {
		case TYPE::CPU_MANAGED: [[fallthrough]];
		case TYPE::CPU_WRAPPER:
			memcpy( ptr,other.ptr, size );
			break;
		case TYPE::GPU_MANAGED: [[fallthrough]];
		case TYPE::GPU_WRAPPER:
			assert_cuda(cudaMemcpy( ptr,other.ptr, size, cudaMemcpyKind::cudaMemcpyDeviceToHost ));
			break;
	}
}


void BufferGPUBase::_set_data(BufferBase const& other) /*override*/ {
	assert_term(size==other.size,"Buffer size mismatch!");
	switch (other.type) {
		case TYPE::CPU_MANAGED: [[fallthrough]];
		case TYPE::CPU_WRAPPER:
			assert_cuda(cudaMemcpy( ptr,other.ptr, size, cudaMemcpyKind::cudaMemcpyHostToDevice ));
			break;
		case TYPE::GPU_MANAGED: [[fallthrough]];
		case TYPE::GPU_WRAPPER:
			assert_cuda(cudaMemcpy( ptr,other.ptr, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice ));
			break;
	}
}


}}
