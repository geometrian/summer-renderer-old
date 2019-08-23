#pragma once


#include "stdafx.hpp"


namespace Summer { namespace CUDA {


class BufferBase {
	public:
		enum class TYPE {
			CPU_MANAGED,
			CPU_WRAPPER,

			GPU_MANAGED,
			GPU_WRAPPER
		};
		TYPE const type;

		union {
			void*       ptr;
			CUdeviceptr ptr_integral;
		};
		size_t const size;

	protected:
		BufferBase(TYPE type, size_t size) : type(type),size(size) {}
	public:
		virtual ~BufferBase() = default;

	protected:
		virtual void _set_data(BufferBase const& other) = 0;
	public:
		BufferBase& operator=(BufferBase const& other) { _set_data(other); return *this; }
};


class BufferCPUBase : public BufferBase {
	protected:
		BufferCPUBase(TYPE type, size_t size) : BufferBase(type,size) {}
	public:
		virtual ~BufferCPUBase() override = default;

	protected:
		virtual void _set_data(BufferBase const& other) override;
	public:
		BufferCPUBase& operator=(BufferBase const& other) { _set_data(other); return *this; }
};

class BufferGPUBase : public BufferBase {
	protected:
		BufferGPUBase(TYPE type, size_t size) : BufferBase(type,size) {}
	public:
		virtual ~BufferGPUBase() override = default;

	protected:
		virtual void _set_data(BufferBase const& other) override;
	public:
		BufferGPUBase& operator=(BufferBase const& other) { _set_data(other); return *this; }
};


class BufferCPUManaged final : public BufferCPUBase {
	public:
		BufferCPUManaged(size_t size                  ) : BufferCPUBase(TYPE::CPU_MANAGED,size) {
			ptr = new uint8_t[size];
		}
		BufferCPUManaged(size_t size, void const* data) : BufferCPUManaged(size) {
			memcpy( ptr,data, size );
		}
		BufferCPUManaged(size_t size, CUdeviceptr data) : BufferCPUManaged(size) {
			assert_cuda(cudaMemcpy( ptr,reinterpret_cast<void const*>(data), size, cudaMemcpyDeviceToHost ));
		}
		template<typename T> BufferCPUManaged(std::vector<T> const& data) : BufferCPUManaged(data.size()*sizeof(T),data.data()) {}
		explicit BufferCPUManaged(BufferBase const& other) : BufferCPUManaged(other.size) {
			*this = other;
		}
		virtual ~BufferCPUManaged() override {
			delete[] static_cast<uint8_t*>(ptr);
		}

		BufferCPUManaged& operator=(BufferBase const& other) { _set_data(other); return *this; }
};

class BufferCPUWrapper final : public BufferCPUBase {
	public:
		BufferCPUWrapper(size_t size, void* data) : BufferCPUBase(TYPE::CPU_WRAPPER,size) {
			ptr = data;
		}
		template<typename T> BufferCPUWrapper(std::vector<T>& data) : BufferCPUWrapper(data.size()*sizeof(T),data.data()) {}
		explicit BufferCPUWrapper(BufferCPUManaged const& other) : BufferCPUWrapper(other.size,other.ptr) {}
		explicit BufferCPUWrapper(BufferCPUWrapper const& other) = default;
		virtual ~BufferCPUWrapper() override = default;

		BufferCPUWrapper& operator=(BufferBase const& other) { _set_data(other); return *this; }
};

class BufferGPUManaged final : public BufferGPUBase {
	public:
		BufferGPUManaged(size_t size                  ) : BufferGPUBase(TYPE::GPU_MANAGED,size) {
			assert_cuda(cudaMalloc(&ptr,size));
		}
		BufferGPUManaged(size_t size, void const* data) : BufferGPUManaged(size) {
			assert_cuda(cudaMemcpy( ptr,data, size, cudaMemcpyKind::cudaMemcpyHostToDevice ));
		}
		BufferGPUManaged(size_t size, CUdeviceptr data) : BufferGPUManaged(size) {
			assert_cuda(cudaMemcpy( ptr,reinterpret_cast<void const*>(data), size, cudaMemcpyKind::cudaMemcpyDeviceToDevice ));
		}
		template<typename T> BufferGPUManaged(std::vector<T> const& data) : BufferGPUManaged(data.size()*sizeof(T),data.data()) {}
		explicit BufferGPUManaged(BufferBase const& other) : BufferGPUManaged(other.size) {
			*this = other;
		}
		virtual ~BufferGPUManaged() override {
			assert_cuda(cudaFree(ptr));
		}

		BufferGPUManaged& operator=(BufferBase const& other) { _set_data(other); return *this; }
};

class BufferGPUWrapper final : public BufferGPUBase {
	public:
		BufferGPUWrapper(size_t size, CUdeviceptr data) : BufferGPUBase(TYPE::GPU_WRAPPER,size) {
			ptr_integral = data;
		}
		explicit BufferGPUWrapper(BufferGPUManaged const& other) : BufferGPUWrapper(other.size,other.ptr_integral) {}
		explicit BufferGPUWrapper(BufferGPUWrapper const& other) = default;
		virtual ~BufferGPUWrapper() override = default;

		BufferGPUWrapper& operator=(BufferBase const& other) { _set_data(other); return *this; }
};


}}
