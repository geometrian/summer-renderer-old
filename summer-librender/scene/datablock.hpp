#pragma once


#include "../stdafx.hpp"


namespace Summer { namespace Scene {


class DataBlock final {
	public:
		struct {
			CUDA::BufferCPUManaged* cpu;
			CUDA::BufferGPUManaged* gpu;
		} data;

		class View final {
			public:
				DataBlock*const datablock;

				size_t const offset;
				size_t const stride;
				size_t const size;

			public:
				View(DataBlock* datablock, size_t offset,size_t stride,size_t size);
				~View() = default;
		};

		class AccessorBase {
			public:
				enum class TYPE {
					U8, S8, U16, S16, U32, S32, F32, F64,
					U8x2, S8x2, U16x2, S16x2, U32x2, S32x2, F32x2, F64x2,
					U8x3, S8x3, U16x3, S16x3, U32x3, S32x3, F32x3, F64x3,
					U8x4, S8x4, U16x4, S16x4, U32x4, S32x4, F32x4, F64x4,
					U8x2x2, S8x2x2, U16x2x2, S16x2x2, U32x2x2, S32x2x2, F32x2x2, F64x2x2,
					U8x3x3, S8x3x3, U16x3x3, S16x3x3, U32x3x3, S32x3x3, F32x3x3, F64x3x3,
					U8x4x4, S8x4x4, U16x4x4, S16x4x4, U32x4x4, S32x4x4, F32x4x4, F64x4x4
				} const type;

				View const*const view;

				size_t const offset;
				size_t const num_elements;

			protected:
				AccessorBase(TYPE type, View const* view, size_t offset,size_t num_elements);
			public:
				virtual ~AccessorBase() = default;

				uint8_t*               get_ptr_cpu() const;
				CUDA::Pointer<uint8_t> get_ptr_gpu() const;
		};
		template<size_t sizeof_elem> class AccessorSizedBase : public AccessorBase {
			protected:
				AccessorSizedBase(TYPE type, View const* view, size_t offset,size_t num_elements) :
					AccessorBase(type, view, offset,num_elements)
				{
					//assert_term((view->size-offset)%sizeof_elem==0,"Datablock view length does not match element stride!");
					assert_term((view->size-offset)/sizeof_elem>=num_elements,"Accessor larger than buffer view!");
				}
			public:
				virtual ~AccessorSizedBase() = default;
		};
		template<typename T> class Accessor final : public AccessorSizedBase<sizeof(T)> {
			public:
				#ifdef BUILD_DEBUG
				T* ptr_cpu;
				#endif

			private:
				static AccessorBase::TYPE _get_type();
			public:
				Accessor(View const* view, size_t offset,size_t num_elements) :
					AccessorSizedBase<sizeof(T)>(Accessor<T>::_get_type(),view,offset,num_elements)
				{
					#ifdef BUILD_DEBUG
					ptr_cpu = get_ptr_cpu_typed();
					#endif
				}
				virtual ~Accessor() = default;

				T* get_ptr_cpu_typed() const {
					uint8_t* ptr = AccessorBase::get_ptr_cpu();
					return reinterpret_cast<T*>(ptr);
				}
		};
		static_assert(sizeof(Vec3f)==3*sizeof(float),"Implementation error!");

	public:
		explicit DataBlock(std::vector<uint8_t> const& data);
		~DataBlock();

		void upload();
};


}}
