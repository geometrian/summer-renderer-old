#pragma once


#include "stdafx.hpp"

#include "program-set.hpp"


namespace Summer { namespace OptiX {


class Pipeline;


class ShaderBindingTable final {
	friend class Pipeline;
	private:
		template<class EntryData>
		class __align__(OPTIX_SBT_RECORD_ALIGNMENT) _Entry final {
			friend class ShaderBindingTable;
			private:
				alignas(OPTIX_SBT_RECORD_ALIGNMENT) uint8_t _optix_header[OPTIX_SBT_RECORD_HEADER_SIZE];
			public:
				EntryData data;
		};

	public:
		template< class DataRaygen, class DataMiss, class DataHitOps >
		class Builder final {
			public:
				            std::pair<ProgramRaygen  const*,DataRaygen const*>  raygen;
				std::vector<std::pair<ProgramMiss    const*,DataMiss   const*>> miss;
				std::vector<std::pair<ProgramsHitOps const*,DataHitOps const*>> hitsops;

				size_t get_buffer_size() const {
					return
						               sizeof(_Entry<DataRaygen>) +
						miss.   size()*sizeof(_Entry<DataMiss  >) +
						hitsops.size()*sizeof(_Entry<DataHitOps>)
					;
				}
		};

	private:
		std::vector<OptixProgramGroup> _program_sets;

		OptixShaderBindingTable _sbt;

		CUDA::BufferGPUManaged _buffer_recs;

	private:
		template<class Data> static void _prepare_copy( ProgramSetBase const* program_set, Data const*__restrict src,_Entry<Data>*__restrict dst ) {
			assert_optix(optixSbtRecordPackHeader( program_set->program_set, &dst->_optix_header ));
			memcpy( &dst->data,src, sizeof(Data) );
		}
	public:
		template< class DataRaygen, class DataMiss, class DataHitOps >
		explicit ShaderBindingTable(Builder<DataRaygen,DataMiss,DataHitOps> const& builder) :
			_buffer_recs( builder.get_buffer_size() )
		{
			uint8_t* ptr;

			//Fill list of program sets used in the SBT for later use when creating pipelines.
			{
				std::set<OptixProgramGroup> tmp;
				                                         tmp.insert(builder.raygen.first->program_set);
				for (auto const& iter : builder.miss   ) tmp.insert(iter.          first->program_set);
				for (auto const& iter : builder.hitsops) {
					//	It can be convenient to have empty records in the SBT.  These don't have
					//		valid program sets, so skip those.
					if (iter.first!=nullptr) tmp.insert(iter.first->program_set);
				}

				std::copy( tmp.cbegin(),tmp.cend(), std::back_inserter(_program_sets) );
			}

			//Copy user-provided entries to CUDA buffer and fill OptiX headers.
			CUDA::BufferCPUManaged buffer_tmp( _buffer_recs.size );
			ptr = static_cast<uint8_t*>(buffer_tmp.ptr);
			{
				_Entry<DataRaygen>* rec_ptr = reinterpret_cast<_Entry<DataRaygen>*>(ptr);
				_prepare_copy( builder.raygen.first, builder.raygen.second,rec_ptr );

				ptr += sizeof(_Entry<DataRaygen>);
			}
			for (auto const& iter : builder.miss) {
				_Entry<DataMiss>* rec_ptr = reinterpret_cast<_Entry<DataMiss>*>(ptr);
				_prepare_copy( iter.first, iter.second,rec_ptr );

				ptr += sizeof(_Entry<DataMiss>);
			}
			for (auto const& iter : builder.hitsops) {
				_Entry<DataHitOps>* rec_ptr = reinterpret_cast<_Entry<DataHitOps>*>(ptr);
				if (iter.first!=nullptr) {
					_prepare_copy( iter.first, iter.second,rec_ptr );
				} else {
					memset( rec_ptr, 0x00, sizeof(_Entry<DataHitOps>) );
				}

				ptr += sizeof(_Entry<DataHitOps>);
			}

			//Upload that CUDA buffer to GPU
			_buffer_recs = buffer_tmp;

			//Fill SBT
			{
				memset(&_sbt,0x00,sizeof(OptixShaderBindingTable));

				ptr = static_cast<uint8_t*>(_buffer_recs.ptr);
				_sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(ptr);
				ptr += sizeof(_Entry<DataRaygen>);

				_sbt.missRecordBase          = reinterpret_cast<CUdeviceptr>(ptr);
				_sbt.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(_Entry<DataMiss>));
				_sbt.missRecordCount         = static_cast<unsigned int>(builder.miss.size()     );
				ptr += builder.miss.size() * sizeof(_Entry<DataMiss>);

				_sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(ptr);
				_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(_Entry<DataHitOps>));
				_sbt.hitgroupRecordCount         = static_cast<unsigned int>(builder.hitsops.size()    );
				//ptr += builder.hitsops.size() * sizeof(EntryHitgroup);
			}
		}
		~ShaderBindingTable() = default;
};


}}
