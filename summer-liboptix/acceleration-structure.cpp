#include "acceleration-structure.hpp"

#include "context.hpp"


namespace Summer { namespace OptiX {


OptixBuildInput& AccelerationStructure::BuilderTriangles::_add_mesh_triangles(BufferAccessor const& vertices) {
	_build_inputs.emplace_back();
	OptixBuildInput& build_input = _build_inputs.back();
	//memset(&build_input,0x00,sizeof(OptixBuildInput));

	build_input.type = OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	build_input.triangleArray.vertexBuffers = vertices.ptrs_arr; //One for each motion key
	build_input.triangleArray.numVertices   = static_cast<unsigned int>(vertices.count);
	build_input.triangleArray.vertexFormat = OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3;
	build_input.triangleArray.vertexStrideInBytes = static_cast<unsigned int>(vertices.stride);

	build_input.triangleArray.preTransform = reinterpret_cast<CUdeviceptr>(nullptr);

	build_input.triangleArray.flags = reinterpret_cast<unsigned int const*>(_geom_flags);

	build_input.triangleArray.numSbtRecords               = 1u;
	build_input.triangleArray.sbtIndexOffsetBuffer        = reinterpret_cast<CUdeviceptr>(nullptr); 
	build_input.triangleArray.sbtIndexOffsetSizeInBytes   = 0u; 
	build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0u; 

	build_input.triangleArray.primitiveIndexOffset = 0u;

	return build_input;
}

void AccelerationStructure::BuilderTriangles::add_mesh_triangles_basic      (BufferAccessor const& vertices                               ) {
	OptixBuildInput& build_input = _add_mesh_triangles(vertices);

	build_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(nullptr);
	build_input.triangleArray.numIndexTriplets = 0u;
	build_input.triangleArray.indexFormat = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	build_input.triangleArray.indexStrideInBytes = 0u;
}
void AccelerationStructure::BuilderTriangles::add_mesh_triangles_indexed_u16(BufferAccessor const& vertices, BufferAccessor const& indices) {
	OptixBuildInput& build_input = _add_mesh_triangles(vertices);

	build_input.triangleArray.indexBuffer = indices.ptrs_arr[0];
	build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.count/3);
	build_input.triangleArray.indexFormat = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
	build_input.triangleArray.indexStrideInBytes = static_cast<unsigned int>(indices.stride);
}
void AccelerationStructure::BuilderTriangles::add_mesh_triangles_indexed_u32(BufferAccessor const& vertices, BufferAccessor const& indices) {
	OptixBuildInput& build_input = _add_mesh_triangles(vertices);

	build_input.triangleArray.indexBuffer = indices.ptrs_arr[0];
	build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.count/3);
	build_input.triangleArray.indexFormat = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	build_input.triangleArray.indexStrideInBytes = static_cast<unsigned int>(indices.stride);
}

void AccelerationStructure::BuilderTriangles::finish() /*override*/ {
	assert_term(!_finished,"Already called `.finish()`!");
	DEBUG_ONLY(_finished = true;)
}


static unsigned int _offset = 0u;

AccelerationStructure::BuilderInstances::BuilderInstances() : _instances_gpu(nullptr) {
	_build_inputs.emplace_back();
}
AccelerationStructure::BuilderInstances::~BuilderInstances() {
	delete _instances_gpu;
}

void AccelerationStructure::BuilderInstances::add_instance(CUdeviceptr child_traversable, Mat3x4f const& transform) {
	_instances.emplace_back();
	OptixInstance& instance = _instances.back();

	float tmp[12] = {
		transform[0][0], transform[1][0], transform[2][0], transform[3][0],
		transform[0][1], transform[1][1], transform[2][1], transform[3][1],
		transform[0][2], transform[1][2], transform[2][2], transform[3][2],
	};
	memcpy(instance.transform,tmp,12*sizeof(float));

	instance.instanceId = static_cast<unsigned int>(_instances.size());

	instance.sbtOffset = _offset++;

	instance.visibilityMask = 0xFF;

	instance.flags = OptixInstanceFlags::OPTIX_INSTANCE_FLAG_NONE;

	instance.traversableHandle = child_traversable;
}

void AccelerationStructure::BuilderInstances::finish() /*override*/ {
	assert_term(!_finished,"Already called `.finish()`!");

	OptixBuildInput& build_input = _build_inputs.back();
	//memset(&build_input,0x00,sizeof(OptixBuildInput));

	build_input.type = OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_INSTANCES;

	_instances_gpu = new CUDA::BufferGPUManaged( _instances.size()*sizeof(OptixInstance) );
	*_instances_gpu = CUDA::BufferCPUWrapper(_instances);
	build_input.instanceArray.instances = _instances_gpu->ptr_integral;
	build_input.instanceArray.numInstances = static_cast<unsigned int>(_instances.size());

	build_input.instanceArray.aabbs = reinterpret_cast<CUdeviceptr>(nullptr);
	build_input.instanceArray.numAabbs = 0u;

	DEBUG_ONLY(_finished = true;)
}


void AccelerationStructure::_build(std::vector<OptixBuildInput> const& build_inputs) {
	OptixAccelBuildOptions opt_build;
	{
		opt_build.buildFlags =
			OptixBuildFlags::OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
			OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION
		;
		opt_build.operation = OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD;
		opt_build.motionOptions = { 1, OptixMotionFlags::OPTIX_MOTION_FLAG_NONE, 0.0f,1.0f };
	}

	OptixAccelBufferSizes build_buf_sizes;
	assert_optix(optixAccelComputeMemoryUsage(
		context_optix->context,
		&opt_build,
		build_inputs.data(), static_cast<unsigned int>(build_inputs.size()),
		&build_buf_sizes
	));

	CUDA::BufferGPUManaged buf_size( sizeof(uint64_t) );

	CUDA::BufferGPUManaged buf_tmp( build_buf_sizes.tempSizeInBytes   );
	CUDA::BufferGPUManaged buf_out( build_buf_sizes.outputSizeInBytes );

	CUstream stream = context_optix->context_cuda->stream;

	uint64_t compactable_size;
	{
		OptixAccelEmitDesc emitDesc;
		emitDesc.type   = OptixAccelPropertyType::OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = buf_size.ptr_integral;

		assert_optix(optixAccelBuild(
			context_optix->context,
			stream,

			&opt_build,

			build_inputs.data(), static_cast<unsigned int>(build_inputs.size()),
			buf_tmp.ptr_integral, buf_tmp.size,
			buf_out.ptr_integral, buf_out.size,

			&handle,

			&emitDesc, 1
		));
		assert_cuda(cudaDeviceSynchronize());

		CUDA::BufferCPUWrapper wrap( sizeof(uint64_t),&compactable_size );
		wrap = buf_size;
	}

	_buffer = new CUDA::BufferGPUManaged( static_cast<size_t>(compactable_size) );
	{
		assert_optix(optixAccelCompact(
			context_optix->context,
			stream,

			handle,

			_buffer->ptr_integral, _buffer->size,

			&handle
		));
		assert_cuda(cudaDeviceSynchronize());
	}
}

AccelerationStructure::AccelerationStructure(Context const* context_optix, _BuilderBase const& builder) :
	context_optix(context_optix)
{
	assert_term(builder._finished,"Did not call `.finish()` on builder!");
	_build(builder._build_inputs);
}
AccelerationStructure::~AccelerationStructure() {
	delete _buffer;
}


}}
