#include "image.hpp"


namespace Summer { namespace Scene {


Image2D::Image2D(FORMAT format, Vec2zu const& res) :
	format(format), res(res)
{
	data.gpu = nullptr;
}
Image2D::~Image2D() {
	if (data.gpu!=nullptr) {
		assert_cuda(cudaFreeArray(data.gpu));
	}
}

void Image2D::upload() {
	size_t sizeof_texel;
	cudaChannelFormatDesc channel_descr;
	switch (format) {
		#define SUMMER_CASE(FMT,FMTCUDA)\
			case FMT:\
				channel_descr = cudaCreateChannelDesc<FMTCUDA>();\
				sizeof_texel = ImageFormatInfo<FMT>::sizeof_texel;\
				break;

		SUMMER_CASE(FORMAT::SCALAR_F32,  float1)
		SUMMER_CASE(FORMAT::VEC2_F32,    float2)
		SUMMER_CASE(FORMAT::VEC3_F32,    float3)

		SUMMER_CASE(FORMAT::CIEXYZ_F32,  float3)
		SUMMER_CASE(FORMAT::CIEXYZ_A_F32,float4)

		SUMMER_CASE(FORMAT::lRGB_F32,    float3)
		SUMMER_CASE(FORMAT::lRGB_A_F32,  float4)
		SUMMER_CASE(FORMAT::sRGB_U8,     uchar3)
		SUMMER_CASE(FORMAT::sRGB_A_U8,   uchar4)

		#undef SUMMER_CASE
		nodefault;
	}

	assert_cuda(cudaMallocArray(
		&data.gpu,
		&channel_descr,
		res[0], res[1]
	));
      
	size_t pitch = res[0] * sizeof_texel;
	assert_cuda(cudaMemcpy2DToArray(
		data.gpu,
		0, 0,
		data.cpu.data(),
		pitch, pitch, res[1],
		cudaMemcpyKind::cudaMemcpyHostToDevice
	));
}


}}
