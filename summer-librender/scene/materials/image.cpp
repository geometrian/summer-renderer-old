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
		SUMMER_CASE(FORMAT::sRGB8,   uchar3)
		SUMMER_CASE(FORMAT::sRGB8_A8,uchar4)
		SUMMER_CASE(FORMAT::DEPTH32F,float1)
		SUMMER_CASE(FORMAT::lRGB32F, float3)
		SUMMER_CASE(FORMAT::lRGBA32F,float4)
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
