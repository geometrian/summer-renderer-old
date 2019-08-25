#include "sampler.hpp"


namespace Summer { namespace Scene {


Sampler::TypeEdges::TypeEdges() :
	s(TYPE_EDGE::REPEAT),
	t(TYPE_EDGE::REPEAT)
{}


Sampler::Sampler() :
	type_filter(TYPE_FILTER::MIP_NONE_TEX_LINEAR)
{}

void Sampler::_get_texture_descrs_helper(cudaTextureDesc* descr, bool conv_sRGB) const {
	switch (type_edges.s) {
		case TYPE_EDGE::CLAMP:         descr->addressMode[0]=cudaTextureAddressMode::cudaAddressModeClamp;  break;
		case TYPE_EDGE::REPEAT:        descr->addressMode[0]=cudaTextureAddressMode::cudaAddressModeWrap;   break;
		case TYPE_EDGE::REPEAT_MIRROR: descr->addressMode[0]=cudaTextureAddressMode::cudaAddressModeMirror; break;
		nodefault;
	}
	switch (type_edges.t) {
		case TYPE_EDGE::CLAMP:         descr->addressMode[1]=cudaTextureAddressMode::cudaAddressModeClamp;  break;
		case TYPE_EDGE::REPEAT:        descr->addressMode[1]=cudaTextureAddressMode::cudaAddressModeWrap;   break;
		case TYPE_EDGE::REPEAT_MIRROR: descr->addressMode[1]=cudaTextureAddressMode::cudaAddressModeMirror; break;
		nodefault;
	}

	switch (type_filter) {
		case TYPE_FILTER::MIP_NONE_TEX_CLOSEST:    [[fallthrough]];
		case TYPE_FILTER::MIP_CLOSEST_TEX_CLOSEST: [[fallthrough]];
		case TYPE_FILTER::MIP_LINEAR_TEX_CLOSEST:
			descr->filterMode = cudaTextureFilterMode::cudaFilterModePoint;
			break;
		case TYPE_FILTER::MIP_NONE_TEX_LINEAR:    [[fallthrough]];
		case TYPE_FILTER::MIP_CLOSEST_TEX_LINEAR: [[fallthrough]];
		case TYPE_FILTER::MIP_LINEAR_TEX_LINEAR:
			descr->filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
			break;
		nodefault;
	}

	descr->readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
	descr->sRGB     = conv_sRGB ? 1 : 0;

	descr->borderColor[0] = 1.0f;
	descr->borderColor[1] = 0.0f;
	descr->borderColor[2] = 1.0f;
	descr->borderColor[3] = 1.0f;

	descr->normalizedCoords = 1;

	descr->maxAnisotropy   = 1u;
	descr->mipmapLevelBias = 0.0f;
	switch (type_filter) {
		case TYPE_FILTER::MIP_NONE_TEX_CLOSEST: [[fallthrough]];
		case TYPE_FILTER::MIP_NONE_TEX_LINEAR:
			descr->mipmapFilterMode    = cudaTextureFilterMode::cudaFilterModePoint;
			descr->minMipmapLevelClamp = 0.0f;
			descr->maxMipmapLevelClamp = 0.0f;
			break;
		case TYPE_FILTER::MIP_CLOSEST_TEX_CLOSEST: [[fallthrough]];
		case TYPE_FILTER::MIP_CLOSEST_TEX_LINEAR:
			descr->mipmapFilterMode    = cudaTextureFilterMode::cudaFilterModePoint;
			descr->minMipmapLevelClamp = 0.0f;
			descr->maxMipmapLevelClamp = std::numeric_limits<float>::infinity();
			break;
		case TYPE_FILTER::MIP_LINEAR_TEX_CLOSEST: [[fallthrough]];
		case TYPE_FILTER::MIP_LINEAR_TEX_LINEAR:
			descr->mipmapFilterMode    = cudaTextureFilterMode::cudaFilterModeLinear;
			descr->minMipmapLevelClamp = 0.0f;
			descr->maxMipmapLevelClamp = std::numeric_limits<float>::infinity();
			break;
		nodefault;
	}
}
void Sampler::get_texture_descrs(cudaTextureDesc descrs[2]) const {
	_get_texture_descrs_helper(descrs,  false);
	_get_texture_descrs_helper(descrs+1,true );
}


}}
