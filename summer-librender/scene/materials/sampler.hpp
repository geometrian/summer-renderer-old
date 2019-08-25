#pragma once


#include "../../stdafx.hpp"


namespace Summer { namespace Scene {


class Sampler final {
	public:
		//Note: the filter is not broken down by whether the texture is minified or magnified.  It
		//	doesn't make much sense mathematically, and combining them allows for performance
		//	improvement (CUDA texture descriptors only allow one).  The main impact should be that
		//	textures that are magnified point-sampled but minified linear are not possible.
		enum class TYPE_FILTER : uint32_t {
			MIP_NONE_TEX_CLOSEST    = 0b001'01u,
			MIP_NONE_TEX_LINEAR     = 0b001'10u,
			MIP_CLOSEST_TEX_CLOSEST = 0b010'01u,
			MIP_CLOSEST_TEX_LINEAR  = 0b010'10u,
			MIP_LINEAR_TEX_CLOSEST  = 0b100'01u,
			MIP_LINEAR_TEX_LINEAR   = 0b100'10u,

			MSK_TEX_CLOSEST = 0b000'01u,
			MSK_TEX_LINEAR  = 0b000'10u,

			MSK_MIP_NONE    = 0b001'00u,
			MSK_MIP_CLOSEST = 0b010'00u,
			MSK_MIP_LINEAR  = 0b100'00u
		};
		TYPE_FILTER type_filter;

		enum class TYPE_EDGE {
			CLAMP, REPEAT, REPEAT_MIRROR
		};
		class TypeEdges final { public:
			TYPE_EDGE s;
			TYPE_EDGE t;

			TypeEdges();
			~TypeEdges() = default;
		} type_edges;

	public:
		Sampler();
		~Sampler() = default;

	private:
		void _get_texture_descrs_helper(cudaTextureDesc* descr, bool conv_sRGB) const;
	public:
		void get_texture_descrs(cudaTextureDesc descrs[2]) const;
};


}}
