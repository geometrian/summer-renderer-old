#pragma once


#include "stdafx.hpp"


namespace Summer {


class RenderSettings final {
	public:
		//Integrator used for computing lighting output
		enum class LIGHTING_INTEGRATOR {
			//None (lighting output layer will be unavailable)
			NONE,

			//Old-style lighting components
			AMBIENT_OCCLUSION,
			DIRECT_LIGHTING_UNSHADOWED,
			SHADOWS,

			//Unbiased integrators over incomplete path-space
			DIRECT_LIGHTING,
			WHITTED,
			COOK,

			//Unbiased integrators over full path-space (except possibly some δ interactions)
			PATH_TRACING,
			LIGHT_TRACING,
			BIDIRECTIONAL_PATH_TRACING,
			METROPOLIS_LIGHT_TRANSPORT,

			//Consistent integrators over full path-space (except possibly some δ interactions)
			PHOTONMAPPING,

			//Consistent integrators over full path-space
			VERTEX_CONNECTION_MERGING
		};

		//Output layers
		enum class LAYERS : uint32_t {
			SAMPLING_WEIGHTS                       = 0b00000000000000'000'0000000000'000'01u,
			SAMPLING_COUNT                         = 0b00000000000000'000'0000000000'000'10u,

			LIGHTING_RAW                           = 0b00000000000000'000'0000000000'001'00u,
			LIGHTING_POSTPROCESSED                 = 0b00000000000000'000'0000000000'010'00u,
			LIGHTING_VARIANCE                      = 0b00000000000000'000'0000000000'100'00u,

			SCENE_COVERAGE                         = 0b00000000000000'000'0000000001'000'00u,
			SCENE_DISTANCE                         = 0b00000000000000'000'0000000010'000'00u,
			SCENE_DEPTH                            = 0b00000000000000'000'0000000100'000'00u,
			SCENE_NORMALS_GEOMETRIC                = 0b00000000000000'000'0000001000'000'00u,
			SCENE_NORMALS_SHADING                  = 0b00000000000000'000'0000010000'000'00u,
			SCENE_MOTION_VECTORS                   = 0b00000000000000'000'0000100000'000'00u,
			SCENE_TRIANGLE_BARYCENTRIC_COORDINATES = 0b00000000000000'000'0001000000'000'00u,
			SCENE_TEXTURE_COORDINATES              = 0b00000000000000'000'0010000000'000'00u,
			SCENE_ALBEDO                           = 0b00000000000000'000'0100000000'000'00u,
			SCENE_FRESNEL_TERM                     = 0b00000000000000'000'1000000000'000'00u,

			STATISTIC_ACCEL_OPS                    = 0b00000000000000'001'0000000000'000'00u,
			STATISTIC_DEPTH_COMPLEXITY             = 0b00000000000000'010'0000000000'000'00u,
			STATISTIC_PHOTON_DENSITY               = 0b00000000000000'100'0000000000'000'00u,

			MSK_SAMPLING                           = 0b00000000000000'000'0000000000'000'11u,
			MSK_LIGHTING                           = 0b00000000000000'000'0000000000'111'00u,
			MSK_SCENE                              = 0b00000000000000'000'1111111111'000'00u,
			MSK_STATISTIC                          = 0b00000000000000'111'0000000000'000'00u
		};

		LIGHTING_INTEGRATOR lighting_integrator;

		LAYERS layer_primary_output;

		size_t index_scene;
		size_t index_camera;

		float time_range[2];

		#if 0
		enum class SAMPLING {
			//Constant number of samples
			CONSTANT,
			//Adaptive number of samples per-(sub)pixel
			ADAPTIVE,

			//Progressively add samples per-(sub)pixel indefinitely
			CONSTANT_PROGRESSIVE,
			//Progressively add samples per-(sub)pixel indefinitely, denoising the output each time
			CONSTANT_PROGRESSIVE_DENOISE,
			//Adaptively add samples per-(sub)pixel indefinitely
			ADAPTIVE_PROGRESSIVE,
			//Adaptively add samples per-(sub)pixel indefinitely, denoising the output each time
			ADAPTIVE_PROGRESSIVE_DENOISE
		};

		class PostProcessStep final {
			public:
				enum class TYPE {
					DENOISE//,
					//BLOOM
				} type;
		};
		std::vector<PostProcessStep> postprocessing;

		enum class RECONSTRUCTION {
			BOX,
			MITCHELL_NETRAVALI
		};

		struct {
			bool progressive;
			bool adaptive;

			SAMPLING type;
		} sampling;
		#endif
};


}
