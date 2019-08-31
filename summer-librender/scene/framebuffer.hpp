#pragma once


#include "../stdafx.hpp"

#include "../render-settings.hpp"

#include "materials/texture.hpp"


namespace Summer { namespace Scene {


class Framebuffer final {
	public:
		Vec2zu const res;

		template<Image2D::FORMAT fmt> class Layer final {
			friend class Framebuffer;
			public:
				Texture2D texture;
				CUDA::WritePixelBufferObject2D pbo;

			public:
				explicit Layer(Vec2zu const& res);
				~Layer() = default;

			private:
				void _update_texture();
		};
		class Layers final {
			friend class Framebuffer;
			public:
				//Layers used for output and postprocessing.  The sampling weights/count layer is
				//	mandatory; all others are optional (albeit there should be at-least one other
				//	or-else rendering doesn't make sense).

				//Samples weights and counts.  The weights are, for each pixel, the total weight of
				//	all samples affecting it.  Divide the other layers by this to reconstruct their
				//	value.  The counts are, for each pixel, the total number of all samples
				//	affecting it.
				Layer<Image2D::FORMAT::VEC2_F32  >* sampling_weights_and_count;

				//Result of the chosen lighting integrator
				Layer<Image2D::FORMAT::CIEXYZ_F32>* lighting_integration;
				//Postprocessing
				Layer<Image2D::FORMAT::lRGB_F32  >* lighting_tmp;
				Layer<Image2D::FORMAT::lRGB_F32  >* lighting_final;
				//Estimate of variance in lighting buffer; used to guide adaptive sampling
				Layer<Image2D::FORMAT::SCALAR_F32>* lighting_samples_variance;

				//Transparency of scene
				Layer<Image2D::FORMAT::SCALAR_F32>* scene_coverage;
				//Distance and depth
				Layer<Image2D::FORMAT::SCALAR_F32>* scene_distance;
				Layer<Image2D::FORMAT::SCALAR_F32>* scene_depth;
				//Normals
				Layer<Image2D::FORMAT::VEC3_F32  >* scene_normals_geometric;
				Layer<Image2D::FORMAT::VEC3_F32  >* scene_normals_shading;
				//3D motion vectors
				Layer<Image2D::FORMAT::VEC3_F32  >* scene_motion_vectors;
				//2D barycentric coordinates (the third coordinate can be reconstructed trivially)
				Layer<Image2D::FORMAT::VEC2_F32  >* scene_triangle_barycentric_coordinates;
				//Texture coordinates
				Layer<Image2D::FORMAT::VEC2_F32  >* scene_texture_coordinates;
				//Albedo
				Layer<Image2D::FORMAT::lRGB_A_F32>* scene_albedo;
				//TODO: IOR
				//Fresnel (s-polarized, p-polarized)
				Layer<Image2D::FORMAT::VEC2_F32  >* scene_fresnel_term;

				//Acceleration structure cost (nodes traversed, primitives tested)
				Layer<Image2D::FORMAT::VEC2_F32  >* stats_accelstruct;
				//Depth complexity
				Layer<Image2D::FORMAT::SCALAR_F32>* stats_depthcomplexity;
				//Photon density
				Layer<Image2D::FORMAT::SCALAR_F32>* stats_photondensity;

			public:
				Layers(Vec2zu const& res, RenderSettings::LAYERS layers);
				~Layers();

				//Clears the layers that are rendered (and not e.g. the output of postprocessing).
				void clear_rendered(CUDA::Context const* context_cuda);

			private:
				void   _map(CUDA::Context const* context_cuda);
				void _unmap(                                 );
		} layers;
		DEBUG_ONLY(private: bool _mapped; public:)

		class InterfaceGPU final {
			public:
				Vec2zu res;

				struct {
					CUDA::Pointer<Vec2f> sampling_weights_and_count;

					CUDA::Pointer<Vec3f> lighting_integration;
					CUDA::Pointer<float> lighting_samples_variance;

					CUDA::Pointer<float> scene_coverage;
					CUDA::Pointer<float> scene_distance;
					CUDA::Pointer<float> scene_depth;
					CUDA::Pointer<Vec3f> scene_normals_geometric;
					CUDA::Pointer<Vec3f> scene_normals_shading;
					CUDA::Pointer<Vec3f> scene_motion_vectors;
					CUDA::Pointer<Vec2f> scene_triangle_barycentric_coordinates;
					CUDA::Pointer<Vec2f> scene_texture_coordinates;
					CUDA::Pointer<Vec4f> scene_albedo;
					CUDA::Pointer<Vec2f> scene_fresnel_term;

					CUDA::Pointer<Vec2f> stats_accelstruct;
					CUDA::Pointer<float> stats_depthcomplexity;
					CUDA::Pointer<float> stats_photondensity;
				} layers;
		};

	public:
		Framebuffer(Vec2zu const& res, RenderSettings::LAYERS layers);
		~Framebuffer();

		InterfaceGPU get_interface() const;

		void launch_prepare(CUDA::Context const* context_cuda);
		void launch_finish (                                 );

		void process_and_draw(RenderSettings const& render_settings);
};


}}
