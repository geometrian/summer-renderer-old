#pragma once


#include "../stdafx.hpp"

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
				Layer<Image2D::FORMAT::sRGB8_A8 > rgba;
				//Layer<Image2D::FORMAT::lRGB32F  > albedo;
				//Layer<Image2D::FORMAT::VEC3_32F > normal;
				//Layer<Image2D::FORMAT::DEPTH32F > depth;

			public:
				explicit Layers(Vec2zu const& res) : rgba(res) {}//,albedo(res),normal(res),depth(res) {}
				~Layers() = default;

			private:
				void _launch_prepare(CUDA::Context const* context_cuda);
				void _launch_finish (                                 );
		} layers;
		DEBUG_ONLY(private: bool _mapped; public:)

		class InterfaceGPU final {
			public:
				Vec2zu res;

				CUDA::Pointer<uint32_t> rgba;
		};

	public:
		explicit Framebuffer(Vec2zu const& res);
		~Framebuffer() = default;

		InterfaceGPU get_interface() const {
			assert_term(_mapped,"Must map buffers by calling `.launch_prepare(...)` first!");
			return { res, layers.rgba.pbo.mapped_ptr };
		}

		void launch_prepare(CUDA::Context const* context_cuda);
		void launch_finish (                                 );

		void draw() const;
};


}}
