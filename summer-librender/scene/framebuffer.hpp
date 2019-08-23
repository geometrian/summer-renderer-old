#pragma once


#include "../stdafx.hpp"

#include "texture.hpp"


namespace Summer { namespace Scene {


class Framebuffer final {
	public:
		Vec2zu const res;

		template<GLint fmt_internal> class Layer final {
			friend class Framebuffer;
			public:
				Texture2D<fmt_internal> texture;
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
				Layer<GL_RGBA8 > rgba;
				//Layer<GL_RGB8  > albedo;
				//Layer<GL_RGB32F> normal;
				//Layer<GL_R32F  > depth;

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
