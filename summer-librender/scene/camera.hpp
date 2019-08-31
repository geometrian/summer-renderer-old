#pragma once


#include "../stdafx.hpp"

#include "framebuffer.hpp"


namespace Summer { namespace Scene {


class Camera final {
	public:
		enum class TYPE {
			LOOKAT
		} type;

		class LookAt final { public:
			Vec3f position;
			Vec3f center;
			Vec3f up;
		};
		union { LookAt lookat; };

		Framebuffer framebuffer;

		class InterfaceGPU final {
			public:
				TYPE type;

				union { LookAt lookat; };

				Framebuffer::InterfaceGPU framebuffer;

				__device__ Ray get_ray(Vec2u const& pixel) const;
		};

	public:
		Camera() = default;
		Camera(TYPE type, Vec2zu const& res,RenderSettings::LAYERS layers) :
			type(type), framebuffer(res,layers)
		{}
		~Camera() = default;

		InterfaceGPU get_interface() const {
			return { type, lookat, framebuffer.get_interface() };
		}
};


}}
