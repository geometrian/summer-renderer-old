#include "camera.hpp"


namespace Summer { namespace Scene {


__device__ Ray Camera::InterfaceGPU::get_ray(Vec2f const& pixel) const {
	Vec2f res = Vec2f(framebuffer.res);
	Vec2f uv = pixel / res;
	float aspect = res.x / res.y;

	Vec3f nz = glm::normalize( lookat.center - lookat.position );
	Vec3f  x = glm::normalize(glm::cross( nz, lookat.up ));
	Vec3f  y = glm::cross(x,nz);

	Vec3f dir = glm::normalize(
		nz + ((uv.x-0.5f)*aspect)*x + (uv.y-0.5f)*y
	);

	return { lookat.position, dir };
}


}}
