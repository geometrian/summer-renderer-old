#include "framebuffer.hpp"


namespace Summer { namespace Scene {


#if 0
pixels = new uint8_t[res[1]*res[0]*4];
for (size_t i=0;i<res[0];++i) {
	for (size_t j=0;j<res[1];++j) {
		uint8_t brightness = (i%8>=4)^(j%8>=4) ? 0x00 : 0xFF;
		pixels[4*(j*res[0]+i)  ] = brightness;
		pixels[4*(j*res[0]+i)+1] = brightness;
		pixels[4*(j*res[0]+i)+2] = brightness;
		pixels[4*(j*res[0]+i)+3] = 128;
	}
}
#endif


template<Image2D::FORMAT fmt> Framebuffer::Layer<fmt>::Layer(Vec2zu const& res) :
	texture(res),
	pbo(res,ImageFormatInfo<fmt>::sizeof_texel)
{
	texture.set_gpu_opengl<fmt>();
}

template<Image2D::FORMAT fmt> void Framebuffer::Layer<fmt>::_update_texture() {
	texture.copy_pbo_to_opengl(&pbo);
}

template class Framebuffer::Layer<Image2D::FORMAT::sRGB8   >;
template class Framebuffer::Layer<Image2D::FORMAT::sRGB8_A8>;
template class Framebuffer::Layer<Image2D::FORMAT::DEPTH32F>;
template class Framebuffer::Layer<Image2D::FORMAT::lRGB32F >;
template class Framebuffer::Layer<Image2D::FORMAT::lRGBA32F>;


void Framebuffer::Layers::_launch_prepare(CUDA::Context const* context_cuda) {
	rgba.pbo.map(context_cuda);
}
void Framebuffer::Layers::_launch_finish (                                 ) {
	rgba.pbo.unmap();

	rgba._update_texture();
}


Framebuffer::Framebuffer(Vec2zu const& res) :
	res(res), layers(res) DEBUG_ONLY(COMMA _mapped(false))
{}

void Framebuffer::launch_prepare(CUDA::Context const* context_cuda) {
	assert_term(!_mapped,"Already mapped!");
	layers._launch_prepare(context_cuda);
	DEBUG_ONLY(_mapped = true;)
}
void Framebuffer::launch_finish (                                 ) {
	assert_term(_mapped,"Not mapped!");
	layers._launch_finish();
	DEBUG_ONLY(_mapped = false;)
}

void Framebuffer::draw() const {
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(
		0.0, static_cast<double>(res[0]),
		0.0, static_cast<double>(res[1]),
		-1.0, 1.0
	);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_FRAMEBUFFER_SRGB);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, layers.rgba.texture.data.gpu_gl.handle);

	#if 1
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f,0.0f); glVertex2f(0.0f,                      0.0f                      );
	glTexCoord2f(1.0f,0.0f); glVertex2f(static_cast<float>(res[0]),0.0f                      );
	glTexCoord2f(1.0f,1.0f); glVertex2f(static_cast<float>(res[0]),static_cast<float>(res[1]));
	glTexCoord2f(0.0f,1.0f); glVertex2f(0.0f,                      static_cast<float>(res[1]));
	glEnd();
	#endif

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_FRAMEBUFFER_SRGB);

	/*CUDA::BufferCPUManaged pixels_cpu( layers.rgba.pbo.size );
	layers.rgba.pbo.copy_to_buffer(context_cuda,&pixels_cpu);

	glClear(GL_COLOR_BUFFER_BIT);

	glDrawPixels(static_cast<GLsizei>(res[0]),static_cast<GLsizei>(res[1]),GL_RGBA,GL_UNSIGNED_BYTE,pixels_cpu.ptr);*/
}


}}
