# Summer Renderer

<table style="width:100%"><tr>
	<td style="padding:0px"><img src="https://graphics.geometrian.com/summer/gallery/1.png"/></td>
	<td style="padding:0px"><img src="https://graphics.geometrian.com/summer/gallery/2.png"/></td>
	<td style="padding:0px"><img src="https://graphics.geometrian.com/summer/gallery/3.png"/></td>
</tr></table>



## A Physical GPU-Based Renderer

Summer Renderer is a physically-based GPU renderer built around the following goals:

- **Be physically correct** (up to the Newtonian ray-optics approximation).  No hacks that compromise realism, even if they make the renderer easier for artists to use.  For example, Summer is spectral only.  All integrators are unbiased, or at-least consistent.
- **Be easy-to-understand and extensible**.  Summer's code is simple, clean, modular, object-oriented C++.  Keeping the code clean is even-more-important than maximum performance.  Nevertheless:
- **Be blazingly fast**.  Summer is implemented on the GPU in a fairly intelligent way, and aims to support the most-advanced importance-sampling techniques and path-space integrators.
- **Be free**.  Summer's code is free—both as-in beer and as-in freedom.  You can modify it, use it commercially, and do prettymuch anything except claim you invented it (you know, *lie*).

Summer is in the early stages and does not currently fully live up to these goals.  The website is (will be) [graphics.geometrian.com/summer](https://graphics.geometrian.com/summer/).  Contributions are welcome, and regardless, hopefully progress will come very soon.

Summer wasn't named in any special way.  At a conference in Strasbourg in the summer of 2019, I was trying to think of a name for a new GPU renderer.  The season, summer, has the right connotation: intensity, power, magic, happiness, association with light, etc., and I figured it would serve as a decent reference point while I thought up a better name—but I couldn't in the end.



## Components

Summer has several key components:

- **Core Libraries**: These implement the actual renderer and helper functionality.  The other components are thin wrappers around it.
- **Standalone Renderer**: This is a standalone renderer which can produce a rendered image or interactive preview from an input GLTF file.
- **Blender Plugin** (*Currently Nonexistent*): This is a renderer for [Blender](https://www.blender.org/).



## Building

Build is standard CMake.  However, unfortunately, there are a lot of dependencies.  It would be nice to remove some of these.

The core renderer is built deeply on OptiX, and therefore also CUDA.  All vector math is done via GLM.

- **[OptiX 7](https://developer.nvidia.com/optix)**: Download from the NVIDIA developer website.  You will need to be a registered developer with NVIDIA (for some reason).
- **[CUDA 10+](https://developer.nvidia.com/cuda-zone)**: Download from the NVIDIA developer website.
- **[GLM](https://glm.g-truc.net/)**: Download the headers.  You can use the "FindGLM.cmake" file produced by building it, but [one is also included](cmake/FindGLM.cmake), which looks in reasonable places.

The standalone renderer visualizes the result in OpenGL.  Models are loaded from GLTF files.

- **[GLFW 3+](https://www.glfw.org/)**: GLFW is used to make a window and OpenGL context.  Download it from the GLFW site.  Building from source would be preferable, since some additional features newer than the latest release can be used.
- **[GLAD](https://github.com/Dav1dde/glad)**: GLAD is used to make the OpenGL context have reasonable features in it.  The easiest way may be to use [their webservice](https://glad.dav1d.de/) to generate the required header/source file.
- **[TinyGLTF](https://github.com/syoyo/tinygltf)**: Download from the GitHub.  It is header-only, but still needs to be someplace where it can be found.  See also the [FindTinyGLTF.cmake](cmake/FindTinyGLTF.cmake) file I wrote for it.

There are some additional dependencies which should be bundled and found with your CMake assuming you have a sane development environment.

- **OpenGL**
- **Threading**



## Contributing

Contributions are welcome, especially those that improve the build system or fix bugs.  If you'd like to get involved with development that affects more code, it would probably be a good idea to [email me](https://geometrian.com/contact/index.php) to talk it over first, so that we can coordinate code changes better.
