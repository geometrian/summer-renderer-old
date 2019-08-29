/*
This file `#include`s all the bottom-level CUDA modules (the "translation units", by-analogy with
C++) so that when this file is compiled, everything is compiled into a single PTX file.

We have to do this since OptiX does not support proper linking: compiled program modules can
apparently only come from a single PTX file, and a single PTX file can apparently only be produced
from compilation of a single ".cu" file.  We could separate the ".cu" code by integrator, since only
one integrator is needed at a time, but this scarcely helps: most of the code is shared, and we
still have to link most of the implementation from separate files anyway.

This is messy and error-prone, but until OptiX stops breaking CUDA's compilation model, something of
this ilk is inevitable.
*/


#include "miss-color.cu"

#include "raygen-forward.cu"

#include "render-albedo.cu"
#include "render-normals.cu"
#include "render-pathtrace.cu"
#include "render-texcs.cu"
#include "render-tri-bary.cu"

#include "../scene/materials/material.cu"
