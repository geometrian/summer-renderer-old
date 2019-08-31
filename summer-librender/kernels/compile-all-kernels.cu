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


//#include "generic-forward.cu"

#include "integrate-lightnone.cu"
#include "integrate-pathtrace.cu"

#include "../scene/materials/material.cu"

#include "../scene/camera.cu"
