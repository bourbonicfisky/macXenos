// Provides the STB image-write implementation for the lean conformance target.
// In the full build this symbol comes from trace_dump.cc; the conformance
// executable deliberately doesn't link that TU, so define it here once.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
