#include <jni.h>
#include "NitroCosSimOnLoad.hpp"

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return margelo::nitro::nitrocossim::initialize(vm);
}
