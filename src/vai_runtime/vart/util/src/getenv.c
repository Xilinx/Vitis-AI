#include <stdio.h>
#include <stdlib.h>
const char* vitis_ai_getenv_s(const char* name) {
#if _WIN32
  size_t len = 0;
  char* ret = NULL;
  auto err = _dupenv_s(&ret, &len, name);

  if (err != 0) {
    fprintf(stderr, "cannot read env %s", name);
    abort();
  } else {
  }
  return ret;
#else
  return getenv(name);
#endif
}
