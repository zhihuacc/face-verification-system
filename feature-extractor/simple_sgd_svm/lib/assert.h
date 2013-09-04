
#ifndef ASSERT_H
#define ASSERT_H 1

#include <iostream>
#include <cstdlib>

#define assertfail(msg) do { \
  std::cerr << "(" << __FILE__ << ":" << __LINE__ << ") " \
            << msg << std::endl; ::exit(10); } while(0)

#define assert(expr) \
  do { if (!(expr)) assertfail("Assertion failed: " << #expr); } while(0)

#endif
