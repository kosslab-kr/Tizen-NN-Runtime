#ifndef __NNFW_STD_MEMORY_H__
#define __NNFW_STD_MEMORY_H__

#include <memory>

namespace nnfw
{

template <typename T, typename... Args> std::unique_ptr<T> make_unique(Args &&... args)
{
  // NOTE std::make_unique is missing in C++11 standard
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace nnfw

#endif // __NNFW_STD_MEMORY_H__
