#include <NeuralNetworks.h>

#include "memory.h"

int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd, size_t offset,
                                       ANeuralNetworksMemory **memory)
{
  *memory = new ANeuralNetworksMemory{};

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory *memory)
{
  delete memory;
}
