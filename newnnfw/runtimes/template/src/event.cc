#include <NeuralNetworks.h>

#include "event.h"

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent *event)
{
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent *event)
{
  delete event;
}
