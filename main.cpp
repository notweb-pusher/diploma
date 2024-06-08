#include "NeuralNet.h"
#include "Except.h"

int main()
{
    try
    {
        runMNIST();
    } catch (...)
    {
        except::react();
    }
    return 0;
}
