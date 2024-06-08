#include "Except.h"

namespace except
{
    void react()
    {
        try
        {
            throw;
        }
        catch (...)
        {
            std::cerr << "Handling UB\n";
        }
    }
}