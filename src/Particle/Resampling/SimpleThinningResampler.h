#ifndef SIMPLETHINNING_RESAMPLER_BASE_H
#define SIMPLETHINNING_RESAMPLER_BASE_H
#include "ResamplerBase.h"
#include "Ippl.h"
namespace ippl {
    /**
     * @brief "Simple Thinning" strategy macroparticle resampler https://doi.org/10.1016/j.cpc.2021.107826
     * 
     * @tparam bunchType 
     * @tparam T 
     * @tparam Dim 
     */
    template <typename bunchType, typename T, unsigned Dim>
        requires(hasWeightMember<bunchType>)
    class SimpleThinningResampler {
    
    T remainingFraction_m; // Deletion probability of a particle in thinnee bunch.
    public:
        constexpr static unsigned dim = Dim;
        SimpleThinningResampler(T remainingFraction) : remainingFraction_m(remainingFraction) {

        }
        void operator()(bunchType& x);
    };

}  // Namespace ippl
#include "SimpleThinningResampler.hpp"
#endif