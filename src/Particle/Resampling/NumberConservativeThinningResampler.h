#ifndef NUMBERCONSERVATIVE_THINNING_RESAMPLER_BASE_H
#define NUMBERCONSERVATIVE_THINNING_RESAMPLER_BASE_H
#include "ResamplerBase.h"
#include "Ippl.h"
namespace ippl {
    /**
     * @brief "Number-Conservative Thinning" strategy macroparticle resampler https://doi.org/10.1016/j.cpc.2021.107826
     * 
     * @tparam bunchType 
     * @tparam T 
     * @tparam Dim 
     */
    template <typename bunchType, typename T, unsigned Dim>
        requires(hasWeightMember<bunchType>)
    class NumberConservativeThinningResampler {
    
    size_t selectionCount_m; // Selection count
    public:
        constexpr static unsigned dim = Dim;
        NumberConservativeThinningResampler(size_t selectionCount) : selectionCount_m(selectionCount) {

        }
        void operator()(bunchType& x);
    };

}  // Namespace ippl
#include "NumberConservativeThinningResampler.hpp"
#endif