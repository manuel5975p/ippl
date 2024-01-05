#ifndef NUMBERCONSERVATIVE_THINNING_RESAMPLER_BASE_H
#define NUMBERCONSERVATIVE_THINNING_RESAMPLER_BASE_H
#include "ResamplerBase.h"
#include "Ippl.h"
#include <functional>
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
    class EnergyConservativeThinningResampler {
    
    size_t selectionCount_m; // Selection count
    std::function<Kokkos::View<T*>(bunchType&)> energyViewGenerator; //
    public:
        constexpr static unsigned dim = Dim;
        EnergyConservativeThinningResampler(size_t selectionCount, auto energy_expression) : selectionCount_m(selectionCount) {
            energyViewGenerator = [energy_expression](bunchType& bunch){
                Kokkos::View<T*> ret("ret", bunch.getLocalNum());
                Kokkos::parallel_for(bunch.getLocalNum(), KOKKOS_LAMBDA(size_t idx){
                    ret(idx) = energy_expression(idx);
                });
                return ret;
            };
        }
        void operator()(bunchType& x);
    };

}  // Namespace ippl
#include "EnergyConservativeThinningResampler.hpp"
#endif