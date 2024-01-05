#include "SimpleThinningResampler.h"
#include <Kokkos_Random.hpp>
namespace ippl{
    template<typename bunchType, typename T, unsigned Dim>
        requires(hasWeightMember<bunchType>)
    void SimpleThinningResampler<bunchType, T, Dim>::operator()(bunchType& x){
        const size_t size = x.getLocalNum();
        auto wview = x.weight.getView();
        Kokkos::View<bool*> remove("SimpleThinningResampler::remove", x.getLocalNum());
        Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
        
        const T one_over_rem = 1.0 / remainingFraction_m;
        const T dp = 1 - remainingFraction_m;
        
        size_t destroyCount = 0;
        Kokkos::parallel_reduce(size, KOKKOS_LAMBDA(size_t idx, size_t& refDestroyCount){
            wview(idx) *= one_over_rem;
            auto state = random_pool.get_state();
            const bool destroy = state.drand(0.0, 1.0) < dp;
            remove(idx) = destroy;
            refDestroyCount += destroy;
            random_pool.free_state(state);
        }, destroyCount);
        Kokkos::fence();
        
        x.destroy(remove, destroyCount);
    }
}