#include "NumberConservativeThinningResampler.h"
#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
namespace ippl{
    template<typename bunchType, typename T, unsigned Dim>
        requires(hasWeightMember<bunchType>)
    void NumberConservativeThinningResampler<bunchType, T, Dim>::operator()(bunchType& x){
        const size_t local_size = x.getLocalNum();
        auto wview = x.weight.getView();
        T wsum = x.weight.sum();
        T wsum_only_over_local = 0;
        Kokkos::parallel_reduce(local_size, KOKKOS_LAMBDA(size_t idx, T& wsum_ref){
            wsum_ref += wview(idx);
        }, wsum_only_over_local);
                
        Kokkos::View<T*> cumulativeProbabilities("NumberConservativeThinningResampler::cumulativeProbabilities", local_size);
        Kokkos::View<unsigned int*> selectionCount("NumberConservativeThinningResampler::remove", local_size);
        Kokkos::View<bool*> remove("NumberConservativeThinningResampler::remove", local_size);
        Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
        Kokkos::fence(); //Wait for wsum_only_over_local

        const size_t local_selection_count = this->selectionCount_m * wsum_only_over_local / wsum; //uuuuuuhh yes

        T scan_dummy = 0.0; //Input for parallel scan. But we already know it's going to be = 1.
        Kokkos::parallel_scan("Cumulative selection weights scan", local_size,
            KOKKOS_LAMBDA(size_t idx, T& partial_sum, bool is_final) {
            partial_sum += wview(idx) / wsum_only_over_local;
            if(is_final) cumulativeProbabilities(idx) = partial_sum;
        }, scan_dummy);
        (void)scan_dummy;
        Kokkos::parallel_for(local_selection_count, KOKKOS_LAMBDA(size_t){
            auto s = random_pool.get_state();
            size_t picked_index = s.urand64(local_size);
            Kokkos::atomic_add(&selectionCount(picked_index), 1u);
            random_pool.free_state(s);
        });
        Kokkos::fence();

        size_t destroyCount = 0;
        const T normalization_factor = wsum_only_over_local / this->selectionCount_m;
        Kokkos::parallel_reduce(local_size, KOKKOS_LAMBDA(size_t idx, size_t& refDestroyCount){
            if(selectionCount(idx)){
                remove(idx) = false;
                wview(idx) = selectionCount(idx) * normalization_factor; 
            }
            else{
                remove(idx) = true;
                refDestroyCount++;
            }
        }, destroyCount);
        Kokkos::fence();
        
        x.destroy(remove, destroyCount);
    }
}