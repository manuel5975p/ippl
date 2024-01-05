#ifndef RESAMPLER_BASE_H
#define RESAMPLER_BASE_H
#include "Ippl.h"

#include <concepts>
template <typename T>
concept hasWeightMember = requires(T t) {
    { t.weight };
};

// concept hasMacrodummyMember = requires(T t){
//     {t.macrodummy};
// };

namespace ippl {
    enum QuantityType {
        intensive,
        extensive
    };
    template <class PLayout>
    class WeightedParticle;
    // clang-format off
    template <typename T, QuantityType QType>
    class MacroParticleAttrib : public ParticleAttrib<T> {
        ParticleAttrib<T>* correspondingWeightAttribute;

    public:
        using ParticleAttrib<T>::operator=;
        using typename ParticleAttrib<T>::execution_space;
        constexpr static QuantityType type() { return QType; }
        #define DefineMacroParticleReduction(fun, name, op, MPI_Op)      \
        T name() {                                                       \
            auto aview        = this->getView();                         \
            auto wview        = correspondingWeightAttribute->getView(); \
            T temp            = 0.0;                                     \
            using policy_type = Kokkos::RangePolicy<execution_space>;    \
            Kokkos::parallel_reduce(                                     \
                "fun", policy_type(0, *(this->localNum_mp)),             \
                KOKKOS_CLASS_LAMBDA(const size_t i, T& valL) {           \
                    T myVal = aview(i) * wview(i);                       \
                    op;                                                  \
                },                                                       \
                Kokkos::fun<T>(temp));                                   \
            T globaltemp = 0.0;                                          \
            Comm->allreduce(temp, globaltemp, 1, MPI_Op<T>());           \
            return globaltemp;                                           \
        }
        DefineMacroParticleReduction(Sum, sum, valL += myVal, std::plus)
        DefineMacroParticleReduction(Prod, mul, valL *= myVal, std::multiplies)
        DefineMacroParticleReduction(Min, min, if (myVal < valL) valL = myVal, std::less)
        DefineMacroParticleReduction(Prod, prod, valL *= myVal, std::multiplies)
        // T sum(){
        //     this->getView();
        // }
        template <typename PLayout>
        friend class WeightedParticle;
    };
    // clang-format on
    template <class PLayout>
    class WeightedParticle : public ParticleBase<PLayout> {
    public:
        using scalar    = typename PLayout::value_type;
        using size_type = detail::size_type;
        ParticleAttrib<scalar> weight;
        WeightedParticle(PLayout& pl)
            : ParticleBase<PLayout>(pl) {
            ParticleBase<PLayout>::addAttribute(weight);
        }
        void create(size_t nlocal) {
            ParticleBase<PLayout>::create(nlocal);
            auto vweight = weight.getView();
            Kokkos::parallel_for(
                ippl::getRangePolicy(vweight), KOKKOS_LAMBDA(size_t idx) { vweight(idx) = 1.0; });
        }
        template <typename T, QuantityType QType>
        void addAttribute(MacroParticleAttrib<T, QType>& pa) {
            ParticleBase<PLayout>::addAttribute(pa);
            pa.setParticleCount(*ParticleBase<PLayout>::getLocalNumPointer());
            pa.correspondingWeightAttribute = &weight;
        }
    };
    template <typename bunchType>
        requires(hasWeightMember<bunchType>)
    class ResamplerBase {
    public:
        virtual void operator()(bunchType& x) = 0;
        virtual ~ResamplerBase()              = default;
    };

}  // Namespace ippl
#endif