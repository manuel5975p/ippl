//
// Class ParticleAttrib
//   Templated class for all particle attribute classes.
//
//   This templated class is used to represent a single particle attribute.
//   An attribute is one data element within a particle object, and is
//   stored as a Kokkos::View. This class stores the type information for the
//   attribute, and provides methods to create and destroy new items, and
//   to perform operations involving this attribute with others.
//
//   ParticleAttrib is the primary element involved in expressions for
//   particles (just as BareField is the primary element there).  This file
//   defines the necessary templated classes and functions to make
//   ParticleAttrib a capable expression-template participant.
//
#include "Ippl.h"

#include "Communicate/DataTypes.h"

#include "Utility/IpplTimings.h"

namespace ippl {

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::create(size_type n) {
        size_type required = *(this->localNum_mp) + n;
        if (this->size() < required) {
            int overalloc = Comm->getDefaultOverallocation();
            this->realloc(required * overalloc);
        }
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::destroy(const hash_type& deleteIndex,
                                                   const hash_type& keepIndex,
                                                   size_type invalidCount) {
        // Replace all invalid particles in the valid region with valid
        // particles in the invalid region
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::destroy()", policy_type(0, invalidCount),
            KOKKOS_CLASS_LAMBDA(const size_t i) {
                dview_m(deleteIndex(i)) = dview_m(keepIndex(i));
            });
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::pack(const hash_type& hash) {
        auto size = hash.extent(0);
        if (buf_m.extent(0) < size) {
            int overalloc = Comm->getDefaultOverallocation();
            Kokkos::realloc(buf_m, size * overalloc);
        }

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::pack()", policy_type(0, size),
            KOKKOS_CLASS_LAMBDA(const size_t i) { buf_m(i) = dview_m(hash(i)); });
        Kokkos::fence();
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::unpack(size_type nrecvs) {
        auto size          = dview_m.extent(0);
        size_type required = *(this->localNum_mp) + nrecvs;
        if (size < required) {
            int overalloc = Comm->getDefaultOverallocation();
            this->resize(required * overalloc);
        }

        size_type count   = *(this->localNum_mp);
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::unpack()", policy_type(0, nrecvs),
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(count + i) = buf_m(i); });
        Kokkos::fence();
    }

    template <typename T, class... Properties>
    // KOKKOS_INLINE_FUNCTION
    ParticleAttrib<T, Properties...>& ParticleAttrib<T, Properties...>::operator=(T x) {
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::operator=()", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(i) = x; });
        return *this;
    }

    template <typename T, class... Properties>
    template <typename E, size_t N>
    // KOKKOS_INLINE_FUNCTION
    ParticleAttrib<T, Properties...>& ParticleAttrib<T, Properties...>::operator=(
        detail::Expression<E, N> const& expr) {
        using capture_type = detail::CapturedExpression<E, N>;
        capture_type expr_ = reinterpret_cast<const capture_type&>(expr);

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::operator=()", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(i) = expr_(i); });
        return *this;
    }

    template <typename T, class... Properties>
    template <typename Field, class PT, bool volumetricallyCorrect>
    void ParticleAttrib<T, Properties...>::scatter(
        Field& f, const ParticleAttrib<Vector<PT, Field::dim>, Properties...>& pp) const {
        constexpr unsigned Dim = Field::dim;
        using PositionType     = typename Field::Mesh_t::value_type;

        static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
        IpplTimings::startTimer(scatterTimer);
        using view_type = typename Field::view_type;
        view_type view  = f.getView();

        using mesh_type       = typename Field::Mesh_t;
        const mesh_type& mesh = f.get_mesh();

        using vector_type = typename mesh_type::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        // Scale factor accounting for volumetric density consistency
        typename vector_type::value_type spatial_scale = 1.0;
        if constexpr (volumetricallyCorrect)
            for (unsigned i = 0; i < Dim; i++) {
                spatial_scale *= invdx[i];
            }

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::scatter", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // find nearest grid point
                vector_type l                        = (pp(idx) - origin) * invdx + 0.5;
                
                Vector<int, Field::dim> index        = l;
                if(index[0] < view.extent(0) - nghost &&
                   index[1] < view.extent(1) - nghost &&
                   index[2] < view.extent(2) - nghost &&
                   index[0] >= nghost &&
                   index[1] >= nghost &&
                   index[2] >= nghost
                    ){
                Vector<PositionType, Field::dim> whi = l - index;
                Vector<PositionType, Field::dim> wlo = 1.0 - whi;

                Vector<size_t, Field::dim> args = index - lDom.first() + nghost;

                // scatter
                const value_type& val = dview_m(idx);
                detail::scatterToField(std::make_index_sequence<1 << Field::dim>{}, view, wlo, whi,
                                       args, val * spatial_scale);
                }
            });
        IpplTimings::stopTimer(scatterTimer);

        static IpplTimings::TimerRef accumulateHaloTimer = IpplTimings::getTimer("accumulateHalo");
        IpplTimings::startTimer(accumulateHaloTimer);
        f.accumulateHalo();
        IpplTimings::stopTimer(accumulateHaloTimer);
    }
    template <typename T, class... Properties>
    template <typename Field, class P2, bool volumetricallyCorrect>
    void ParticleAttrib<T, Properties...>::scatter(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp1,
        const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp2, T dt_scale) const {
        constexpr unsigned Dim = Field::dim;

        static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
        IpplTimings::startTimer(scatterTimer);
        using view_type = typename Field::view_type;
        view_type view  = f.getView();

        using mesh_type       = typename Field::Mesh_t;
        const mesh_type& mesh = f.get_mesh();

        using vector_type = typename mesh_type::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type dx     = mesh.getMeshSpacing();
        const vector_type origin = mesh.getOrigin();
        const vector_type invdx  = 1.0 / dx;
        // Scale factor accounting for volumetric density consistency
        typename vector_type::value_type spatial_scale = 1.0;
        if constexpr (volumetricallyCorrect)
            for (unsigned int d = 0; d < Dim; d++) {
                spatial_scale *= invdx[d];
            }

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim> lDom        = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::scatter", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // Compute offset to origin. Conversion to grid space coordinates happens later!
                vector_type from = (pp1(idx) - origin);
                vector_type to   = (pp2(idx) - origin);

                // val is charge (or other quantity)
                const value_type& val = dview_m(idx);
                detail::ZigzagScatterToField(std::make_index_sequence<1 << Field::dim>{}, view,
                                             from, to, dx, val * spatial_scale * dt_scale, lDom,
                                             nghost);
            });
        IpplTimings::stopTimer(scatterTimer);

        static IpplTimings::TimerRef accumulateHaloTimer = IpplTimings::getTimer("accumulateHalo");
        IpplTimings::startTimer(accumulateHaloTimer);
        f.accumulateHalo();
        IpplTimings::stopTimer(accumulateHaloTimer);
    }

    template <typename T, class... Properties>
    template <typename Field, typename P2>
    void ParticleAttrib<T, Properties...>::gather(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp) {
        constexpr unsigned Dim = Field::dim;
        using PositionType     = typename Field::Mesh_t::value_type;

        static IpplTimings::TimerRef fillHaloTimer = IpplTimings::getTimer("fillHalo");
        IpplTimings::startTimer(fillHaloTimer);
        f.fillHalo();
        IpplTimings::stopTimer(fillHaloTimer);

        static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");
        IpplTimings::startTimer(gatherTimer);
        const typename Field::view_type view = f.getView();

        using mesh_type       = typename Field::Mesh_t;
        const mesh_type& mesh = f.get_mesh();

        using vector_type = typename mesh_type::vector_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;
        //std::cout << "INVDX" << invdx << "\n";

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();
        //std::cout << "Particle count " << *(this->localNum_mp) << "\n";
        //assert(*(this->localNum_mp) == 1);
        auto this_view = dview_m;
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::gather", policy_type(0, *(this->localNum_mp)),
            KOKKOS_LAMBDA(const size_t idx) {
                // find nearest grid point
                //std::cout << idx << "\n";
                //std::cout << "Particle pos: " << pp(idx) << std::endl;
                vector_type l                        = (pp(idx) - origin) * invdx + 0.5;
                Vector<int, Field::dim> index        = l;
                //std::cout << "Access pos: " << l << "\n";
                //std::cout << "Access index: " << index << "\n";
                if(index[0] >= 0 && (size_t)index[0] < view.extent(0) &&
                   index[1] >= 0 && (size_t)index[1] < view.extent(1) &&
                   index[2] >= 0 && (size_t)index[2] < view.extent(2)){
                    Vector<PositionType, Field::dim> whi = l - index;
                    Vector<PositionType, Field::dim> wlo = 1.0 - whi;

                    Vector<size_t, Field::dim> args = index - lDom.first() + nghost;

                    // gather
                    this_view(idx) = detail::gatherFromField(std::make_index_sequence<1 << Field::dim>{},
                                                       view, wlo, whi, args);
                }
            });
        IpplTimings::stopTimer(gatherTimer);
    }
    template <typename T, class... Properties>
    template <typename Function, typename TParticleAttrib>
        requires(std::is_convertible_v<std::invoke_result_t<Function, typename TParticleAttrib::value_type>, T>)
    void ParticleAttrib<T, Properties...>::gatherExternalField(Function f, const TParticleAttrib& pp){
        static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");
        IpplTimings::startTimer(gatherTimer);

        Kokkos::parallel_for(
            "ParticleAttrib::gather", *(this->localNum_mp),
            KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // find nearest grid point
                T l = f(pp(idx));


                // gather
                dview_m(idx) += l;
            });
    }
    template <typename T, class... Properties>
    template <typename Field, typename P2>
    void ParticleAttrib<T, Properties...>::scatterVolumetricallyCorrect(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp) const {
        scatter<Field, P2, true>(f, pp);
    }
    //     scatter the data from this attribute onto the given Field, using
    //     the given Position attribute
    //     This performs zigzag deposition!
    template <typename T, class... Properties>
    template <typename Field, typename P2>
    void ParticleAttrib<T, Properties...>::scatterVolumetricallyCorrect(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp1,
        const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp2, T dt_scale) const {
        scatter<Field, P2, true>(f, pp1, pp2, dt_scale);
    }

    /*
     * Non-class function
     *
     */

    template <typename Attrib1, typename Field, typename Attrib2>
    inline void scatter(const Attrib1& attrib, Field& f, const Attrib2& pp) {
        attrib.scatter(f, pp);
    }

    template <typename Attrib1, typename Field, typename Attrib2>
    inline void gather(Attrib1& attrib, Field& f, const Attrib2& pp) {
        attrib.gather(f, pp);
    }

#define DefineParticleReduction(fun, name, op, MPI_Op)            \
    template <typename T, class... Properties>                    \
    T ParticleAttrib<T, Properties...>::name() {                  \
        T temp            = 0.0;                                  \
        using policy_type = Kokkos::RangePolicy<execution_space>; \
        Kokkos::parallel_reduce(                                  \
            "fun", policy_type(0, *(this->localNum_mp)),          \
            KOKKOS_CLASS_LAMBDA(const size_t i, T& valL) {        \
                T myVal = dview_m(i);                             \
                op;                                               \
            },                                                    \
            Kokkos::fun<T>(temp));                                \
        T globaltemp = 0.0;                                       \
        Comm->allreduce(temp, globaltemp, 1, MPI_Op<T>());        \
        return globaltemp;                                        \
    }

    DefineParticleReduction(Sum, sum, valL += myVal, std::plus)
    DefineParticleReduction(Max, max, if (myVal > valL) valL = myVal, std::greater)
    DefineParticleReduction(Min, min, if (myVal < valL) valL = myVal, std::less)
    DefineParticleReduction(Prod, prod, valL *= myVal, std::multiplies)
}  // namespace ippl
