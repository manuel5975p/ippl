//
// Class FDTDSolver
//   Finite Differences Time Domain electromagnetic solver.
//
// Copyright (c) 2022, Sonali Mayani, PSI, Villigen, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

#include <array>

#include "Types/Vector.h"

#include "Field/Field.h"

#include "FDTDSolver.h"
#include <cinttypes>
#include <optional>
#include "Field/HaloCells.h"
#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"
template <typename T>
KOKKOS_INLINE_FUNCTION T squaredNorm(const ippl::Vector<T, 3>& a) {
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}
template <typename T>
KOKKOS_INLINE_FUNCTION auto sq(T x) {
    return x * x;
}
template <typename... Args>
KOKKOS_INLINE_FUNCTION void castToVoid(Args&&... args) {
    (void)(std::tuple<Args...>{args...});
}
#define CAST_TO_VOID(...) castToVoid(__VA_ARGS__)
template <typename T, unsigned Dim>
KOKKOS_INLINE_FUNCTION T dot_prod(const ippl::Vector<T, Dim>& a, const ippl::Vector<T, Dim>& b) {
    T ret = 0.0;
    for (unsigned i = 0; i < Dim; i++) {
        ret += a[i] * b[i];
    }
    return ret;
}

enum axis_aligned_occlusion : unsigned int {
    NONE   = 0u,
    AT_MIN = 1u,
    AT_MAX = 2u,
};
template <typename T>
struct axis_aligned_occlusion_maker {
    using type = axis_aligned_occlusion;
};
template <typename T>
using axis_aligned_occlusion_maker_t = typename axis_aligned_occlusion_maker<T>::type;

KOKKOS_INLINE_FUNCTION axis_aligned_occlusion operator|=(axis_aligned_occlusion& a, axis_aligned_occlusion b) {
    a = (axis_aligned_occlusion)(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
    return a;
}
KOKKOS_INLINE_FUNCTION axis_aligned_occlusion& operator|=(axis_aligned_occlusion& a, unsigned int b) {
    a = (axis_aligned_occlusion)(static_cast<unsigned int>(a) | b);
    return a;
}
KOKKOS_INLINE_FUNCTION axis_aligned_occlusion& operator&=(axis_aligned_occlusion& a, axis_aligned_occlusion b) {
    a = (axis_aligned_occlusion)(static_cast<unsigned int>(a) & static_cast<unsigned int>(b));
    return a;
}
KOKKOS_INLINE_FUNCTION axis_aligned_occlusion& operator&=(axis_aligned_occlusion& a, unsigned int b) {
    a = (axis_aligned_occlusion)(static_cast<unsigned int>(a) & b);
    return a;
}

template<typename _scalar, unsigned _main_axis, unsigned... _side_axes>
struct first_order_abc{
    using scalar = _scalar;
    constexpr static unsigned main_axis = _main_axis;
    constexpr static unsigned side_axes[] = {_side_axes...};
    ippl::Vector<scalar, 3> hr_m;
    int sign;
    scalar beta0;
    scalar beta1;
    scalar beta2;
    scalar beta3;
    scalar beta4;
    first_order_abc() = default;
    KOKKOS_FUNCTION first_order_abc(ippl::Vector<scalar, 3> hr, scalar c, scalar dt, int _sign) : hr_m(hr), sign(_sign) {
        beta0 = (c * dt - hr_m[main_axis]) / (c * dt + hr_m[main_axis]);
        beta1 = 2.0 * hr_m[main_axis] / (c * dt + hr_m[main_axis]);
        beta2 = -1.0;
        beta3 = beta1;
        beta4 = beta0;
    }
    template<typename view_type, typename Coords>
    KOKKOS_INLINE_FUNCTION auto operator()(const view_type& a_n, const view_type& a_nm1,const view_type& a_np1, const Coords& c)const -> typename view_type::value_type{
        using value_t = typename view_type::value_type;
        
        value_t ret = beta0 * (ippl::apply_with_offset<view_type, Coords, ~0u>(a_nm1, c, sign) + apply_with_offset<view_type, Coords, main_axis>(a_np1, c, sign))
                    + beta1 * (ippl::apply_with_offset<view_type, Coords, ~0u>(a_n, c, sign) + apply_with_offset<view_type, Coords, main_axis>(a_n, c, sign))
                    + beta2 * (ippl::apply_with_offset<view_type, Coords, main_axis>(a_nm1, c, sign));
        return ret;
    }
};
template<typename _scalar, unsigned _main_axis, unsigned... _side_axes>
struct second_order_abc{
    using scalar = _scalar;
    constexpr static unsigned main_axis = _main_axis;
    constexpr static unsigned side_axes[] = {_side_axes...};
    ippl::Vector<scalar, 3> hr_m;
    int sign;
    scalar gamma0;
    scalar gamma1;
    scalar gamma2;
    scalar gamma3;
    scalar gamma4;
    scalar gamma5;
    scalar gamma6;
    scalar gamma7;
    scalar gamma8;
    scalar gamma9;
    scalar gamma10;
    scalar gamma11;
    scalar gamma12;
    second_order_abc() = default;

    KOKKOS_FUNCTION second_order_abc(ippl::Vector<scalar, 3> hr, scalar c, scalar dt, int _sign) : hr_m(hr), sign(_sign){
        gamma0 = (c * dt - hr[main_axis]) / (c * dt + hr[main_axis]); 
        gamma1 = hr[main_axis] * (2.0 - sq(c * dt / hr_m[side_axes[1]]) - sq(c * dt / hr_m[side_axes[0]]));
        gamma2 = -1.0;
        gamma3 = gamma1;
        gamma4 = gamma0;
        gamma5 = sq(c * dt / hr_m[side_axes[0]]) * hr_m[main_axis] / (2.0 * (c * dt + hr_m[main_axis]));
        gamma6 = gamma5;
        gamma7 = sq(c * dt / hr_m[side_axes[1]]) * hr_m[main_axis] / (2.0 * (c * dt + hr_m[main_axis]));
        gamma8 = gamma7;
        gamma9 = gamma6;
        gamma10 = gamma0;
        gamma11 = gamma8;
        gamma12 = gamma8;
    }
    template<typename view_type, typename Coords>
    KOKKOS_INLINE_FUNCTION auto operator()(const view_type& a_n, const view_type& a_nm1,const view_type& a_np1, const Coords& c)const -> typename view_type::value_type{
        using value_t = typename view_type::value_type;
        value_t ret(0.0);
        ret += ippl::apply(a_nm1, c) * gamma0 + ippl::apply(a_n, c) * gamma1;
        {
            Coords acc = c;
            acc[main_axis] += sign;
            ret += gamma2 * ippl::apply(a_nm1, acc)
             + gamma3 * ippl::apply(a_n, acc)
             + gamma4 * ippl::apply(a_np1, acc);
        }
        {
            Coords acc = c;
            acc[main_axis] += sign;
            acc[side_axes[0]] += 1;
            ret += gamma5 * ippl::apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[main_axis] += sign;
            acc[side_axes[0]] -= 1;
            ret += gamma6 * ippl::apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[main_axis] += sign;
            acc[side_axes[1]] += 1;
            ret += gamma7 * ippl::apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[main_axis] += sign;
            acc[side_axes[1]] -= 1;
            ret += gamma8 * ippl::apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[side_axes[0]] += 1;
            ret += gamma9 * ippl::apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[side_axes[0]] -= 1;
            ret += gamma10 * ippl::apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[side_axes[1]] += 1;
            ret += gamma11 * ippl::apply(a_n, acc);
        }
        {
            Coords acc = c;
            acc[side_axes[1]] -= 1;
            ret += gamma12 * ippl::apply(a_n, acc);
        }

        return ret;
    }
};
KOKKOS_INLINE_FUNCTION auto variadic_min(auto x){
    return x;
}
KOKKOS_INLINE_FUNCTION auto variadic_min(auto x, auto y, auto... xs){
    return std::min(x, variadic_min(y, xs...));
}

template <typename... index_and_extent_types>
KOKKOS_INLINE_FUNCTION size_t boundary_distance(index_and_extent_types&&... x){
    std::tuple<index_and_extent_types...> tp{x...};
    constexpr size_t hsize = sizeof...(index_and_extent_types) / 2;
    auto difference_minimizer = KOKKOS_LAMBDA<size_t... Index>(const std::index_sequence<Index...>& seq){
        //size_t min_to_0 = std::min({std::get<Index>(tp)...});
        size_t min_to_0 = variadic_min(std::get<Index>(tp)...);
        //size_t min_to_extent = std::min({(std::get<Index + hsize>(tp) - std::get<Index>(tp) - 1)...});
        size_t min_to_extent = variadic_min((std::get<Index + hsize>(tp) - std::get<Index>(tp) - 1)...);
        (void)seq;
        return std::min(min_to_0, min_to_extent);
    };
    return difference_minimizer(std::make_index_sequence<sizeof...(index_and_extent_types) / 2>{});
}

template <typename index_type, unsigned Dim>
KOKKOS_INLINE_FUNCTION ippl::Vector<axis_aligned_occlusion, Dim> boundary_occlusion_of(
    size_t boundary_distance, const ippl::Vector<index_type, Dim> _index,
    const ippl::Vector<index_type, Dim> _extents) {
        using Kokkos::min;
    /*constexpr size_t Dim = std::tuple_size_v<std::tuple<extent_types...>>;

    constexpr auto get_array = []<typename... Ts>(Ts&&... x) {
        return ippl::Vector<size_t, sizeof...(x)>{static_cast<size_t>(x)...};
    };*/

    ippl::Vector<size_t, Dim> index   = _index;
    ippl::Vector<size_t, Dim> extents = _extents;
    ippl::Vector<axis_aligned_occlusion, Dim> ret_array;

    size_t minimal_distance_to_zero             = index[0];
    size_t minimal_distance_to_extent_minus_one = extents[0] - index[0] - 1;
    ret_array[0] = (axis_aligned_occlusion)(index[0] == boundary_distance);
    ret_array[0] |= (axis_aligned_occlusion)(index[0] == (extents[0] - 1 - boundary_distance)) << 1;
    for (size_t i = 1; i < Dim; i++) {
        minimal_distance_to_zero = min(minimal_distance_to_zero, index[i]);
        minimal_distance_to_extent_minus_one =
            min(minimal_distance_to_extent_minus_one, extents[i] - index[i] - 1);
        ret_array[i] = (axis_aligned_occlusion)(index[i] == boundary_distance);
        ret_array[i] |= (axis_aligned_occlusion)(index[i] == (extents[i] - 1 - boundary_distance))
                        << 1;
    }
    bool behindboundary = minimal_distance_to_zero < boundary_distance
                          || minimal_distance_to_extent_minus_one < boundary_distance;
    if (behindboundary) {
        for (size_t i = 0; i < Dim; i++) {
            ret_array[i] = (axis_aligned_occlusion)0;
        }
    }
    return ret_array;
}
template<typename _scalar, class PLayout>
_scalar bunch_energy(const Bunch<_scalar, PLayout>& bantsch){
    using scalar = _scalar;
    scalar ret = 0;
    auto gamma_beta_view = bantsch.gamma_beta.getView();
    auto mass_view = bantsch.mass.getView();
    Kokkos::parallel_reduce(bantsch.getLocalNum(), KOKKOS_LAMBDA(size_t i, _scalar& ref){
        using Kokkos::sqrt;
        ippl::Vector<_scalar, 3> gbi = gamma_beta_view(i);
        gbi *= ippl::detail::Scalar<scalar>(mass_view(i));
        scalar total_energy = mass_view(i) * mass_view(i) + gbi.squaredNorm();
        ref += sqrt(total_energy);
    }, ret);
    return ret;
}
namespace ippl {

    template <typename Tfields, unsigned Dim, class M, class C>
    FDTDSolver<Tfields, Dim, M, C>::FDTDSolver(Field_t* charge, VField_t* current, VField_t* E,
                                               VField_t* B, scalar timestep, size_t pcount, VField_t* rad, bool seed_) : radiation(rad), pl(charge->getLayout(), charge->get_mesh()), bunch(pl), particle_count(pcount) {
        // set the rho and J fields to be references to charge and current
        // since charge and current deposition will happen at each timestep
        rhoN_mp = charge;
        JN_mp   = current;

        // same for E and B fields
        En_mp = E;
        Bn_mp = B;

        // initialize the time-step size
        this->dt = timestep;

        // set the seed flag
        this->seed = seed_;

        // call the initialization function
        initialize();
    }
    template<typename T>
    auto sqr(T x){return x * x;};
    template <typename Tfields, unsigned Dim, class M, class C>
    FDTDSolver<Tfields, Dim, M, C>::~FDTDSolver(){};

    template <typename Tfields, unsigned Dim, class M, class C>
    void FDTDSolver<Tfields, Dim, M, C>::solve() {
        // physical constant
        // using scalar;// = typename Tfields::BareField_t::value_type::
        constexpr scalar c        = 1.0;  // 299792458.0; 
        constexpr scalar mu0      = 1.0;  // 1.25663706212e-6;
        constexpr scalar epsilon0 = 1.0 / (c * c * mu0);

        // finite differences constants
        scalar a1 = 2.0
           * (1.0 - sqr(c * dt / hr_m[0]) - sqr(c * dt / hr_m[1])
                  - sqr(c * dt / hr_m[2]));
        scalar a2 = sqr(c * dt / hr_m[0]);  // a3 = a2
        scalar a4 = sqr(c * dt / hr_m[1]);  // a5 = a4
        scalar a6 = sqr(c * dt / hr_m[2]);  // a7 = a6
        scalar a8 = sqr(c * dt)/* / (hr_m[0] * hr_m[1] * hr_m[2])*/;

        // 1st order absorbing boundary conditions constants
        scalar beta0[3] = {(c * dt - hr_m[0]) / (c * dt + hr_m[0]),
                           (c * dt - hr_m[1]) / (c * dt + hr_m[1]),
                           (c * dt - hr_m[2]) / (c * dt + hr_m[2])};
        scalar beta1[3] = {scalar(2) * hr_m[0] / (c * dt + hr_m[0]), scalar(2) * hr_m[1] / (c * dt + hr_m[1]),
                           scalar(2) * hr_m[2] / (c * dt + hr_m[2])};
        scalar beta2[3] = {-scalar(1), -scalar(1), -scalar(1)};
        
        

        //static_assert(std::is_invocable_v<ippl::Vector<scalar, 3>, int>);
        // preliminaries for Kokkos loops (ghost cells and views)
        //phiNp1_m         = scalar(0);
        //aNp1_m           = scalar(0);
        ANp1_m           = scalar(0);
        AN_m.fillHalo();
        ANm1_m.fillHalo();
        //aN_m.fillHalo();
        //aNm1_m.fillHalo();
        //phiN_m.fillHalo();
        //phiNm1_m.fillHalo();

        //auto view_phiN   = phiN_m.getView();
        //auto view_phiNm1 = phiNm1_m.getView();
        //auto view_phiNp1 = phiNp1_m.getView();

        
        //auto view_aN   = aN_m.getView();
        //auto view_aNm1 = aNm1_m.getView();
        //auto view_aNp1 = aNp1_m.getView();

        auto view_AN   = AN_m.getView();
        auto view_ANm1 = ANm1_m.getView();
        auto view_ANp1 = ANp1_m.getView();


        #define _bc first_order_abc<scalar,
        _bc 0, 1, 2> abc_x[] = {_bc 0, 1, 2> (hr_m, c, dt,  1), _bc 0, 1, 2>(hr_m, c, dt, -1)};
        _bc 1, 0, 2> abc_y[] = {_bc 1, 0, 2> (hr_m, c, dt,  1), _bc 1, 0, 2>(hr_m, c, dt, -1)};
        _bc 2, 0, 1> abc_z[] = {_bc 2, 0, 1> (hr_m, c, dt,  1), _bc 2, 0, 1>(hr_m, c, dt, -1)};

        auto view_rhoN = this->rhoN_mp->getView();
        auto view_JN   = this->JN_mp->getView();

        const int nghost_A   = AN_m.getNghost();
        const auto& ldom     = layout_mp->getLocalNDIndex();

        //aNm1_m.fillHalo();
        

        //JN_mp->fillHalo();
        // compute scalar potential and vector potential at next time-step
        // first, only the interior points are updated
        // then, if the user has set a seed, the seed is added via TF/SF boundaries
        // (TF/SF = total-field/scattered-field technique)
        // finally, absorbing boundary conditions are imposed
        /*Kokkos::parallel_for(
            "Scalar potential update", ippl::getRangePolicy(view_phiN, nghost_phi),
            KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                // global indices
                const int ig = i + ldom[0].first() - nghost_phi;
                const int jg = j + ldom[1].first() - nghost_phi;
                const int kg = k + ldom[2].first() - nghost_phi;

                // interior values
                bool isInterior = ((ig >= 0) && (jg >= 0) && (kg >= 0) && (ig < nr_m[0])
                                       && (jg < nr_m[1]) && (kg < nr_m[2]));
                scalar interior = -view_phiNm1(i, j, k) + a1 * view_phiN(i, j, k)
                                  + a2 * (view_phiN(i + 1, j, k) + view_phiN(i - 1, j, k))
                                  + a4 * (view_phiN(i, j + 1, k) + view_phiN(i, j - 1, k))
                                  + a6 * (view_phiN(i, j, k + 1) + view_phiN(i, j, k - 1))
                                  + a8 * (-view_rhoN(i, j, k) / epsilon0);

                view_phiNp1(i, j, k) = isInterior * interior + !isInterior * view_phiNp1(i, j, k);
            });

        for (size_t gd = 0; gd < Dim; ++gd) {
            Kokkos::parallel_for(
                "Vector potential update", ippl::getRangePolicy(view_aN, nghost_a),
                KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // global indices
                    const int ig = i + ldom[0].first() - nghost_a;
                    const int jg = j + ldom[1].first() - nghost_a;
                    const int kg = k + ldom[2].first() - nghost_a;

                    // interior values
                    bool isInterior = ((ig >= 0) && (jg >= 0) && (kg >= 0) && (ig < nr_m[0])
                                           && (jg < nr_m[1]) && (kg < nr_m[2]));
                    scalar interior = -view_aNm1(i, j, k)[gd] + a1 * view_aN(i, j, k)[gd]
                                      + a2 * (view_aN(i + 1, j, k)[gd] + view_aN(i - 1, j, k)[gd])
                                      + a4 * (view_aN(i, j + 1, k)[gd] + view_aN(i, j - 1, k)[gd])
                                      + a6 * (view_aN(i, j, k + 1)[gd] + view_aN(i, j, k - 1)[gd])
                                      + a8 * (-view_JN(i, j, k)[gd] * mu0);
                    if (!isInterior && gd == 0 && jg == 1) {
                        // std::printf("%f, %f, %f\n", view_aNp1(i, j, k)[0], view_aNp1(i, j, k)[1],
                        // view_aNp1(i, j, k)[2]); std::printf("%d, %d, %d\n", ig, jg, kg);
                    }
                    view_aNp1(i, j, k)[gd] =
                        isInterior * interior + !isInterior * view_aNp1(i, j, k)[gd];
                });
        }*/
        (*rhoN_mp) = 0.0;
        bunch.Q.scatter(*rhoN_mp, bunch.R);
        AN_m.getFieldBC().apply(AN_m);
        Kokkos::parallel_for(
            "4-Vector potential update", ippl::getRangePolicy(view_AN, nghost_A),
            KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                for (size_t gd = 0; gd < Vector4_t::dim; ++gd) {
                    // global indices
                    const int ig = i + ldom[0].first() - nghost_A;
                    const int jg = j + ldom[1].first() - nghost_A;
                    const int kg = k + ldom[2].first() - nghost_A;

                    // interior values
                    bool isInterior = ((ig >= 0) && (jg >= 0) && (kg >= 0) && (ig < nr_m[0])
                                           && (jg < nr_m[1]) && (kg < nr_m[2]));
                    scalar interior = -view_ANm1(i, j, k)[gd] + a1 * view_AN(i, j, k)[gd]
                                      + a2 * (view_AN(i + 1, j, k)[gd] + view_AN(i - 1, j, k)[gd])
                                      + a4 * (view_AN(i, j + 1, k)[gd] + view_AN(i, j - 1, k)[gd])
                                      + a6 * (view_AN(i, j, k + 1)[gd] + view_AN(i, j, k - 1)[gd])
                                      + a8 * (gd == 0 ? (-view_rhoN(i, j, k) / epsilon0) : (-view_JN(i, j, k)[gd - 1] * mu0));
                    view_ANp1(i, j, k)[gd] =
                        isInterior * interior + !isInterior * view_ANp1(i, j, k)[gd];
                }
            }
        );

        // interior points need to have been updated before TF/SF seed and ABCs
        Kokkos::fence();

        // add seed field via TF/SF boundaries
        if (seed && false) {
            iteration++;

            // the scattered field boundary is the 2nd point after the boundary
            // the total field boundary is the 3rd point after the boundary
            for (size_t gd = 0; gd < 1; ++gd) {
                Kokkos::parallel_for(
                    "Vector potential seeding", ippl::getRangePolicy(view_AN, nghost_A),
                    KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                        // global indices
                        const size_t ig = i + ldom[0].first() - nghost_A;
                        const size_t jg = j + ldom[1].first() - nghost_A;
                        const size_t kg = k + ldom[2].first() - nghost_A;

                        size_t distance = boundary_distance(ig, jg, kg, nr_m[0], nr_m[1], nr_m[2]);
                        ippl::Vector<scalar, Dim> as{a2, a4, a6};
                        ippl::Vector<axis_aligned_occlusion, Dim> sfocc = boundary_occlusion_of(0, ippl::Vector<size_t, 3>{ig, jg, kg}, ippl::Vector<size_t, 3>{(size_t)nr_m[0], (size_t)nr_m[1], (size_t)nr_m[2]});
                        ippl::Vector<axis_aligned_occlusion, Dim> tfocc = boundary_occlusion_of(1, ippl::Vector<size_t, 3>{ig, jg, kg}, ippl::Vector<size_t, 3>{(size_t)nr_m[0], (size_t)nr_m[1], (size_t)nr_m[2]});
                        // SF boundary in all 3 dimensions
                        scalar tfsfaccum = 0.0;
                        const scalar seed_function_eval = gaussian(iteration, i, j, k);
                        for(unsigned int d = 0;d < Dim;d++){
                            tfsfaccum +=  as[d] * seed_function_eval * scalar(sfocc[d] == axis_aligned_occlusion::AT_MIN);
                            tfsfaccum += -as[d] * seed_function_eval * scalar(sfocc[d] == axis_aligned_occlusion::AT_MAX);
                            tfsfaccum +=  as[d] * seed_function_eval * scalar(tfocc[d] == axis_aligned_occlusion::AT_MIN);
                            tfsfaccum += -as[d] * seed_function_eval * scalar(tfocc[d] == axis_aligned_occlusion::AT_MAX);
                        }

                        // update field (add seed)
                        view_ANp1(i, j, k)[gd] += tfsfaccum;
                    });
            }
            Kokkos::fence();
        }
            
        // apply 1st order Absorbing Boundary Conditions
        // for both scalar and vector potentials
        
        ippl::Vector<size_t, 3> extenz = ippl::Vector<size_t, 3>{(size_t)nr_m[0] + 2, (size_t)nr_m[1] + 2, (size_t)nr_m[2] + 2};
        
        Kokkos::parallel_for(
            "Vector potential ABCs", ippl::getRangePolicy(view_AN /*, nghost_a*/),
            KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                for (size_t gd = 0; gd < 3; ++gd) {
                    if(this->bconds[gd] != MUR_ABC_1ST)continue;
                    // global indices
                    const int ig = i + ldom[0].first();// - nghost_a;
                    const int jg = j + ldom[1].first();// - nghost_a;
                    const int kg = k + ldom[2].first();// - nghost_a;

                    assert((ig | jg | kg) >= 0); //Assert all positive

                    ippl::Vector<axis_aligned_occlusion, 3> bocc = boundary_occlusion_of(0, ippl::Vector<size_t, 3>{(size_t)ig, (size_t)jg, (size_t)kg}, extenz);
                    
                    if(+bocc[gd]){
                        int offs = bocc[gd] == AT_MIN ? 1 : -1;
                        if(gd == 0){
                            //view_ANp1(i, j, k) = view_AN(i, j, k) + (view_AN(i + offs, j, k) - view_AN(i, j, k)) * this->dt / hr_m[0];
                            view_ANp1(i, j, k) = abc_x[bocc[gd] >> 1](view_AN, view_ANm1, view_ANp1, ippl::Vector<size_t, 3>({i, j, k}));
                            //view_aNp1(i, j, k) = view_aN(i, j, k) + (view_aN(i + offs, j, k) - view_aN(i, j, k)) * this->dt / hr_m[0];
                            //view_phiNp1(i, j, k) = view_phiN(i, j, k) + (view_phiN(i + offs, j, k) - view_phiN(i, j, k)) * this->dt / hr_m[0];
                            //view_aNp1(i, j, k) = abc_x[bocc[gd] >> 1](view_aN, view_aNm1, view_aNp1, ippl::Vector<size_t, 3>({i, j, k}));
                        }
                        if(gd == 1){
                            //view_ANp1(i, j, k) = view_AN(i, j, k) + (view_AN(i, j + offs, k) - view_AN(i, j, k)) * this->dt / hr_m[1];
                            view_ANp1(i, j, k) = abc_y[bocc[gd] >> 1](view_AN, view_ANm1, view_ANp1, ippl::Vector<size_t, 3>({i, j, k}));
                            //view_aNp1(i, j, k) = view_aN(i, j, k) + (view_aN(i, j + offs, k) - view_aN(i, j, k)) * this->dt / hr_m[1];
                            //view_phiNp1(i, j, k) = view_phiN(i, j, k) + (view_phiN(i, j + offs, k) - view_phiN(i, j, k)) * this->dt / hr_m[1];
                            //view_aNp1(i, j, k) = abc_y[bocc[gd] >> 1](view_aN, view_aNm1, view_aNp1, ippl::Vector<size_t, 3>({i, j, k}));
                        }
                        if(gd == 2){
                            //view_ANp1(i, j, k) = view_AN(i, j, k) + (view_AN(i, j, k + offs) - view_AN(i, j, k)) * this->dt / hr_m[2];
                            view_ANp1(i, j, k) = abc_z[bocc[gd] >> 1](view_AN, view_ANm1, view_ANp1, ippl::Vector<size_t, 3>({i, j, k}));
                            //view_aNp1(i, j, k) = view_aN(i, j, k) + (view_aN(i, j, k + offs) - view_aN(i, j, k)) * this->dt / hr_m[2];
                            //view_phiNp1(i, j, k) = view_phiN(i, j, k) + (view_phiN(i, j, k + offs) - view_phiN(i, j, k)) * this->dt / hr_m[2];
                            //view_aNp1(i, j, k) = abc_z[bocc[gd] >> 1](view_aN, view_aNm1, view_aNp1, ippl::Vector<size_t, 3>({i, j, k}));
                        }
                    }
                }
            }
        );
        Kokkos::fence();
        field_evaluation();
        auto gammabeta_view = bunch.gamma_beta.getView();
        auto Qview = bunch.Q.getView();
        auto rview = bunch.R.getView();
        auto rnp1view = bunch.R_np1.getView();

        Kokkos::View<bool*> invalid("OOB Particcel", bunch.getLocalNum());
        size_t invalid_count = 0;
        
        Kokkos::parallel_reduce(Kokkos::RangePolicy<typename playout_type::RegionLayout_t::view_type::execution_space>(0, bunch.getLocalNum()), KOKKOS_LAMBDA(size_t i, size_t& ref){
            bool out_of_bounds = false;
            ippl::Vector<scalar, Dim> ppos = rview(i);
            for(size_t i = 0;i < Dim;i++){
                out_of_bounds |= (ppos[i] <= 0.0);
                out_of_bounds |= (ppos[i] >= 1.0); //Check against simulation domain
            }
            invalid(i) = out_of_bounds;
            ref += out_of_bounds;
        }, invalid_count);
        //bunch.destroy(invalid, invalid_count);
        bunch.E_gather.gather(*this->En_mp, bunch.R);
        bunch.B_gather.gather(*this->Bn_mp, bunch.R);
        
        auto E_gatherview = bunch.E_gather.getView();
        auto B_gatherview = bunch.B_gather.getView();
        constexpr scalar e_mass = 0.5110;
        scalar this_dt = this->dt;
        int ronk = ippl::Comm->rank();
        //LOG("Particcel count: " << bunch.getLocalNum());
        Kokkos::parallel_for(Kokkos::RangePolicy<typename playout_type::RegionLayout_t::view_type::execution_space>(0, bunch.getLocalNum()), KOKKOS_LAMBDA(const size_t i){
            using Kokkos::sqrt;
            scalar charge = -Qview(i);
            //using ::sqrt;
            const ippl::Vector<scalar, 3> pgammabeta = gammabeta_view(i);
            //LOG(gammabeta_view(i));
            //LOG(B_gatherview(i));
            const ippl::Vector<scalar, 3> t1 = pgammabeta - charge * this_dt * E_gatherview(i) / (2.0 * e_mass); 
            const scalar alpha = -charge * this_dt / (scalar(2) * e_mass * sqrt(1 + dot_prod(t1, t1)));
            ippl::Vector<scalar, 3> crossprod = alpha * ippl::cross(t1, B_gatherview(i));
            const ippl::Vector<scalar, 3> t2 = t1 + alpha * ippl::cross(t1, B_gatherview(i));
            const ippl::Vector<scalar, 3> t3 = t1 + ippl::cross(t2, 2.0 * alpha * (B_gatherview(i) / (1.0 + alpha * alpha * dot_prod(B_gatherview(i), B_gatherview(i)))));
            const ippl::Vector<scalar, 3> ngammabeta = t3 - charge * this_dt * E_gatherview(i) / (2.0 * e_mass);
            if(ronk == 0){
                ippl::Vector<scalar, 3> adder = this_dt * ngammabeta / (sqrt(1.0 + dot_prod(ngammabeta, ngammabeta)));
                //LOG(adder);
            }
            rnp1view(i) = rview(i) + this_dt * ngammabeta / (sqrt(1.0 + dot_prod(ngammabeta, ngammabeta)));
            gammabeta_view(i) = ngammabeta;
        });
        this->JN_mp->operator=(0.0); //reset J to zero for next time step before scatter again
        

                                                                    //What is the scaling factor here?
        bunch.Q.scatter(*this->JN_mp, bunch.R, bunch.R_np1, scalar(1.0) / (dt * hr_m[0] * hr_m[1]));
        
        //bunch.layout_m->update();
        Kokkos::deep_copy(bunch.R.getView(), bunch.R_np1.getView());
        bunch_type bunch_buffer(pl);
        pl.update(bunch);
        //Copy potential at N-1 and N+1 to phiNm1 and phiN for next time step
        Kokkos::deep_copy(ANm1_m.getView(), AN_m.getView());
        Kokkos::deep_copy(AN_m.getView(),   ANp1_m.getView());
        //Kokkos::deep_copy(aNm1_m.getView(), aN_m.getView());
        //Kokkos::deep_copy(aN_m.getView(), aNp1_m.getView());
        //Kokkos::deep_copy(phiNm1_m.getView(), phiN_m.getView());
        //Kokkos::deep_copy(phiN_m.getView(), phiNp1_m.getView()); 
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    Tfields FDTDSolver<Tfields, Dim, M, C>::field_evaluation() {
        // magnetic field is the curl of the vector potential
        // we take the average of the potential at N and N+1
        auto Aview = AN_m.getView();
        auto AM1view = ANm1_m.getView();
        auto AP1view = ANp1_m.getView();
        //auto aview = aN_m.getView();
        auto eview = En_mp->getView();
        auto bview = Bn_mp->getView();
        const auto idt = 1.0 / dt;
        //(*En_mp) = -(aNp1_m - aN_m) / dt - grad(phiN_m);
        //(*Bn_mp) = (curl(aN_m) + curl(aNp1_m)) * 0.5;
        auto gc_expression = gradcurl(AN_m);
        auto gcp1_expression = gradcurl(ANp1_m);
        //auto ebview = EBn_m.getView();
        Kokkos::parallel_for(ippl::getRangePolicy(eview, 1), KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k){
            ippl::Vector<scalar, Dim * 2> gc_eval = gc_expression(i,j,k);
            ippl::Vector<scalar, Dim * 2> gcp1_eval = gcp1_expression(i,j,k);
            eview(i, j, k) = -gc_eval.template head<3>();
            bview(i, j, k) =  (gc_eval.template tail<3>() + gcp1_eval.template tail<3>()) * 0.5;
            //LOG("megnet" << bview(i,j,k));
            eview(i, j, k) -= (AP1view(i,j,k).template tail<3>() - Aview(i,j,k).template tail<3>()) * idt;
        });

        if(radiation){
            (*radiation) = ippl::cross(*En_mp, *Bn_mp);
        }

        //typename FDTDSolver<Tfields, Dim, M, C>::Field_t energy_density = phiN_m.deepCopy();
        //auto edview = energy_density.getView();
        
        scalar eb_energy = 0.0;
        scalar radiation_on_boundary = 0.0;
        ippl::Vector<size_t, 3> extenz = ippl::Vector<size_t, 3>{(size_t)nr_m[0] + 2, (size_t)nr_m[1] + 2, (size_t)nr_m[2] + 2};
        if(radiation)
            Kokkos::parallel_reduce(ippl::getRangePolicy(eview, 1), KOKKOS_LAMBDA(int i, int j, int k, scalar& ref){
                //edview(i,j,k) = aview(i,j,k)[2];
                //std::cout << "E: " << squaredNorm(eview(i,j,k)) << " | B: " << squaredNorm(bview(i,j,k)) << "\n";
                ippl::Vector<axis_aligned_occlusion, 3> bocc = boundary_occlusion_of(2, ippl::Vector<size_t, 3>{(size_t)i, (size_t)j, (size_t)k}, extenz);
                ippl::Vector<double, 3> outward_normal = 0.0;
                for(int dd = 0;dd < 3;dd++){
                    if(bocc[dd] == AT_MAX){
                        outward_normal[dd] = 1.0;
                    }
                    else if(bocc[dd] == AT_MIN){
                        outward_normal[dd] = -1.0;
                    }
                }
                ref += dot_prod(radiation->getView()(i,j,k), outward_normal);
            }, radiation_on_boundary);
        Kokkos::fence();
        radiation_on_boundary *= hr_m[0] * hr_m[0];
        this->absorbed__energy += radiation_on_boundary;
        Kokkos::parallel_reduce(ippl::getRangePolicy(eview, 1), KOKKOS_LAMBDA(int i, int j, int k, scalar& ref){
            //edview(i,j,k) = aview(i,j,k)[2];
            //std::cout << "E: " << squaredNorm(eview(i,j,k)) << " | B: " << squaredNorm(bview(i,j,k)) << "\n";
            ref += squaredNorm(eview(i, j, k));
            ref += squaredNorm(bview(i, j, k));
        }, eb_energy);
        eb_energy *= 0.5;
        Kokkos::fence();
        //std::cout << "E contr: " << EE * hr_m[0] * hr_m[1] * hr_m[2] << "B contr: " << BE * hr_m[0] * hr_m[1] * hr_m[2] << "\n"; 
        //energy_density = (ippl::dot(*En_mp, *En_mp) + ippl::dot(*Bn_mp, *Bn_mp)) * scalar(0.5);
        this->total_energy = eb_energy * hr_m[0] * hr_m[1] * hr_m[2] + bunch_energy(this->bunch);
        return eb_energy * hr_m[0] * hr_m[1] * hr_m[2];
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    KOKKOS_FUNCTION Tfields FDTDSolver<Tfields, Dim, M, C>::gaussian(size_t it, size_t i, size_t j,
                                                    size_t k) const noexcept {
        // return 1.0;
        using Kokkos::exp;
        //using ::exp;


        
        const scalar y = 1.0 - (j - 2) * hr_m[1];  // From the max boundary; fix this sometime
        const scalar t = it * dt;
        scalar plane_wave_excitation_x = (y - t < 0) ? -(y - t) : 0;
        (void)i;
        (void)j;
        (void)k;
        return scalar(100) * exp(-sq((scalar(2.0) - plane_wave_excitation_x)) * scalar(2));

        // scalar arg = Kokkos::pow((1.0 - it * dt) / 0.25, 2);
        // return 100 * Kokkos::exp(-arg);
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    void FDTDSolver<Tfields, Dim, M, C>::initialize() {

        constexpr scalar c        = 1.0;  // 299792458.0;
        constexpr scalar mu0      = 1.0;  // 1.25663706212e-6;
        constexpr scalar epsilon0 = 1.0 / (c * c * mu0);
        constexpr scalar electron_charge = -0.30282212088;
        constexpr scalar electron_mass = 0.5110;
        bunch.create(particle_count / ippl::Comm->size());
        auto Rview = bunch.R.getView();
        auto gbview = bunch.gamma_beta.getView();
        
        
        bunch.Q = electron_charge * 1.0;
        bunch.mass = electron_mass;
        //bunch.R = ippl::Vector<scalar, 3>(0.4);
        //bunch.gamma_beta = ippl::Vector<scalar, 3>{0.0, 1e1, 0.0};
        bunch.R_np1 = ippl::Vector<scalar, 3>(0.3);
        // get layout and mesh
        layout_mp = &(this->rhoN_mp->getLayout());
        mesh_mp   = &(this->rhoN_mp->get_mesh());

        // get mesh spacing, domain, and mesh size
        hr_m     = mesh_mp->getMeshSpacing();
        domain_m = layout_mp->getDomain();
        for (unsigned int i = 0; i < Dim; ++i)
            nr_m[i] = domain_m[i].length();
        // initialize fields
        
        //phiNm1_m.initialize(*mesh_mp, *layout_mp);
        //phiN_m.initialize(*mesh_mp, *layout_mp);
        //phiNp1_m.initialize(*mesh_mp, *layout_mp);

        //aNm1_m.initialize(*mesh_mp, *layout_mp);
        //aN_m.initialize(*mesh_mp, *layout_mp);
        //aNp1_m.initialize(*mesh_mp, *layout_mp);

        ANm1_m.initialize(*mesh_mp, *layout_mp);
        AN_m.initialize(*mesh_mp, *layout_mp);
        ANp1_m.initialize(*mesh_mp, *layout_mp);
        //EBn_m.initialize(*mesh_mp, *layout_mp);

        //phiNm1_m = 0.0;
        //phiN_m   = 0.0;
        //phiNp1_m = 0.0;

        //aNm1_m = 0.0;
        //aN_m   = 0.0;
        //aNp1_m = 0.0;

        ANm1_m = 0.0;
        AN_m   = 0.0;
        ANp1_m = 0.0;
        absorbed__energy = 0;
    };
    template<typename Tfields, unsigned Dim, class M, class C>
    template<typename callable>
    void FDTDSolver<Tfields, Dim, M, C>::apply_to_fields(callable c){
        c(AN_m);
        c(ANp1_m);
        c(ANm1_m);
    }
    template<typename Tfields, unsigned Dim, class M, class C>
    template<typename callable>
    void FDTDSolver<Tfields, Dim, M, C>::fill_initialcondition(callable c){
        //auto view_a      = aN_m  .getView();
        //auto view_an1    = aNm1_m.getView();
        auto view_A      = AN_m  .getView();
        auto view_An1    = ANm1_m.getView();
        //auto view_phi    = phiN_m.getView();
        //auto view_phin1  = phiNm1_m.getView();
        auto ldom = layout_mp->getLocalNDIndex();
        //std::cout << "Rank " << ippl::Comm->rank() << " has y offset " << ldom[1].first() << "\n";
        int nghost = AN_m.getNghost();
        Kokkos::parallel_for(
            "Assign sinusoidal source at center", ippl::getRangePolicy(view_A, 0), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k){
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                scalar x = (scalar(ig) + 0.5) * hr_m[0];// + origin[0];
                scalar y = (scalar(jg) + 0.5) * hr_m[1];// + origin[1];
                scalar z = (scalar(kg) + 0.5) * hr_m[2];// + origin[2];
                //view_a  (i, j, k) = c(x, y, z);
                //view_an1(i, j, k) = c(x, y, z);
                //view_phi(i,j,k)   = 0.0;
                //view_phin1(i,j,k) = 0.0;
                //view_a  (i, j, k) = c(x, y, z);
                //view_an1(i, j, k) = c(x, y, z);
                for(int l = 0;l < 4;l++){
                    view_A  (i, j, k)[l] = c(i, j, k, x, y, z)[l];
                    view_An1(i, j, k)[l] = c(i, j, k, x, y, z)[l];
                    
                }
            }
        );
    }
    template<typename Tfields, unsigned Dim, class M, class C>
    template<typename callable>
    Tfields FDTDSolver<Tfields, Dim, M, C>::volumetric_integral(callable c){
        auto view_A      = AN_m.getView();
        //auto view_an1    = aNm1_m.getView();
        auto ldom = layout_mp->getLocalNDIndex();
        int nghost = AN_m.getNghost();
        Tfields ret = 0.0;
        Tfields volume = hr_m[0] * hr_m[1] * hr_m[2];
        Kokkos::parallel_reduce(
            "Assign sinusoidal source at center", ippl::getRangePolicy(view_A), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, scalar& ref){
                
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                scalar x = (ig + 0.5) * hr_m[0];// + origin[0];
                scalar y = (jg + 0.5) * hr_m[1];// + origin[1];
                scalar z = (kg + 0.5) * hr_m[2];// + origin[2];

                ref += c(i, j, k, x, y, z) * volume;
            }, ret
        );
        return ret;
    }
}  // namespace ippl
