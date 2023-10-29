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
template <typename View, typename Coords, unsigned int axis, size_t... Idx>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) apply_impl_with_offset(const View& view,
                                                           const Coords& coords,
                                                           int offset,
                                                           const std::index_sequence<Idx...>&) {
    return view((coords[Idx] + offset * !!(Idx == axis))...);
}
template <typename View, typename Coords, unsigned axis>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) apply_with_offset(const View& view, const Coords& coords, int offset) {
    using Indices = std::make_index_sequence<ippl::ExtractExpressionRank::getRank<Coords>()>;
    return apply_impl_with_offset<View, Coords, axis>(view, coords, offset, Indices{});
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
        
        value_t ret = beta0 * (apply_with_offset<view_type, Coords, ~0u>(a_nm1, c, sign) + apply_with_offset<view_type, Coords, main_axis>(a_np1, c, sign))
                    + beta1 * (apply_with_offset<view_type, Coords, ~0u>(a_n, c, sign) + apply_with_offset<view_type, Coords, main_axis>(a_n, c, sign))
                    + beta2 * (apply_with_offset<view_type, Coords, main_axis>(a_nm1, c, sign));
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
template <typename... index_and_extent_types>
KOKKOS_INLINE_FUNCTION size_t boundary_distance(index_and_extent_types&&... x){
    std::tuple<index_and_extent_types...> tp{x...};
    constexpr size_t hsize = sizeof...(index_and_extent_types) / 2;
    auto difference_minimizer = KOKKOS_LAMBDA<size_t... Index>(const std::index_sequence<Index...>& seq){
        size_t min_to_0 = std::min({std::get<Index>(tp)...});
        size_t min_to_extent = std::min({(std::get<Index + hsize>(tp) - std::get<Index>(tp) - 1)...});
        (void)seq;
        return std::min(min_to_0, min_to_extent);
    };
    return difference_minimizer(std::make_index_sequence<sizeof...(index_and_extent_types) / 2>{});
}
template <typename... extent_types>
KOKKOS_INLINE_FUNCTION std::array<axis_aligned_occlusion, sizeof...(extent_types)> boundary_occlusion_of(
    size_t boundary_distance, const std::tuple<extent_types...>& _index,
    const std::tuple<extent_types...>& _extents) {
    constexpr size_t Dim = std::tuple_size_v<std::tuple<extent_types...>>;

    constexpr auto get_array = []<typename... Ts>(Ts&&... x) {
        return std::array<size_t, sizeof...(x)>{static_cast<size_t>(x)...};
    };

    std::array<size_t, Dim> index   = std::apply(get_array, _index);
    std::array<size_t, Dim> extents = std::apply(get_array, _extents);
    std::array<axis_aligned_occlusion, Dim> ret_array;

    size_t minimal_distance_to_zero             = index[0];
    size_t minimal_distance_to_extent_minus_one = extents[0] - index[0] - 1;
    ret_array[0] = (axis_aligned_occlusion)(index[0] == boundary_distance);
    ret_array[0] |= (axis_aligned_occlusion)(index[0] == (extents[0] - 1 - boundary_distance)) << 1;
    for (size_t i = 1; i < Dim; i++) {
        minimal_distance_to_zero = std::min(minimal_distance_to_zero, index[i]);
        minimal_distance_to_extent_minus_one =
            std::min(minimal_distance_to_extent_minus_one, extents[i] - index[i] - 1);
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
// template<typename T>
// ippl::Vector<T, 3> cross(const ippl::Vector<T, 3>& a, const ippl::Vector<T, 3>& b){
//     ippl::Vector<T, 3> ret;
//     ret[0] = a[1]* b[2] - a[2] * b[1];
//     ret[2] = a[0] * b[1] - a[1] * b[0];
//     ret[1] = a[2] * b[0] - a[0] * b[2];
//     return ret;
// }
namespace ippl {

    template <typename Tfields, unsigned Dim, class M, class C>
    FDTDSolver<Tfields, Dim, M, C>::FDTDSolver(Field_t* charge, VField_t* current, VField_t* E,
                                               VField_t* B, scalar timestep, bool seed_) : pl(charge->getLayout(), charge->get_mesh()), bunch(pl), particle_count(1) {
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

    template <typename Tfields, unsigned Dim, class M, class C>
    FDTDSolver<Tfields, Dim, M, C>::~FDTDSolver(){};

    template <typename Tfields, unsigned Dim, class M, class C>
    void FDTDSolver<Tfields, Dim, M, C>::solve() {
        // physical constant
        // using scalar;// = typename Tfields::BareField_t::value_type::
        constexpr scalar c        = 1.0;  // 299792458.0; 
        constexpr scalar mu0      = 1.0;  // 1.25663706212e-6;
        constexpr scalar epsilon0 = 1.0 / (c * c * mu0);

        
        
        auto sqr = KOKKOS_LAMBDA(scalar x){return x * x;};
        // finite differences constants
        scalar a1 = 2.0
           * (1.0 - sqr(c * dt / hr_m[0]) - sqr(c * dt / hr_m[1])
                  - sqr(c * dt / hr_m[2]));
        scalar a2 = sqr(c * dt / hr_m[0]);  // a3 = a2
        scalar a4 = sqr(c * dt / hr_m[1]);  // a5 = a4
        scalar a6 = sqr(c * dt / hr_m[2]);  // a7 = a6
        scalar a8 = sqr(c * dt);

        // 1st order absorbing boundary conditions constants
        scalar beta0[3] = {(c * dt - hr_m[0]) / (c * dt + hr_m[0]),
                           (c * dt - hr_m[1]) / (c * dt + hr_m[1]),
                           (c * dt - hr_m[2]) / (c * dt + hr_m[2])};
        scalar beta1[3] = {scalar(2) * hr_m[0] / (c * dt + hr_m[0]), scalar(2) * hr_m[1] / (c * dt + hr_m[1]),
                           scalar(2) * hr_m[2] / (c * dt + hr_m[2])};
        scalar beta2[3] = {-scalar(1), -scalar(1), -scalar(1)};
        
        

        //static_assert(std::is_invocable_v<ippl::Vector<scalar, 3>, int>);
        // preliminaries for Kokkos loops (ghost cells and views)
        phiNp1_m         = scalar(0);
        aNp1_m           = scalar(0);
        aN_m.fillHalo();
        aNm1_m.fillHalo();
        phiN_m.fillHalo();
        phiNm1_m.fillHalo();

        auto view_phiN   = phiN_m.getView();
        auto view_phiNm1 = phiNm1_m.getView();
        auto view_phiNp1 = phiNp1_m.getView();

        
        auto view_aN   = aN_m.getView();
        auto view_aNm1 = aNm1_m.getView();
        auto view_aNp1 = aNp1_m.getView();
        #define _bc first_order_abc<scalar,
        _bc 0, 1, 2> abc_x[] = {_bc 0, 1, 2> (hr_m, c, dt,  1), _bc 0, 1, 2>(hr_m, c, dt, -1)};
        _bc 1, 0, 2> abc_y[] = {_bc 1, 0, 2> (hr_m, c, dt,  1), _bc 1, 0, 2>(hr_m, c, dt, -1)};
        _bc 2, 0, 1> abc_z[] = {_bc 2, 0, 1> (hr_m, c, dt,  1), _bc 2, 0, 1>(hr_m, c, dt, -1)};


        auto view_rhoN = this->rhoN_mp->getView();
        auto view_JN   = this->JN_mp->getView();

        const int nghost_phi = phiN_m.getNghost();
        const int nghost_a   = aN_m.getNghost();
        const auto& ldom     = layout_mp->getLocalNDIndex();

        //aNm1_m.fillHalo();
        

        //JN_mp->fillHalo();
        // compute scalar potential and vector potential at next time-step
        // first, only the interior points are updated
        // then, if the user has set a seed, the seed is added via TF/SF boundaries
        // (TF/SF = total-field/scattered-field technique)
        // finally, absorbing boundary conditions are imposed
        Kokkos::parallel_for(
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
                    bool isInterior = true;//((ig >= 0) && (jg >= 0) && (kg >= 0) && (ig < nr_m[0])
                                           //&& (jg < nr_m[1]) && (kg < nr_m[2]));
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
        }

        // interior points need to have been updated before TF/SF seed and ABCs
        Kokkos::fence();

        // add seed field via TF/SF boundaries
        if (seed && false) {
            iteration++;

            // the scattered field boundary is the 2nd point after the boundary
            // the total field boundary is the 3rd point after the boundary
            for (size_t gd = 0; gd < 1; ++gd) {
                Kokkos::parallel_for(
                    "Vector potential seeding", ippl::getRangePolicy(view_aN, nghost_a),
                    KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                        // global indices
                        const int ig = i + ldom[0].first() - nghost_a;
                        const int jg = j + ldom[1].first() - nghost_a;
                        const int kg = k + ldom[2].first() - nghost_a;

                        // SF boundary in all 3 dimensions
                        bool isXmin_SF = ((ig == 1) && (jg > 1) && (kg > 1) && (jg < nr_m[1] - 2)
                                          && (kg < nr_m[2] - 2));
                        scalar xmin_SF = a2 * gaussian(iteration, i, j, k);
                        bool isYmin_SF = ((ig > 1) && (jg == 1) && (kg > 1) && (ig < nr_m[0] - 2)
                                          && (kg < nr_m[2] - 2));
                        scalar ymin_SF = a4 * gaussian(iteration, i, j, k);
                        bool isZmin_SF = ((ig > 1) && (jg > 1) && (kg == 1) && (ig < nr_m[0] - 2)
                                          && (jg < nr_m[1] - 2));
                        scalar zmin_SF = a6 * gaussian(iteration, i, j, k);
                        bool isXmax_SF = ((ig == nr_m[0] - 2) && (jg > 1) && (kg > 1)
                                          && (jg < nr_m[1] - 2) && (kg < nr_m[2] - 2));
                        scalar xmax_SF = -a2 * gaussian(iteration, i, j, k);
                        bool isYmax_SF = ((ig > 1) && (jg == nr_m[1] - 2) && (kg > 1)
                                          && (ig < nr_m[0] - 2) && (kg < nr_m[2] - 2));
                        scalar ymax_SF = -a4 * gaussian(iteration, i, j, k);
                        bool isZmax_SF = ((ig > 1) && (jg > 1) && (kg == nr_m[2] - 2)
                                          && (ig < nr_m[0] - 2) && (jg < nr_m[1] - 2));
                        scalar zmax_SF = -a6 * gaussian(iteration, i, j, k);

                        // TF boundary
                        bool isXmin_TF = ((ig == 2) && (jg > 2) && (kg > 2) && (jg < nr_m[1] - 3)
                                          && (kg < nr_m[2] - 3));
                        scalar xmin_TF = -a2 * gaussian(iteration, i, j, k);
                        bool isYmin_TF = ((ig > 2) && (jg == 2) && (kg > 2) && (ig < nr_m[0] - 3)
                                          && (kg < nr_m[2] - 3));
                        scalar ymin_TF = -a4 * gaussian(iteration, i, j, k);
                        bool isZmin_TF = ((ig > 2) && (jg > 2) && (kg == 2) && (ig < nr_m[0] - 3)
                                          && (jg < nr_m[1] - 3));
                        scalar zmin_TF = -a6 * gaussian(iteration, i, j, k);
                        bool isXmax_TF = ((ig == nr_m[0] - 3) && (jg > 2) && (kg > 2)
                                          && (jg < nr_m[1] - 3) && (kg < nr_m[2] - 3));
                        scalar xmax_TF = a2 * gaussian(iteration, i, j, k);
                        bool isYmax_TF = ((ig > 2) && (jg == nr_m[1] - 3) && (kg > 2)
                                          && (ig < nr_m[0] - 3) && (kg < nr_m[2] - 3));
                        scalar ymax_TF = a4 * gaussian(iteration, i, j, k);
                        bool isZmax_TF = ((ig > 2) && (jg > 2) && (kg == nr_m[2] - 3)
                                          && (ig < nr_m[0] - 3) && (jg < nr_m[1] - 3));
                        scalar zmax_TF = a6 * gaussian(iteration, i, j, k);
                        isXmax_SF      = false;
                        isXmax_TF      = false;
                        isXmin_SF      = false;
                        isYmin_TF      = false;
                        // isYmax_SF = false;
                        // isYmax_TF = false;
                        isYmin_SF = false;
                        isXmin_TF = false;
                        isZmax_SF = false;
                        isZmax_TF = false;
                        isZmin_SF = false;
                        isZmin_TF = false;
                        // update field (add seed)
                        view_aNp1(i, j, k)[gd] +=
                            isXmin_SF * xmin_SF + isYmin_SF * ymin_SF + isZmin_SF * zmin_SF
                            + isXmax_SF * xmax_SF + isYmax_SF * ymax_SF + isZmax_SF * zmax_SF
                            + isXmin_TF * xmin_TF + isYmin_TF * ymin_TF + isZmin_TF * zmin_TF
                            + isXmax_TF * xmax_TF + isYmax_TF * ymax_TF + isZmax_TF * zmax_TF;
                    });
            }
        }
        Kokkos::fence();
            
        // apply 1st order Absorbing Boundary Conditions
        // for both scalar and vector potentials
            Kokkos::parallel_for(
                "Scalar potential ABCs", ippl::getRangePolicy(view_phiN, nghost_phi),
                KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // global indices
                    const int ig = i + ldom[0].first() - nghost_phi;
                    const int jg = j + ldom[1].first() - nghost_phi;
                    const int kg = k + ldom[2].first() - nghost_phi;

                    // boundary values: 1st order Absorbing Boundary Conditions
                    bool isXmin = ((ig == 0) && (jg > 0) && (kg > 0) && (jg < nr_m[1] - 1)
                                   && (kg < nr_m[2] - 1));
                    scalar xmin = beta0[0] * (view_phiNm1(i, j, k) + view_phiNp1(i + 1, j, k))
                                  + beta1[0] * (view_phiN(i, j, k) + view_phiN(i + 1, j, k))
                                  + beta2[0] * (view_phiNm1(i + 1, j, k));
                    bool isYmin = ((ig > 0) && (jg == 0) && (kg > 0) && (ig < nr_m[0] - 1)
                                   && (kg < nr_m[2] - 1));
                    scalar ymin = beta0[1] * (view_phiNm1(i, j, k) + view_phiNp1(i, j + 1, k))
                                  + beta1[1] * (view_phiN(i, j, k) + view_phiN(i, j + 1, k))
                                  + beta2[1] * (view_phiNm1(i, j + 1, k));
                    bool isZmin = ((ig > 0) && (jg > 0) && (kg == 0) && (ig < nr_m[0] - 1)
                                   && (jg < nr_m[1] - 1));
                    scalar zmin = beta0[2] * (view_phiNm1(i, j, k) + view_phiNp1(i, j, k + 1))
                                  + beta1[2] * (view_phiN(i, j, k) + view_phiN(i, j, k + 1))
                                  + beta2[2] * (view_phiNm1(i, j, k + 1));
                    bool isXmax = ((ig == nr_m[0] - 1) && (jg > 0) && (kg > 0) && (jg < nr_m[1] - 1)
                                   && (kg < nr_m[2] - 1));
                    scalar xmax = beta0[0] * (view_phiNm1(i, j, k) + view_phiNp1(i - 1, j, k))
                                  + beta1[0] * (view_phiN(i, j, k) + view_phiN(i - 1, j, k))
                                  + beta2[0] * (view_phiNm1(i - 1, j, k));
                    bool isYmax = ((ig > 0) && (jg == nr_m[1] - 1) && (kg > 0) && (ig < nr_m[0] - 1)
                                   && (kg < nr_m[2] - 1));
                    scalar ymax = beta0[1] * (view_phiNm1(i, j, k) + view_phiNp1(i, j - 1, k))
                                  + beta1[1] * (view_phiN(i, j, k) + view_phiN(i, j - 1, k))
                                  + beta2[1] * (view_phiNm1(i, j - 1, k));
                    bool isZmax = ((ig > 0) && (jg > 0) && (kg == nr_m[2] - 1) && (ig < nr_m[0] - 1)
                                   && (jg < nr_m[1] - 1));
                    scalar zmax = beta0[2] * (view_phiNm1(i, j, k) + view_phiNp1(i, j, k - 1))
                                  + beta1[2] * (view_phiN(i, j, k) + view_phiN(i, j, k - 1))
                                  + beta2[2] * (view_phiNm1(i, j, k - 1));
                    bool isInterior      = !(isXmin | isXmax | isYmin | isYmax | isZmin | isZmax);
                    view_phiNp1(i, j, k) = isXmin * xmin + isYmin * ymin + isZmin * zmin
                                           + isXmax * xmax + isYmax * ymax + isZmax * zmax
                                           + isInterior * view_phiNp1(i, j, k);
                });
            //bool isBoundary = (lDomains[myRank][d].max() == domain[d].max())
            //                  || (lDomains[myRank][d].min() == domain[d].min());
            for (size_t gd = 0; gd < 3; ++gd) {
                if(this->bconds[gd] != MUR_ABC_1ST)continue;
                std::tuple<size_t, size_t, size_t> extenz = std::make_tuple(nr_m[0] + 2, nr_m[1] + 2, nr_m[2] + 2);
                Kokkos::parallel_for(
                    "Vector potential ABCs", ippl::getRangePolicy(view_aN /*, nghost_a*/),
                    KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                        // global indices
                        const int ig = i + ldom[0].first();// - nghost_a;
                        const int jg = j + ldom[1].first();// - nghost_a;
                        const int kg = k + ldom[2].first();// - nghost_a;

                        assert((ig | jg | kg) >= 0); //Assert all positive

                        std::array<axis_aligned_occlusion, 3> bocc = boundary_occlusion_of(0, std::make_tuple((size_t)ig, (size_t)jg, (size_t)kg), extenz);
                        
                        //for(unsigned ax = 0;ax < Dim;ax++){
                            if(+bocc[gd] && (!(+bocc[gd] & (+bocc[gd] - 1)))){
                                if(gd == 0)
                                    view_aNp1(i, j, k) = abc_x[bocc[gd] >> 1](view_aN, view_aNm1, view_aNp1, ippl::Vector<size_t, 3>({i, j, k}));
                                if(gd == 1)
                                    view_aNp1(i, j, k) = abc_y[bocc[gd] >> 1](view_aN, view_aNm1, view_aNp1, ippl::Vector<size_t, 3>({i, j, k}));
                                if(gd == 2)
                                    view_aNp1(i, j, k) = abc_z[bocc[gd] >> 1](view_aN, view_aNm1, view_aNp1, ippl::Vector<size_t, 3>({i, j, k}));
                            }
                        return;
                        //}
                        // boundary values: 1st order Absorbing Boundary Conditions
                        bool isXmin = ((ig == 0) && (jg > 0) && (kg > 0) && (jg < nr_m[1] - 1)
                                       && (kg < nr_m[2] - 1));
                        scalar xmin =
                            beta0[0] * (view_aNm1(i, j, k)[gd] + view_aNp1(i + 1, j, k)[gd])
                            + beta1[0] * (view_aN(i, j, k)[gd] + view_aN(i + 1, j, k)[gd])
                            + beta2[0] * (view_aNm1(i + 1, j, k)[gd]);
                        bool isYmin = ((ig > 0) && (jg == 0) && (kg > 0) && (ig < nr_m[0] - 1)
                                       && (kg < nr_m[2] - 1));
                        scalar ymin =
                            beta0[1] * (view_aNm1(i, j, k)[gd] + view_aNp1(i, j + 1, k)[gd])
                            + beta1[1] * (view_aN(i, j, k)[gd] + view_aN(i, j + 1, k)[gd])
                            + beta2[1] * (view_aNm1(i, j + 1, k)[gd]);
                        bool isZmin = ((ig > 0) && (jg > 0) && (kg == 0) && (ig < nr_m[0] - 1)
                                       && (jg < nr_m[1] - 1));
                        scalar zmin =
                            beta0[2] * (view_aNm1(i, j, k)[gd] + view_aNp1(i, j, k + 1)[gd])
                            + beta1[2] * (view_aN(i, j, k)[gd] + view_aN(i, j, k + 1)[gd])
                            + beta2[2] * (view_aNm1(i, j, k + 1)[gd]);
                        bool isXmax = ((ig == nr_m[0] - 1) && (jg > 0) && (kg > 0)
                                       && (jg < nr_m[1] - 1) && (kg < nr_m[2] - 1));
                        scalar xmax =
                            beta0[0] * (view_aNm1(i, j, k)[gd] + view_aNp1(i - 1, j, k)[gd])
                            + beta1[0] * (view_aN(i, j, k)[gd] + view_aN(i - 1, j, k)[gd])
                            + beta2[0] * (view_aNm1(i - 1, j, k)[gd]);
                        bool isYmax = ((ig > 0) && (jg == nr_m[1] - 1) && (kg > 0)
                                       && (ig < nr_m[0] - 1) && (kg < nr_m[2] - 1));
                        scalar ymax =
                            beta0[1] * (view_aNm1(i, j, k)[gd] + view_aNp1(i, j - 1, k)[gd])
                            + beta1[1] * (view_aN(i, j, k)[gd] + view_aN(i, j - 1, k)[gd])
                            + beta2[1] * (view_aNm1(i, j - 1, k)[gd]);
                        bool isZmax = ((ig > 0) && (jg > 0) && (kg == nr_m[2] - 1)
                                       && (ig < nr_m[0] - 1) && (jg < nr_m[1] - 1));
                        scalar zmax =
                            beta0[2] * (view_aNm1(i, j, k)[gd] + view_aNp1(i, j, k - 1)[gd])
                            + beta1[2] * (view_aN(i, j, k)[gd] + view_aN(i, j, k - 1)[gd])
                            + beta2[2] * (view_aNm1(i, j, k - 1)[gd]);

                        bool isInterior = !(isXmin | isXmax | isYmin | isYmax | isZmin | isZmax);
                        view_aNp1(i, j, k)[gd] = isXmin * xmin + isYmin * ymin + isZmin * zmin
                                                 + isXmax * xmax + isYmax * ymax + isZmax * zmax
                                                 + isInterior * view_aNp1(i, j, k)[gd];
                    });
            }
        Kokkos::fence();
        //for (size_t gd = 0; gd < Dim; ++gd) {
            if(false)
            Kokkos::parallel_for(
                "Periodic boundaryie", ippl::getRangePolicy(view_aNp1, 0),
                KOKKOS_CLASS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    size_t wraparound_i = i, wraparound_j = j, wraparound_k = k;
                    if(bconds[0] == PERIODIC_FACE){
                        if ((int)i < nghost_a) {
                            wraparound_i += view_aN.extent(0) - 2 * nghost_a;
                        } else if (i > view_aN.extent(0) - nghost_a - 1) {
                            wraparound_i -= view_aN.extent(0) - 2 * nghost_a;
                            
                        }
                    }
                    if(bconds[1] == PERIODIC_FACE){
                        if ((int)j < nghost_a) {
                            wraparound_j += view_aN.extent(1) - 2 * nghost_a;
                        } else if (j > view_aN.extent(1) - nghost_a - 1) {
                            wraparound_j -= view_aN.extent(1) - 2 * nghost_a;
                        }
                    }
                    if(bconds[2] == PERIODIC_FACE){
                        if ((int)k < nghost_a) {
                            wraparound_k += view_aN.extent(2) - 2 * nghost_a;
                        } else if (k > view_aN.extent(2) - nghost_a - 1) {
                            wraparound_k -= view_aN.extent(2) - 2 * nghost_a;
                        }
                    }

                    view_aNp1(i, j, k)[0] = view_aNp1(wraparound_i, wraparound_j, wraparound_k)[0];
                    view_aNp1(i, j, k)[1] = view_aNp1(wraparound_i, wraparound_j, wraparound_k)[1];
                    view_aNp1(i, j, k)[2] = view_aNp1(wraparound_i, wraparound_j, wraparound_k)[2];

                    view_phiNp1(i, j, k) = view_phiNp1(wraparound_i, wraparound_j, wraparound_k);
                });
        //}
        Kokkos::fence();
        // evaluate E and B fields at N
        //std::cout << "Energy: " << (double)field_evaluation() << " "
        //          << (double)this->absorbed__energy << "\n";

        // store potentials at N in Nm1, and Np1 in N
        field_evaluation();
        

        bunch.E_gather.gather(*this->En_mp, bunch.R);
        bunch.B_gather.gather(*this->Bn_mp, bunch.R);
        auto gammabeta_view = bunch.gamma_beta.getView();
        auto rview = bunch.R.getView();
        auto rnp1view = bunch.R_np1.getView();
        auto E_gatherview = bunch.E_gather.getView();
        auto B_gatherview = bunch.B_gather.getView();
        constexpr scalar e_mass = 0.5110;
        constexpr scalar e_charge = -0.30282212088;
        scalar this_dt = this->dt;
        Kokkos::parallel_for(bunch.getLocalNum(), KOKKOS_LAMBDA(const size_t i){
            const ippl::Vector<scalar, 3> pgammabeta = gammabeta_view(i);
            
            
            const ippl::Vector<scalar, 3> t1 = pgammabeta - e_charge * this_dt * E_gatherview(i) / (2.0 * e_mass); 
            const scalar alpha = -e_charge * this_dt / (2 * e_mass * Kokkos::sqrt(1 + dot_prod(t1, t1)));
            const ippl::Vector<scalar, 3> t2 = t1 + alpha * ippl::cross(t1, B_gatherview(i));

            const ippl::Vector<scalar, 3> t3 = t1 + ippl::cross(t2, 2.0 * alpha * (B_gatherview(i) / (1.0 + alpha * alpha * dot_prod(B_gatherview(i), B_gatherview(i)))));
            const ippl::Vector<scalar, 3> ngammabeta = t3 - e_charge * this_dt * E_gatherview(i) / (2.0 * e_mass);

            rnp1view(i) = rview(i) + this_dt * ngammabeta / (Kokkos::sqrt(1.0 + dot_prod(ngammabeta, ngammabeta)));            
        });
        //auto crpd = bunch.Q * ippl::cross(bunch.gamma_beta, bunch.B_gather);
        //bunch.gamma_beta = bunch.gamma_beta + (crpd / e_mass + bunch.E_gather * e_charge / e_mass);
        //bunch.R_np1 = bunch.R + bunch.gamma_beta * dt;
        this->JN_mp->operator=(0.0); //reset J to zero for next time step before scatter again
        //for(size_t i = 0;i < JN_mp->getView().extent(1);i++)
        //    JN_mp->getView()(20, i, 20) = ippl::Vector<scalar, 3>{0,1,0};
        bunch.Q.scatter(*this->JN_mp, bunch.R, bunch.R_np1, scalar(1.0) / dt);
        
        //bunch.layout_m->update();
        Kokkos::deep_copy(bunch.R.getView(), bunch.R_np1.getView());
        bunch_type bunch_buffer(pl);
        pl.update(bunch, bunch_buffer);
        //Copy potential at N-1 and N+1 to phiNm1 and phiN for next time step
        Kokkos::deep_copy(aNm1_m.getView(), aN_m.getView());
        Kokkos::deep_copy(aN_m.getView(), aNp1_m.getView());
        Kokkos::deep_copy(phiNm1_m.getView(), phiN_m.getView());
        Kokkos::deep_copy(phiN_m.getView(), phiNp1_m.getView()); 
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    Tfields FDTDSolver<Tfields, Dim, M, C>::field_evaluation() {
        // magnetic field is the curl of the vector potential
        // we take the average of the potential at N and N+1
        auto Aview   = this->aN_m.getView();
        auto Ap1view = this->aNp1_m.getView();
        if (false){}
            /*lambda_dispatch(
                *Bn_mp, 1,
                KOKKOS_LAMBDA(size_t i, size_t j, size_t k, boundary_occlusion occ) {
                    if (__builtin_popcount((unsigned int)occ) == 1) {
                        switch (occ) {
                            case x_min:
                                Aview(i, j, k) = Aview(i + 1, j, k) * 2.0 - Aview(i + 2, j, k);
                                Ap1view(i, j, k) =
                                    Ap1view(i + 1, j, k) * 2.0 - Ap1view(i + 2, j, k);
                                break;
                            case x_max:
                                Aview(i, j, k) = Aview(i - 1, j, k) * 2.0 - Aview(i - 2, j, k);
                                Ap1view(i, j, k) =
                                    Ap1view(i - 1, j, k) * 2.0 - Ap1view(i - 2, j, k);
                                break;
                            case y_min:
                                Aview(i, j, k) = Aview(i, j + 1, k) * 2.0 - Aview(i, j + 2, k);
                                Ap1view(i, j, k) =
                                    Ap1view(i, j + 1, k) * 2.0 - Ap1view(i, j + 2, k);
                                break;
                            case y_max:
                                Aview(i, j, k) = Aview(i, j - 1, k) * 2.0 - Aview(i, j - 2, k);
                                Ap1view(i, j, k) =
                                    Ap1view(i, j - 1, k) * 2.0 - Ap1view(i, j - 2, k);
                                break;
                            case z_min:
                                Aview(i, j, k) = Aview(i, j, k + 1) * 2.0 - Aview(i, j, k + 2);
                                Ap1view(i, j, k) =
                                    Ap1view(i, j, k + 1) * 2.0 - Ap1view(i, j, k + 2);
                                break;
                            case z_max:
                                Aview(i, j, k) = Aview(i, j, k - 1) * 2.0 - Aview(i, j, k - 2);
                                Ap1view(i, j, k) =
                                    Ap1view(i, j, k - 1) * 2.0 - Ap1view(i, j, k - 2);
                                break;
                        }
                    }
                },
                KOKKOS_LAMBDA(size_t i, size_t j, size_t k) { CAST_TO_VOID(i, j, k); });*/

        (*Bn_mp) = 0.5 * (curl(aN_m) + curl(aNp1_m));
        //auto bdiv = ippl::div(*Bn_mp);
        //scalar divi = volumetric_integral(KOKKOS_LAMBDA(size_t i,  size_t j,  size_t k, scalar x, scalar y, scalar z) -> scalar{
        //    (void)x;
        //    (void)y;
        //    (void)z;
        //    if(boundary_distance(i,j,k,Bn_mp->getView().extent(0),Bn_mp->getView().extent(1), Bn_mp->getView().extent(2)) > 0)
        //        return bdiv(i, j, k);
        //    return 0.0;
        //});
        ///std::cout << "Divergence Integral = " << divi << "\n";
        // electric field is the time derivative of the vector potential
        // minus the gradient of the scalar potential
        (*En_mp) = -(aNp1_m - aN_m) / dt - grad(phiN_m);

        //std::cout << En_mp->getView()(5,5,5) << "\n";
        
        return 0.0;
        auto Bview = Bn_mp->getView();
        auto Eview = En_mp->getView();

        // std::cout << hr_m << " spac\n";
        /*scalar maxE = 0;
        double maxB = 0;
        size_t discard = 2;
        for(size_t i = discard;i < Bview.extent(0) - discard;i++){
            for(size_t j = discard;j < Bview.extent(1) - discard;j++){
                for(size_t k = discard;k < Bview.extent(2) - discard;k++){
                    maxB = std::max({maxB, std::abs(Bview(i, j, k)[0])
                                         , std::abs(Bview(i, j, k)[1])
                                         , std::abs(Bview(i, j, k)[2])});
                }
            }
        }
        for(size_t i = discard;i < Eview.extent(0) - discard;i++){
            for(size_t j = discard;j < Eview.extent(1) - discard;j++){
                for(size_t k = discard;k < Eview.extent(2) - discard;k++){
                    maxE = std::max({maxE, std::abs(Eview(i, j, k)[0])
                                         , std::abs(Eview(i, j, k)[1])
                                         , std::abs(Eview(i, j, k)[2])});
                }
            }
        }*/
        // std::printf("maxB, maxE: %f, %f\n", maxB, maxE);

        Kokkos::View<scalar***> energy_density("Energies", Eview.extent(0), Eview.extent(1),
                                               Eview.extent(2));
        Kokkos::View<ippl::Vector<scalar, 3>***> radiation_density(
            "Radiations", Eview.extent(0), Eview.extent(1), Eview.extent(2));
        Kokkos::parallel_for(
            ippl::getRangePolicy(energy_density), KOKKOS_LAMBDA(size_t i, size_t j, size_t k) {
                ippl::Vector<scalar, 3> E = Eview(i, j, k);
                ippl::Vector<scalar, 3> B = Bview(i, j, k);
                // std::cout << "j = " << j << "\n";
                ippl::Vector<scalar, 3> poynting = ippl::cross(E, B);
                //std::cout << dot_prod(E, B) << "\n";
                energy_density(i, j, k)    = (dot_prod(B, B) + dot_prod(E, E)) * 0.5;
                radiation_density(i, j, k) = poynting;
            });
        Kokkos::fence();

        /*lambda_dispatch(*Bn_mp, 1, KOKKOS_LAMBDA(size_t i, size_t j, size_t k, boundary_occlusion
        occ){
            //std::cout << "Reached\n";
            if(occ == boundary_occlusion::y_min){
                ippl::Vector<scalar, 3> E = Eview(i, j, k);
                ippl::Vector<double, 3> B = Bview(i, j, k);
                //std::cout << "j = " << j << "\n";
                ippl::Vector<double, 3> poynting = cross(E, B);
                energies(i, j, k) = (squaredNorm(B) + squaredNorm(E)) * hr_m[0] * hr_m[1] * hr_m[2];
                std::cout << "poynting " << poynting[1] << "\n";
                *pointer_to_absorbed_energy_xD += (squaredNorm(B) + squaredNorm(E)) * hr_m[0] *
        hr_m[2] * dt; //Because the surface normal is negative -y unit vector
            }
        },
        KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
            (void)i;(void)j;(void)k;
        }, 2);*/

        scalar totalenergy       = 0.0;
        accumulation_type absorb = 0.0;
        if(false)
        Kokkos::parallel_reduce(
            ippl::getRangePolicy(energy_density, 12),
            KOKKOS_LAMBDA(size_t i, size_t j, size_t k, scalar& ref) {
                ref += energy_density(i, j, k);
            },
            totalenergy);
        if(false)
        Kokkos::parallel_reduce(
            ippl::getRangePolicy(energy_density, 2),
            KOKKOS_CLASS_LAMBDA(size_t i, size_t j, size_t k, accumulation_type & ref) {
                std::array<axis_aligned_occlusion, 3> ijk_occlusion_for_cells_before_abc =
                    boundary_occlusion_of(
                        12, std::make_tuple(i, j, k),
                        std::make_tuple(radiation_density.extent(0), radiation_density.extent(1),
                                        radiation_density.extent(2)));
                /*if((i == 5 || j == 5 || k == 5)){
                    printf("%d %d %d\n", ijk_occlusion_for_cells_before_abc[0],
                ijk_occlusion_for_cells_before_abc[1], ijk_occlusion_for_cells_before_abc[2]);
                }
                else{

                }*/
                const scalar volume = hr_m[0] * hr_m[1] * hr_m[2];
                ippl::Vector<scalar, 3> normal(0.0);
                for (size_t d = 0; d < 3; d++) {
                    // bool skip = false;
                    if (ijk_occlusion_for_cells_before_abc[d] & AT_MAX) {
                        normal[d] = 1.0;
                    }
                    if (ijk_occlusion_for_cells_before_abc[d] & AT_MIN) {
                        normal[d] = -1.0;
                    }
                    if (ijk_occlusion_for_cells_before_abc[d] == 0) {
                        normal[d] = 0.0;
                    }
                    if (ref != 0.0
                        && dot_prod(normal, radiation_density(i, j, k)) * volume / hr_m[d] != 0.0
                        && d != 1 && normal[d] != 0.0) {
                        scalar ratio = ((dot_prod(normal, radiation_density(i, j, k)) * volume / hr_m[d]));
                        // if(std::abs(ratio) > 1000.0)
                        // printf("Ratio: %f", ratio);
                        (void)ratio;
                    }
                    ref += dot_prod(normal, radiation_density(i, j, k)) * volume / hr_m[d];
                    //if (false) {
                    //    char buffer[4096] = {0};
                    //    char* bp          = buffer;
                    //    if(d == 1 && normal[d] == -1.0 && i == nr_m[0] / 2 && k == nr_m[2] / 2 /*&& dot_prod(Eview(i, j, k), Bview(i, j, k)) != 0.0*/){
                    //        bp += sprintf(bp, "E: %f, %f, %f\n", Eview(i, j, k)[0],
                    //                      Eview(i, j, k)[1], Eview(i, j, k)[2]);
                    //        bp += sprintf(bp, "B: %f, %f, %f\n", Bview(i, j, k)[0],
                    //                      Bview(i, j, k)[1], Bview(i, j, k)[2]);
                    //        bp += sprintf(bp, "Dot product: %f\n",
                    //                      dot_prod(Eview(i, j, k), Bview(i, j, k)));
                    //        ippl::Vector<scalar, 3> cross_prod =
                    //            ippl::cross(Eview(i, j, k), Bview(i, j, k));
                    //        bp += sprintf(bp, "Cross: %f, %f, %f\n", cross_prod[0], cross_prod[1],
                    //                      cross_prod[2]);
                    //        bp += sprintf(bp, "Normal: %f, %f, %f\n", normal[0], normal[1],
                    //                      normal[2]);
                    //        bp += sprintf(bp, "Dot: %f\n\n", dot_prod(cross_prod, normal));
                    //        puts(buffer);
                    //    }
                    //}
                    // else{
                    //     skip = true;
                    // }
                }
            },
            absorb);
        Kokkos::fence();
        this->total_energy = totalenergy * hr_m[0] * hr_m[1] * hr_m[2];
        this->absorbed__energy -= absorb * dt;
        return totalenergy * hr_m[0] * hr_m[1] * hr_m[2];
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    KOKKOS_FUNCTION Tfields FDTDSolver<Tfields, Dim, M, C>::gaussian(size_t it, size_t i, size_t j,
                                                    size_t k) const noexcept {
        // return 1.0;
        const scalar y = 1.0 - (j - 2) * hr_m[1];  // From the max boundary; fix this sometime
        const scalar t = it * dt;
        scalar plane_wave_excitation_x = (y - t < 0) ? -(y - t) : 0;
        (void)i;
        (void)j;
        (void)k;
        return 100 * Kokkos::exp(-sq((2.0 - plane_wave_excitation_x)) * 2);

        // scalar arg = Kokkos::pow((1.0 - it * dt) / 0.25, 2);
        // return 100 * Kokkos::exp(-arg);
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    void FDTDSolver<Tfields, Dim, M, C>::initialize() {

        constexpr scalar c        = 1.0;  // 299792458.0;
        constexpr scalar mu0      = 1.0;  // 1.25663706212e-6;
        constexpr scalar epsilon0 = 1.0 / (c * c * mu0);
        constexpr scalar electron_charge = 0.30282212088;
        bunch.create(!!!ippl::Comm->rank());
        bunch.Q = electron_charge;
        bunch.R = ippl::Vector<scalar, 3>(0.4);
        bunch.gamma_beta = ippl::Vector<scalar, 3>{0.0, 1e6, 0.0};
        bunch.R_np1 = ippl::Vector<scalar, 3>(0.4);
        // get layout and mesh
        layout_mp = &(this->rhoN_mp->getLayout());
        mesh_mp   = &(this->rhoN_mp->get_mesh());

        // get mesh spacing, domain, and mesh size
        hr_m     = mesh_mp->getMeshSpacing();
        domain_m = layout_mp->getDomain();
        for (unsigned int i = 0; i < Dim; ++i)
            nr_m[i] = domain_m[i].length();
        // initialize fields
        phiNm1_m.initialize(*mesh_mp, *layout_mp);
        phiN_m.initialize(*mesh_mp, *layout_mp);
        phiNp1_m.initialize(*mesh_mp, *layout_mp);

        aNm1_m.initialize(*mesh_mp, *layout_mp);
        aN_m.initialize(*mesh_mp, *layout_mp);
        aNp1_m.initialize(*mesh_mp, *layout_mp);

        phiNm1_m = 0.0;
        phiN_m   = 0.0;
        phiNp1_m = 0.0;

        aNm1_m = 0.0;
        aN_m   = 0.0;
        aNp1_m = 0.0;
    };
    template<typename Tfields, unsigned Dim, class M, class C>
    template<typename callable>
    void FDTDSolver<Tfields, Dim, M, C>::fill_initialcondition(callable c){
        auto view_a      = aN_m.getView();
        auto view_an1    = aNm1_m.getView();
        auto ldom = layout_mp->getLocalNDIndex();
        std::cout << "Rank " << ippl::Comm->rank() << " has y offset " << ldom[1].first() << "\n";
        int nghost = aN_m.getNghost();
        Kokkos::parallel_for(
            "Assign sinusoidal source at center", ippl::getRangePolicy(view_a), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k){
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                scalar x = (scalar(ig) + 0.5) * hr_m[0];// + origin[0];
                scalar y = (scalar(jg) + 0.5) * hr_m[1];// + origin[1];
                scalar z = (scalar(kg) + 0.5) * hr_m[2];// + origin[2];
                view_a  (i, j, k) = c(x, y, z);
                view_an1(i, j, k) = c(x, y, z);
            }
        );
    }
    template<typename Tfields, unsigned Dim, class M, class C>
    template<typename callable>
    Tfields FDTDSolver<Tfields, Dim, M, C>::volumetric_integral(callable c){
        auto view_a      = aN_m.getView();
        auto view_an1    = aNm1_m.getView();
        auto ldom = layout_mp->getLocalNDIndex();
        int nghost = aN_m.getNghost();
        Tfields ret = 0.0;
        Tfields volume = hr_m[0] * hr_m[1] * hr_m[2];
        Kokkos::parallel_reduce(
            "Assign sinusoidal source at center", ippl::getRangePolicy(view_a), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, scalar& ref){
                
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
