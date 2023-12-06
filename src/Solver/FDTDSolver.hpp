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

#include "Types/Vector.h"

#include "Field/Field.h"

#include "FDTDSolver.h"
#include <array>
#include "Field/HaloCells.h"
#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"
template<typename T>
T squaredNorm(const ippl::Vector<T, 3>& a){
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}
template<typename T>
auto sq(T x){
    return x * x;
}
template<typename... Args>
void castToVoid(Args&&... args) {
    (void)(std::tuple<Args...>{args...});
}
#define CAST_TO_VOID(...) castToVoid(__VA_ARGS__)
template<typename T, unsigned Dim>
T dot_prod(const ippl::Vector<T, Dim>& a, const ippl::Vector<T, Dim>& b){
    T ret = 0.0;
    for(unsigned i = 0;i < Dim;i++){
        ret += a[i] * b[i];
    }
    return ret;
}

enum axis_aligned_occlusion : unsigned int{
    NONE = 0u, AT_MIN = 1u, AT_MAX = 2u, 
};
template<typename T>
struct axis_aligned_occlusion_maker{
    using type = axis_aligned_occlusion;
};
template<typename T>
using axis_aligned_occlusion_maker_t = typename axis_aligned_occlusion_maker<T>::type;

axis_aligned_occlusion operator|=(axis_aligned_occlusion& a, axis_aligned_occlusion b){
    a = (axis_aligned_occlusion)(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
    return a;
}
axis_aligned_occlusion& operator|=(axis_aligned_occlusion& a, unsigned int b){
    a = (axis_aligned_occlusion)(static_cast<unsigned int>(a) | b);
    return a;
}
axis_aligned_occlusion& operator&=(axis_aligned_occlusion& a, axis_aligned_occlusion b){
    a = (axis_aligned_occlusion)(static_cast<unsigned int>(a) & static_cast<unsigned int>(b));
    return a;
}
axis_aligned_occlusion& operator&=(axis_aligned_occlusion& a, unsigned int b){
    a = (axis_aligned_occlusion)(static_cast<unsigned int>(a) & b);
    return a;
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
//template<typename T>
//ippl::Vector<T, 3> cross(const ippl::Vector<T, 3>& a, const ippl::Vector<T, 3>& b){
//    ippl::Vector<T, 3> ret;
//    ret[0] = a[1]* b[2] - a[2] * b[1];
//    ret[2] = a[0] * b[1] - a[1] * b[0];
//    ret[1] = a[2] * b[0] - a[0] * b[2];
//    return ret;
//}
namespace ippl {

    template <typename Tfields, unsigned Dim, class M, class C>
    FDTDSolver<Tfields, Dim, M, C>::FDTDSolver(Field_t& charge, VField_t& current, VField_t& E,
                                               VField_t& B, double timestep, bool seed_) {
        // set the rho and J fields to be references to charge and current
        // since charge and current deposition will happen at each timestep
        rhoN_mp = &charge;
        JN_mp   = &current;

        // same for E and B fields
        En_mp = &E;
        Bn_mp = &B;

        // initialize the time-step size
        this->dt = timestep;

        // set the seed flag
        this->seed = seed_;

        // call the initialization function
        initialize();
    }

    template <typename Tfields, unsigned Dim, class M, class C>
    FDTDSolver<Tfields, Dim, M, C>::~FDTDSolver(){};

    template<typename T>
    auto sqr(T x){return x * x;};

    template <typename Tfields, unsigned Dim, class M, class C>
    void FDTDSolver<Tfields, Dim, M, C>::solve() {
        // physical constant
        //using scalar;// = typename Tfields::BareField_t::value_type::

        constexpr double c        = 1.0;  // 299792458.0;
        constexpr double mu0      = 1.0;  // 1.25663706212e-6;
        constexpr double epsilon0 = 1.0 / (c * c * mu0);

        // finite differences constants
        double a1 = 2.0
                    * (1.0 - std::pow(c * dt / hr_m[0], 2) - std::pow(c * dt / hr_m[1], 2)
                       - std::pow(c * dt / hr_m[2], 2));
        double a2 = std::pow(c * dt / hr_m[0], 2);  // a3 = a2
        double a4 = std::pow(c * dt / hr_m[1], 2);  // a5 = a4
        double a6 = std::pow(c * dt / hr_m[2], 2);  // a7 = a6
        double a8 = std::pow(c * dt, 2);

        // 1st order absorbing boundary conditions constants
        double beta0[3] = {(c * dt - hr_m[0]) / (c * dt + hr_m[0]),
                           (c * dt - hr_m[1]) / (c * dt + hr_m[1]),
                           (c * dt - hr_m[2]) / (c * dt + hr_m[2])};
        double beta1[3] = {2.0 * hr_m[0] / (c * dt + hr_m[0]),
                           2.0 * hr_m[1] / (c * dt + hr_m[1]),
                           2.0 * hr_m[2] / (c * dt + hr_m[2])};
        double beta2[3] = {-1.0, -1.0, -1.0};

        // preliminaries for Kokkos loops (ghost cells and views)
        phiNp1_m = 0.0;
        aNp1_m = 0.0;
        auto view_phiN   = phiN_m.getView();
        auto view_phiNm1 = phiNm1_m.getView();
        auto view_phiNp1 = phiNp1_m.getView();

        auto view_aN   = aN_m.getView();
        auto view_aNm1 = aNm1_m.getView();
        auto view_aNp1 = aNp1_m.getView();

        auto view_rhoN = this->rhoN_mp->getView();
        auto view_JN   = this->JN_mp->getView();

        const int nghost_phi = phiN_m.getNghost();
        const int nghost_a   = aN_m.getNghost();
        const auto& ldom     = layout_mp->getLocalNDIndex();
        aN_m.getFieldBC().apply(aN_m);
        aNm1_m.getFieldBC().apply(aNm1_m);
        phiN_m.getFieldBC().apply(phiN_m);
        phiNm1_m.getFieldBC().apply(phiNm1_m);
        Kokkos::fence();
        // compute scalar potential and vector potential at next time-step
        // first, only the interior points are updated
        // then, if the user has set a seed, the seed is added via TF/SF boundaries
        // (TF/SF = total-field/scattered-field technique)
        // finally, absorbing boundary conditions are imposed
        Kokkos::parallel_for(
            "Scalar potential update", ippl::getRangePolicy(view_phiN, nghost_phi),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                // global indices
                const int ig = i + ldom[0].first() - nghost_phi;
                const int jg = j + ldom[1].first() - nghost_phi;
                const int kg = k + ldom[2].first() - nghost_phi;

                // interior values
                bool isInterior = true;//((ig > 0) && (jg > 0) && (kg > 0) && (ig < nr_m[0] - 1)
                                   //&& (jg < nr_m[1] - 1) && (kg < nr_m[2] - 1));
                double interior = -view_phiNm1(i, j, k) + a1 * view_phiN(i, j, k)
                                  + a2 * (view_phiN(i + 1, j, k) + view_phiN(i - 1, j, k))
                                  + a4 * (view_phiN(i, j + 1, k) + view_phiN(i, j - 1, k))
                                  + a6 * (view_phiN(i, j, k + 1) + view_phiN(i, j, k - 1))
                                  + a8 * (-view_rhoN(i, j, k) / epsilon0);

                view_phiNp1(i, j, k) = isInterior * interior + !isInterior * view_phiNp1(i, j, k);
            });

        for (size_t gd = 0; gd < Dim; ++gd) {
            Kokkos::parallel_for(
                "Vector potential update", ippl::getRangePolicy(view_aN, nghost_a),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // global indices
                    const int ig = i + ldom[0].first() - nghost_a;
                    const int jg = j + ldom[1].first() - nghost_a;
                    const int kg = k + ldom[2].first() - nghost_a;

                    // interior values
                    bool isInterior = true;//((ig > 0) && (jg > 0) && (kg > 0) && (ig < nr_m[0] - 1)
                                       //&& (jg < nr_m[1] - 1) && (kg < nr_m[2] - 1));
                    double interior = -view_aNm1(i, j, k)[gd] + a1 * view_aN(i, j, k)[gd]
                                      + a2 * (view_aN(i + 1, j, k)[gd] + view_aN(i - 1, j, k)[gd])
                                      + a4 * (view_aN(i, j + 1, k)[gd] + view_aN(i, j - 1, k)[gd])
                                      + a6 * (view_aN(i, j, k + 1)[gd] + view_aN(i, j, k - 1)[gd])
                                      + a8 * (-view_JN(i, j, k)[gd] * mu0);
                    view_aNp1(i, j, k)[gd] = isInterior * interior + !isInterior * view_aNp1(i, j, k)[gd];
                });
        }

        // interior points need to have been updated before TF/SF seed and ABCs
        Kokkos::fence();

        // add seed field via TF/SF boundaries
        if (false && seed) {
            iteration++;

            // the scattered field boundary is the 2nd point after the boundary
            // the total field boundary is the 3rd point after the boundary
            for (size_t gd = 0; gd < 1; ++gd) {
                Kokkos::parallel_for(
                    "Vector potential update", ippl::getRangePolicy(view_aN, nghost_a),
                    KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                        // global indices
                        const int ig = i + ldom[0].first() - nghost_a;
                        const int jg = j + ldom[1].first() - nghost_a;
                        const int kg = k + ldom[2].first() - nghost_a;

                        // SF boundary in all 3 dimensions
                        bool isXmin_SF = ((ig == 1) && (jg > 1) && (kg > 1) && (jg < nr_m[1] - 2)
                                          && (kg < nr_m[2] - 2));
                        double xmin_SF = a2 * gaussian(iteration, i, j, k);
                        bool isYmin_SF = ((ig > 1) && (jg == 1) && (kg > 1) && (ig < nr_m[0] - 2)
                                          && (kg < nr_m[2] - 2));
                        double ymin_SF = a4 * gaussian(iteration, i, j, k);
                        bool isZmin_SF = ((ig > 1) && (jg > 1) && (kg == 1) && (ig < nr_m[0] - 2)
                                          && (jg < nr_m[1] - 2));
                        double zmin_SF = a6 * gaussian(iteration, i, j, k);
                        bool isXmax_SF = ((ig == nr_m[0] - 2) && (jg > 1) && (kg > 1)
                                          && (jg < nr_m[1] - 2) && (kg < nr_m[2] - 2));
                        double xmax_SF = -a2 * gaussian(iteration, i, j, k);
                        bool isYmax_SF = ((ig > 1) && (jg == nr_m[1] - 2) && (kg > 1)
                                          && (ig < nr_m[0] - 2) && (kg < nr_m[2] - 2));
                        double ymax_SF = -a4 * gaussian(iteration, i, j, k);
                        bool isZmax_SF = ((ig > 1) && (jg > 1) && (kg == nr_m[2] - 2)
                                          && (ig < nr_m[0] - 2) && (jg < nr_m[1] - 2));
                        double zmax_SF = -a6 * gaussian(iteration, i, j, k);

                        // TF boundary
                        bool isXmin_TF = ((ig == 2) && (jg > 2) && (kg > 2) && (jg < nr_m[1] - 3)
                                          && (kg < nr_m[2] - 3));
                        double xmin_TF = -a2 * gaussian(iteration, i, j, k);
                        bool isYmin_TF = ((ig > 2) && (jg == 2) && (kg > 2) && (ig < nr_m[0] - 3)
                                          && (kg < nr_m[2] - 3));
                        double ymin_TF = -a4 * gaussian(iteration, i, j, k);
                        bool isZmin_TF = ((ig > 2) && (jg > 2) && (kg == 2) && (ig < nr_m[0] - 3)
                                          && (jg < nr_m[1] - 3));
                        double zmin_TF = -a6 * gaussian(iteration, i, j, k);
                        bool isXmax_TF = ((ig == nr_m[0] - 3) && (jg > 2) && (kg > 2)
                                          && (jg < nr_m[1] - 3) && (kg < nr_m[2] - 3));
                        double xmax_TF = a2 * gaussian(iteration, i, j, k);
                        bool isYmax_TF = ((ig > 2) && (jg == nr_m[1] - 3) && (kg > 2)
                                          && (ig < nr_m[0] - 3) && (kg < nr_m[2] - 3));
                        double ymax_TF = a4 * gaussian(iteration, i, j, k);
                        bool isZmax_TF = ((ig > 2) && (jg > 2) && (kg == nr_m[2] - 3)
                                          && (ig < nr_m[0] - 3) && (jg < nr_m[1] - 3));
                        double zmax_TF = a6 * gaussian(iteration, i, j, k);
                        //isXmax_SF = false;
                        //isXmax_TF = false;
                        //isXmin_SF = false;
                        //isYmin_TF = false;
                        //isYmax_SF = false;
                        //isYmax_TF = false;
                        //isYmin_SF = false;
                        //isXmin_TF = false;
                        //isZmax_SF = false;
                        //isZmax_TF = false;
                        //isZmin_SF = false;
                        //isZmin_TF = false;
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
        if(phiN_m.getFieldBC()[0]->getBCType() == NO_FACE)
        Kokkos::parallel_for(
            "Scalar potential ABCs", ippl::getRangePolicy(view_phiN, nghost_phi),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                // global indices
                const int ig = i + ldom[0].first() - nghost_phi;
                const int jg = j + ldom[1].first() - nghost_phi;
                const int kg = k + ldom[2].first() - nghost_phi;

                // boundary values: 1st order Absorbing Boundary Conditions
                bool isXmin =
                    ((ig == 0) && (jg > 0) && (kg > 0) && (jg < nr_m[1] - 1) && (kg < nr_m[2] - 1));
                double xmin = beta0[0] * (view_phiNm1(i, j, k) + view_phiNp1(i + 1, j, k))
                              + beta1[0] * (view_phiN(i, j, k) + view_phiN(i + 1, j, k))
                              + beta2[0] * (view_phiNm1(i + 1, j, k));
                bool isYmin =
                    ((ig > 0) && (jg == 0) && (kg > 0) && (ig < nr_m[0] - 1) && (kg < nr_m[2] - 1));
                double ymin = beta0[1] * (view_phiNm1(i, j, k) + view_phiNp1(i, j + 1, k))
                              + beta1[1] * (view_phiN(i, j, k) + view_phiN(i, j + 1, k))
                              + beta2[1] * (view_phiNm1(i, j + 1, k));
                bool isZmin =
                    ((ig > 0) && (jg > 0) && (kg == 0) && (ig < nr_m[0] - 1) && (jg < nr_m[1] - 1));
                double zmin = beta0[2] * (view_phiNm1(i, j, k) + view_phiNp1(i, j, k + 1))
                              + beta1[2] * (view_phiN(i, j, k) + view_phiN(i, j, k + 1))
                              + beta2[2] * (view_phiNm1(i, j, k + 1));
                bool isXmax = ((ig == nr_m[0] - 1) && (jg > 0) && (kg > 0) && (jg < nr_m[1] - 1)
                               && (kg < nr_m[2] - 1));
                double xmax = beta0[0] * (view_phiNm1(i, j, k) + view_phiNp1(i - 1, j, k))
                              + beta1[0] * (view_phiN(i, j, k) + view_phiN(i - 1, j, k))
                              + beta2[0] * (view_phiNm1(i - 1, j, k));
                bool isYmax = ((ig > 0) && (jg == nr_m[1] - 1) && (kg > 0) && (ig < nr_m[0] - 1)
                               && (kg < nr_m[2] - 1));
                double ymax = beta0[1] * (view_phiNm1(i, j, k) + view_phiNp1(i, j - 1, k))
                              + beta1[1] * (view_phiN(i, j, k) + view_phiN(i, j - 1, k))
                              + beta2[1] * (view_phiNm1(i, j - 1, k));
                bool isZmax = ((ig > 0) && (jg > 0) && (kg == nr_m[2] - 1) && (ig < nr_m[0] - 1)
                               && (jg < nr_m[1] - 1));
                double zmax = beta0[2] * (view_phiNm1(i, j, k) + view_phiNp1(i, j, k - 1))
                              + beta1[2] * (view_phiN(i, j, k) + view_phiN(i, j, k - 1))
                              + beta2[2] * (view_phiNm1(i, j, k - 1));
                bool isInterior = !(isXmin | isXmax | isYmin | isYmax | isZmin | isZmax);
                view_phiNp1(i, j, k) = isXmin * xmin + isYmin * ymin + isZmin * zmin
                                        + isXmax * xmax + isYmax * ymax + isZmax * zmax + isInterior * view_phiNp1(i, j, k);
                                    
            });
        ippl::Vector<size_t, 3> extenz = ippl::Vector<size_t, 3>{(size_t)nr_m[0] + nghost_a * 2, (size_t)nr_m[1] + nghost_a * 2, (size_t)nr_m[2] + nghost_a * 2};
        
        if(aN_m.getFieldBC()[0]->getBCType() == NO_FACE){
        for (size_t gd = 0; gd < Dim; ++gd) {
            Kokkos::parallel_for(
                "Vector potential ABCs", ippl::getRangePolicy(view_aN),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // global indices
                    const int ig = i + ldom[0].first();
                    const int jg = j + ldom[1].first();
                    const int kg = k + ldom[2].first();
                    ippl::Vector<axis_aligned_occlusion, 3> bocc = boundary_occlusion_of(0, ippl::Vector<size_t, 3>{(size_t)ig, (size_t)jg, (size_t)kg}, extenz);
                    // boundary values: 1st order Absorbing Boundary Conditions
                    if(+bocc[gd]){
                        int offs = bocc[gd] == AT_MIN ? 1 : -1;
                        if(gd == 0){
                            view_aNp1(i, j, k) = view_aN(i, j, k) + (view_aN(i + offs, j, k) - view_aN(i, j, k)) * this->dt / hr_m[0];
                        }
                        if(gd == 1){
                            view_aNp1(i, j, k) = view_aN(i, j, k) + (view_aN(i, j + offs, k) - view_aN(i, j, k)) * this->dt / hr_m[1];
                        }
                        if(gd == 2){
                            view_aNp1(i, j, k) = view_aN(i, j, k) + (view_aN(i, j, k + offs) - view_aN(i, j, k)) * this->dt / hr_m[2];
                        }
                    }
                });
        }
        }
        else{
            
        }
        Kokkos::fence();

        // evaluate E and B fields at N
        std::cout << "Energy: " << field_evaluation() << " " << this->absorbed__energy << "\n";

        // store potentials at N in Nm1, and Np1 in N
        Kokkos::deep_copy(aNm1_m.getView(), aN_m.getView());
        Kokkos::deep_copy(aN_m.getView(), aNp1_m.getView());
        Kokkos::deep_copy(phiNm1_m.getView(), phiN_m.getView());
        Kokkos::deep_copy(phiN_m.getView(), phiNp1_m.getView());
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    double FDTDSolver<Tfields, Dim, M, C>::field_evaluation(){
        // magnetic field is the curl of the vector potential
        // we take the average of the potential at N and N+1
        auto Aview = this->aN_m.getView();
        auto Ap1view = this->aNp1_m.getView();
        if(false)
        lambda_dispatch(*Bn_mp, 1, KOKKOS_LAMBDA(size_t i, size_t j, size_t k, boundary_occlusion occ){
            if(__builtin_popcount((unsigned int)occ) == 1){
                switch(occ){
                    case x_min:
                    Aview(i, j, k) = Aview(i + 1, j, k) * 2.0 - Aview(i + 2, j, k);
                    Ap1view(i, j, k) = Ap1view(i + 1, j, k) * 2.0 - Ap1view(i + 2, j, k);
                    break;
                    case x_max:
                    Aview(i, j, k) = Aview(i - 1, j, k) * 2.0 - Aview(i - 2, j, k);
                    Ap1view(i, j, k) = Ap1view(i - 1, j, k) * 2.0 - Ap1view(i - 2, j, k);
                    break;
                    case y_min:
                    Aview(i, j, k) = Aview(i, j+1, k) * 2.0 - Aview(i, j+2, k);
                    Ap1view(i, j, k) = Ap1view(i, j+1, k) * 2.0 - Ap1view(i, j+2, k);
                    break;
                    case y_max:
                    Aview(i, j, k)   = Aview  (i, j-1, k) * 2.0 - Aview  (i, j-2, k);
                    Ap1view(i, j, k) = Ap1view(i, j-1, k) * 2.0 - Ap1view(i, j-2, k);
                    break;
                    case z_min:
                    Aview(i, j, k)   = Aview  (i, j, k+1) * 2.0 - Aview  (i, j, k+2);
                    Ap1view(i, j, k) = Ap1view(i, j, k+1) * 2.0 - Ap1view(i, j, k+2);
                    break;
                    case z_max:
                    Aview(i, j, k)   = Aview  (i, j, k-1) * 2.0 - Aview  (i, j, k-2);
                    Ap1view(i, j, k) = Ap1view(i, j, k-1) * 2.0 - Ap1view(i, j, k-2);
                    break;
                }
            }
        },
        KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
            CAST_TO_VOID(i, j, k);
        });
        
        (*Bn_mp) = 0.5 * (curl(aN_m) + curl(aNp1_m));

        // electric field is the time derivative of the vector potential
        // minus the gradient of the scalar potential
        (*En_mp) = -(aNp1_m - aN_m) / dt - grad(phiN_m);
        //return 0.0;
        auto Bview = Bn_mp->getView();
        auto Eview = En_mp->getView();
        
        
        //std::cout << hr_m << " spac\n";
        /*double maxE = 0;
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
        //std::printf("maxB, maxE: %f, %f\n", maxB, maxE);
        
        Kokkos::View<double***> energy_density("Energies", Eview.extent(0), Eview.extent(1), Eview.extent(2));
        Kokkos::View<ippl::Vector<double, 3>***> radiation_density("Radiations", Eview.extent(0), Eview.extent(1), Eview.extent(2));
        Kokkos::parallel_for(ippl::getRangePolicy(energy_density), KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                ippl::Vector<double, 3> E = Eview(i, j, k);
                ippl::Vector<double, 3> B = Bview(i, j, k);
                //std::cout << "j = " << j << "\n";
                ippl::Vector<double, 3> poynting = ippl::cross(E, B);
                energy_density(i, j, k) = (dot_prod(B, B) + dot_prod(E, E));
                radiation_density(i, j, k) = poynting;
        });
        Kokkos::fence();
        
        /*lambda_dispatch(*Bn_mp, 1, KOKKOS_LAMBDA(size_t i, size_t j, size_t k, boundary_occlusion occ){
            //std::cout << "Reached\n";
            if(occ == boundary_occlusion::y_min){
                ippl::Vector<double, 3> E = Eview(i, j, k);
                ippl::Vector<double, 3> B = Bview(i, j, k);
                //std::cout << "j = " << j << "\n";
                ippl::Vector<double, 3> poynting = cross(E, B);
                energies(i, j, k) = (squaredNorm(B) + squaredNorm(E)) * hr_m[0] * hr_m[1] * hr_m[2];
                std::cout << "poynting " << poynting[1] << "\n";
                *pointer_to_absorbed_energy_xD += (squaredNorm(B) + squaredNorm(E)) * hr_m[0] * hr_m[2] * dt; //Because the surface normal is negative -y unit vector
            }
        },
        KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
            (void)i;(void)j;(void)k;
        }, 2);*/

        double totalenergy = 0.0;
        double absorb      = 0.0;

        Kokkos::parallel_reduce(ippl::getRangePolicy(energy_density, 3), KOKKOS_LAMBDA(size_t i, size_t j, size_t k, double& ref){
            ref += energy_density(i, j, k);
        }, totalenergy);

        Kokkos::parallel_reduce(ippl::getRangePolicy(energy_density, 2), KOKKOS_LAMBDA(size_t i, size_t j, size_t k, double& ref){
            ippl::Vector<axis_aligned_occlusion, 3> ijk_occlusion_for_cells_before_abc = 
            boundary_occlusion_of(
                3, ippl::Vector<size_t, 3>{i, j, k}, 
                ippl::Vector<size_t, 3>{radiation_density.extent(0), radiation_density.extent(1), radiation_density.extent(2)}
            );
            /*if((i == 5 || j == 5 || k == 5)){
                printf("%d %d %d\n", ijk_occlusion_for_cells_before_abc[0], ijk_occlusion_for_cells_before_abc[1], ijk_occlusion_for_cells_before_abc[2]);
            }
            else{
                
            }*/
            const double volume = hr_m[0] * hr_m[1] * hr_m[2];
            ippl::Vector<double, 3> normal(0.0);
            for(size_t d = 0;d < 3;d++){
                
                //bool skip = false;
                if(ijk_occlusion_for_cells_before_abc[d] & AT_MAX){
                    normal[d] = 1.0;
                }
                if(ijk_occlusion_for_cells_before_abc[d] & AT_MIN){
                    normal[d] = -1.0;
                }
                if(ijk_occlusion_for_cells_before_abc[d] == 0){
                    normal[d] = 0.0;
                }
                ref += dot_prod(normal, radiation_density(i, j, k)) * volume / hr_m[d];
                //char buffer[2048] = {0};
                //char* bp = buffer;
                //if(d == 1 && normal[d] == -1.0 && i == nr_m[0] / 2 && k == nr_m[2] / 2 /*&& dot_prod(Eview(i, j, k), Bview(i, j, k)) != 0.0*/){
                //    bp += sprintf(bp, "E: %f, %f, %f\n", Eview(i, j, k)[0], Eview(i, j, k)[1], Eview(i, j, k)[2]);
                //    bp += sprintf(bp, "B: %f, %f, %f\n", Bview(i, j, k)[0], Bview(i, j, k)[1], Bview(i, j, k)[2]);
                //    bp += sprintf(bp, "Dot product: %f\n", dot_prod(Eview(i, j, k), Bview(i, j, k)));
                //    ippl::Vector<double, 3> cross_prod = ippl::cross(Eview(i, j, k), Bview(i, j, k));
                //    bp += sprintf(bp, "Cross: %f, %f, %f\n\n", cross_prod[0], cross_prod[1], cross_prod[2]);
                //    puts(buffer);
                //}
                //else{
                //    skip = true;
                //}
            }
            
        }, absorb);
        Kokkos::fence();
        this->total_energy = totalenergy * hr_m[0] * hr_m[1] * hr_m[2];
        this->absorbed__energy -= absorb * dt;
        return totalenergy * hr_m[0] * hr_m[1] * hr_m[2];
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    double FDTDSolver<Tfields, Dim, M, C>::gaussian(size_t it, size_t i, size_t j, size_t k) const noexcept{
        //return 1.0;
        const double y = 1.0 - j * hr_m[1]; // From the max boundary; fix this sometime
        const double t = it * dt;
        double plane_wave_excitation_x = (y - t < 0) ? -(y - t) : 0;
        (void)i;
        (void)j;
        (void)k;
        return 100 * Kokkos::exp(-sq((2.0 - plane_wave_excitation_x) * 4.0));


        //double arg = Kokkos::pow((1.0 - it * dt) / 0.25, 2);
        //return 100 * Kokkos::exp(-arg);
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    void FDTDSolver<Tfields, Dim, M, C>::initialize() {
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
}  // namespace ippl
