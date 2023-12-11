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

#include <Kokkos_Core_fwd.hpp>
#include "Types/Vector.h"

#include "Field/Field.h"

#include "FDTDSolver.h"
#include <array>
#include "Field/HaloCells.h"
#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"
#include <cstring>
#define LOG(X) std::cout << from_last_slash(__FILE__) << ':' << __LINE__ << ": " << X << "\n"
inline const char* from_last_slash(const char* x){
    size_t len = std::strlen(x);
    const char* end = x + len;
    while(*(end - 1) != '/')--end;
    return end;
}
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
template <typename T>
KOKKOS_INLINE_FUNCTION ippl::Vector<T, 3> cross_prod(const ippl::Vector<T, 3>& a, const ippl::Vector<T, 3>& b) {
    ippl::Vector<T, 3> ret{0.0,0.0,0.0};
    ret[0] = a[1]*b[2]-a[2]*b[1];
    ret[1] = a[2]*b[0]-a[0]*b[2];
    ret[2] = a[0]*b[1]-a[1]*b[0];
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
                                               VField_t& B, size_t pcount, FDTDBoundaryCondition bcond, FDTDParticleUpdateRule pur, FDTDFieldUpdateRule fur, double timestep, bool seed_, VField_t* radiation) : radiation_mp(radiation), bconds_m(bcond), particle_update_m(pur), field_update_m(fur), pcount_m(pcount), pl(charge.getLayout(), charge.get_mesh()), bunch(pl), tracer_bunch(pl){
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


        //Set boundary conditions based on @bcond argument
        typename VField_t::BConds_t vector_bcs;
        typename Field_t::BConds_t scalar_bcs;

        


        // call the initialization function
        initialize();

        using scalar = Tfields;
        auto hr = this->hr_m; 
        scalar dt = this->dt;
        auto bcsetter_single = [&scalar_bcs, &vector_bcs, hr, dt, bcond]<size_t Idx>(const std::index_sequence<Idx>&){
            if(bcond == FDTDBoundaryCondition::PERIODIC){
                //std::cout << "Setting piriodic\n";
                vector_bcs[Idx] = std::make_shared<ippl::PeriodicFace<VField_t>>(Idx);
                scalar_bcs[Idx] = std::make_shared<ippl::PeriodicFace<Field_t>>(Idx);
            }
            else{
                vector_bcs[Idx] = std::make_shared<ippl::NoBcFace<VField_t>>(Idx);
                scalar_bcs[Idx] = std::make_shared<ippl::NoBcFace<Field_t>>(Idx);
            }
            return 0;
        };
        auto bcsetter = [bcsetter_single]<size_t... Idx>(const std::index_sequence<Idx...>&){
            int x = (bcsetter_single(std::index_sequence<Idx>{}) ^ ...);
            (void) x;
        };
        bcsetter(std::make_index_sequence<Dim * 2>{});
        aN_m  .setFieldBC(vector_bcs);
        aNp1_m.setFieldBC(vector_bcs);
        aNm1_m.setFieldBC(vector_bcs);

        phiN_m  .setFieldBC(scalar_bcs);
        phiNm1_m.setFieldBC(scalar_bcs);
        phiNp1_m.setFieldBC(scalar_bcs);
    }

    template <typename Tfields, unsigned Dim, class M, class C>
    FDTDSolver<Tfields, Dim, M, C>::~FDTDSolver(){};

    template<typename T>
    auto sqr(T x){return x * x;};

    template <typename Tfields, unsigned Dim, class M, class C>
    template<typename callable>
    void FDTDSolver<Tfields, Dim, M, C>::fill_initialcondition(callable c){
        //auto view_a      = aN_m  .getView();
        //auto view_an1    = aNm1_m.getView();
        auto view_a      = aN_m  .getView();
        auto view_an1    = aNm1_m.getView();
        //auto view_phi    = phiN_m.getView();
        //auto view_phin1  = phiNm1_m.getView();
        auto ldom = layout_mp->getLocalNDIndex();
        //std::cout << "Rank " << ippl::Comm->rank() << " has y offset " << ldom[1].first() << "\n";
        int nghost = aN_m.getNghost();
        Kokkos::parallel_for(
            "Assign sinusoidal source at center", ippl::getRangePolicy(view_a, 0), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k){
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                Tfields x = (Tfields(ig) + 0.5) * hr_m[0];// + origin[0];
                Tfields y = (Tfields(jg) + 0.5) * hr_m[1];// + origin[1];
                Tfields z = (Tfields(kg) + 0.5) * hr_m[2];// + origin[2];
                view_a  (i, j, k) = c(x, y, z);
                view_an1(i, j, k) = c(x, y, z);
                //view_phi(i,j,k)   = 0.0;
                //view_phin1(i,j,k) = 0.0;
                //view_a  (i, j, k) = c(x, y, z);
                //view_an1(i, j, k) = c(x, y, z);
                
            }
        );
    }

    template <typename Tfields, unsigned Dim, class M, class C>
    void FDTDSolver<Tfields, Dim, M, C>::solve() {
        
        // physical constant
        using scalar = Tfields;

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
        //double beta0[3] = {(c * dt - hr_m[0]) / (c * dt + hr_m[0]),
        //                   (c * dt - hr_m[1]) / (c * dt + hr_m[1]),
        //                   (c * dt - hr_m[2]) / (c * dt + hr_m[2])};
        //double beta1[3] = {2.0 * hr_m[0] / (c * dt + hr_m[0]),
        //                   2.0 * hr_m[1] / (c * dt + hr_m[1]),
        //                   2.0 * hr_m[2] / (c * dt + hr_m[2])};
        //double beta2[3] = {-1.0, -1.0, -1.0};

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


        //Apply Boundary conditions, this will only do something in case of periodic
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
        (*rhoN_mp) = 0.0;
        bunch.Q.scatter(*rhoN_mp, bunch.R);
        if(field_update_m == FDTDFieldUpdateRule::DO){
        Kokkos::parallel_for(
            "Scalar potential update", ippl::getRangePolicy(view_phiN, nghost_phi),
            KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                // global indices
                //const int ig = i + ldom[0].first() - nghost_phi;
                //const int jg = j + ldom[1].first() - nghost_phi;
                //const int kg = k + ldom[2].first() - nghost_phi;

                // interior values
                bool isInterior = true;//((ig > 0) && (jg > 0) && (kg > 0) && (ig < nr_m[0] - 1)
                                   //&& (jg < nr_m[1] - 1) && (kg < nr_m[2] - 1));
                double interior = -view_phiNm1(i, j, k) + a1 * view_phiN(i, j, k)
                                  + a2 * (view_phiN(i + 1, j, k) + view_phiN(i - 1, j, k))
                                  + a4 * (view_phiN(i, j + 1, k) + view_phiN(i, j - 1, k))
                                  + a6 * (view_phiN(i, j, k + 1) + view_phiN(i, j, k - 1))
                                  + a8 * (view_rhoN(i, j, k) / epsilon0);

                view_phiNp1(i, j, k) = isInterior * interior + !isInterior * view_phiNp1(i, j, k);
            });

        for (size_t gd = 0; gd < Dim; ++gd) {
            Kokkos::parallel_for(
                "Vector potential update", ippl::getRangePolicy(view_aN, nghost_a),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // global indices
                    //const int ig = i + ldom[0].first() - nghost_a;
                    //const int jg = j + ldom[1].first() - nghost_a;
                    //const int kg = k + ldom[2].first() - nghost_a;

                    // interior values
                    bool isInterior = true;//((ig > 0) && (jg > 0) && (kg > 0) && (ig < nr_m[0] - 1)
                                       //&& (jg < nr_m[1] - 1) && (kg < nr_m[2] - 1));
                    double interior = -view_aNm1(i, j, k)[gd] + a1 * view_aN(i, j, k)[gd]
                                      + a2 * (view_aN(i + 1, j, k)[gd] + view_aN(i - 1, j, k)[gd])
                                      + a4 * (view_aN(i, j + 1, k)[gd] + view_aN(i, j - 1, k)[gd])
                                      + a6 * (view_aN(i, j, k + 1)[gd] + view_aN(i, j, k - 1)[gd])
                                      + a8 * (view_JN(i, j, k)[gd] * mu0);
                    view_aNp1(i, j, k)[gd] = isInterior * interior + !isInterior * view_aNp1(i, j, k)[gd];
                });
        }

        // interior points need to have been updated before ABCs
        Kokkos::fence();

        // apply 1st order Absorbing Boundary Conditions
        // for both scalar and vector potentials
        first_order_abc<scalar, 0, 1, 2> abc_x[] = {first_order_abc<scalar, 0, 1, 2> (hr_m, c, dt,  1), first_order_abc<scalar, 0, 1, 2>(hr_m, c, dt, -1)};
        first_order_abc<scalar, 1, 0, 2> abc_y[] = {first_order_abc<scalar, 1, 0, 2> (hr_m, c, dt,  1), first_order_abc<scalar, 1, 0, 2>(hr_m, c, dt, -1)};
        first_order_abc<scalar, 2, 0, 1> abc_z[] = {first_order_abc<scalar, 2, 0, 1> (hr_m, c, dt,  1), first_order_abc<scalar, 2, 0, 1>(hr_m, c, dt, -1)};
        ippl::Vector<size_t, 3> extenz = ippl::Vector<size_t, 3>{(size_t)nr_m[0] + nghost_a * 2, (size_t)nr_m[1] + nghost_a * 2, (size_t)nr_m[2] + nghost_a * 2};
        
        if(bconds_m == FDTDBoundaryCondition::ABC_MUR){
            Kokkos::parallel_for(
                "Scalar potential ABCs", ippl::getRangePolicy(view_phiN, 0),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // global indices
                    const int ig = i + ldom[0].first();
                    const int jg = j + ldom[1].first();
                    const int kg = k + ldom[2].first();
                    ippl::Vector<axis_aligned_occlusion, 3> bocc = boundary_occlusion_of(0, ippl::Vector<size_t, 3>{(size_t)ig, (size_t)jg, (size_t)kg}, extenz);
                    // boundary values: 1st order Absorbing Boundary Conditions

                    
                    for(unsigned _d = 0;_d < Dim;_d++){
                        if(+bocc[_d]){
                            int offs = bocc[_d] == AT_MIN ? 1 : -1;
                            if(_d == 0){
                                view_phiNp1(i, j, k) = view_phiN(i, j, k) + (view_phiN(i + offs, j, k) - view_phiN(i, j, k)) * this->dt / hr_m[0];
                            }
                            if(_d == 1){
                                view_phiNp1(i, j, k) = view_phiN(i, j, k) + (view_phiN(i, j + offs, k) - view_phiN(i, j, k)) * this->dt / hr_m[1];
                            }
                            if(_d == 2){
                                view_phiNp1(i, j, k) = view_phiN(i, j, k) + (view_phiN(i, j, k + offs) - view_phiN(i, j, k)) * this->dt / hr_m[2];
                            }
                        }
                    }
                }
            );
            for (size_t gd = 0; gd < Dim; ++gd) {
                Kokkos::parallel_for(
                    "Vector potential ABCs", ippl::getRangePolicy(view_aN, 0),
                    KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                        // global indices
                        const int ig = i + ldom[0].first();
                        const int jg = j + ldom[1].first();
                        const int kg = k + ldom[2].first();
                        ippl::Vector<axis_aligned_occlusion, 3> bocc = boundary_occlusion_of(0, ippl::Vector<size_t, 3>{(size_t)ig, (size_t)jg, (size_t)kg}, extenz);
                        for(unsigned _d = 0;_d < Dim;_d++){
                            if(+bocc[_d]){
                                int offs = bocc[_d] == AT_MIN ? 1 : -1;
                                if(_d == 0){
                                    view_aNp1(i, j, k) = view_aN(i, j, k) + (view_aN(i + offs, j, k) - view_aN(i, j, k)) * this->dt / hr_m[0];
                                }
                                if(_d == 1){
                                    view_aNp1(i, j, k) = view_aN(i, j, k) + (view_aN(i, j + offs, k) - view_aN(i, j, k)) * this->dt / hr_m[1];
                                }
                                if(_d == 2){
                                    view_aNp1(i, j, k) = view_aN(i, j, k) + (view_aN(i, j, k + offs) - view_aN(i, j, k)) * this->dt / hr_m[2];
                                }
                            }
                        }
                    }
                );
            }
        }




        else if(bconds_m == FDTDBoundaryCondition::ABC_FALLAHI){
            Kokkos::parallel_for(
                "Scalar potential ABCs", ippl::getRangePolicy(view_phiN, 0),
                KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                    // global indices
                    const int ig = i + ldom[0].first();
                    const int jg = j + ldom[1].first();
                    const int kg = k + ldom[2].first();
                    ippl::Vector<axis_aligned_occlusion, 3> bocc = boundary_occlusion_of(0, ippl::Vector<size_t, 3>{(size_t)ig, (size_t)jg, (size_t)kg}, extenz);
                    // boundary values: 1st order Absorbing Boundary Conditions

                    
                    for(unsigned _d = 0;_d < Dim;_d++){
                        if(+bocc[_d]){
                            if(_d == 0){
                                view_phiNp1(i, j, k) = abc_x[bocc[_d] >> 1](view_phiN, view_phiNm1, view_phiNp1, ippl::Vector<size_t, 3>({i, j, k}));
                            }
                            if(_d == 1){
                                view_phiNp1(i, j, k) = abc_y[bocc[_d] >> 1](view_phiN, view_phiNm1, view_phiNp1, ippl::Vector<size_t, 3>({i, j, k}));
                            }
                            if(_d == 2){
                                view_phiNp1(i, j, k) = abc_z[bocc[_d] >> 1](view_phiN, view_phiNm1, view_phiNp1, ippl::Vector<size_t, 3>({i, j, k}));
                            }
                        }
                    }
                }
            );
            for (size_t gd = 0; gd < Dim; ++gd) {
                Kokkos::parallel_for(
                    "Vector potential ABCs", ippl::getRangePolicy(view_aN, 0),
                    KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                        // global indices
                        const int ig = i + ldom[0].first();
                        const int jg = j + ldom[1].first();
                        const int kg = k + ldom[2].first();
                        ippl::Vector<axis_aligned_occlusion, 3> bocc = boundary_occlusion_of(0, ippl::Vector<size_t, 3>{(size_t)ig, (size_t)jg, (size_t)kg}, extenz);
                        for(unsigned _d = 0;_d < Dim;_d++){
                            if(+bocc[_d]){
                                if(_d == 0){
                                    view_aNp1(i, j, k) = abc_x[bocc[_d] >> 1](view_aN, view_aNm1, view_aNp1, ippl::Vector<size_t, 3>({i, j, k}));
                                }
                                if(_d == 1){
                                    view_aNp1(i, j, k) = abc_y[bocc[_d] >> 1](view_aN, view_aNm1, view_aNp1, ippl::Vector<size_t, 3>({i, j, k}));
                                }
                                if(_d == 2){
                                    view_aNp1(i, j, k) = abc_z[bocc[_d] >> 1](view_aN, view_aNm1, view_aNp1, ippl::Vector<size_t, 3>({i, j, k}));
                                }
                            }
                        }
                    }
                );
            }
        }
        else if(bconds_m == FDTDBoundaryCondition::PERIODIC){
            //Do nothing, as this is done by BC::apply (See above)
        }
        }
        else{
            Kokkos::deep_copy(aNp1_m.getView(), aN_m.getView());
            Kokkos::deep_copy(phiNp1_m.getView(), phiN_m.getView());
        }
        Kokkos::fence();
        //phiN_m = 0.0;
        //phiNp1_m = 0.0;
        field_evaluation();
        
        Kokkos::fence();
        bunch.E_gather.gather(*(this->En_mp), bunch.R);
        bunch.B_gather.gather(*(this->Bn_mp), bunch.R);
        Kokkos::fence();
        constexpr scalar e_mass = 0.5110;
        auto Qview = bunch.Q.getView();
        auto rview = bunch.R.getView();
        auto rnp1view = bunch.R_np1.getView();

        auto E_gatherview = bunch.E_gather.getView();
        auto B_gatherview = bunch.B_gather.getView();
        auto gammabeta_view = bunch.gamma_beta.getView();

        const int trank = ippl::Comm->rank();
        (void)trank;
        const scalar this_dt = this->dt;
        const scalar time = this_dt * iteration;
        const size_t iter = iteration;
        (void)iter;
        switch(this->particle_update_m){
            case FDTDParticleUpdateRule::LORENTZ:{
                Kokkos::parallel_for(Kokkos::RangePolicy<typename playout_type::RegionLayout_t::view_type::execution_space>(0, bunch.getLocalNum()), KOKKOS_LAMBDA(const size_t i){
                    using Kokkos::sqrt;
                    //LOG("Egather: " << E_gatherview(i));
                    //LOG("Bgather: " << B_gatherview(i));
                    const scalar charge = -Qview(i);
                    const ippl::Vector<scalar, 3> pgammabeta = gammabeta_view(i);
                    const ippl::Vector<scalar, 3> t1 = pgammabeta - charge * this_dt * E_gatherview(i) / (2.0 * e_mass); 
                    const scalar alpha = -charge * this_dt / (scalar(2) * e_mass * sqrt(1 + dot_prod(t1, t1)));
                    ippl::Vector<scalar, 3> crossprod = alpha * ippl::cross(t1, B_gatherview(i));
                    const ippl::Vector<scalar, 3> t2 = t1 + alpha * ippl::cross(t1, B_gatherview(i));
                    const ippl::Vector<scalar, 3> t3 = t1 + ippl::cross(t2, 2.0 * alpha * (B_gatherview(i) / (1.0 + alpha * alpha * dot_prod(B_gatherview(i), B_gatherview(i)))));
                    const ippl::Vector<scalar, 3> ngammabeta = t3 - charge * this_dt * E_gatherview(i) / (2.0 * e_mass);
                    rnp1view(i) = rview(i) + this_dt * ngammabeta / (sqrt(1.0 + dot_prod(ngammabeta, ngammabeta)));
                    gammabeta_view(i) = ngammabeta;
                });
            }break;
            case FDTDParticleUpdateRule::DIPOLE_ORBIT:{
                using Kokkos::sin;
                scalar yvel = sin(time) * 0.2;
                Kokkos::parallel_for(Kokkos::RangePolicy<typename playout_type::RegionLayout_t::view_type::execution_space>(0, bunch.getLocalNum()), KOKKOS_LAMBDA(const size_t i){
                    using Kokkos::sqrt;
                    rnp1view(i) = rview(i);
                    rnp1view(i)[1] = rview(i)[1] + this_dt * yvel;
                    //if(_iter == 0){
                    //    std::cout << rnp1view(i) << " from " << rview(i) << std::endl;
                    //}
                });
            }
            break;
            case FDTDParticleUpdateRule::CIRCULAR_ORBIT:{
                using Kokkos::sin;
                using Kokkos::cos;
                const scalar xpos  = cos(time) * 0.3 + 0.5;
                const scalar ypos  = sin(time) * 0.3 + 0.5;

                const scalar xposn = cos(time + dt) * 0.3 + 0.5;
                const scalar yposn = sin(time + dt) * 0.3 + 0.5;
                LOG("XP: " << rnp1view(0)[0]);
                const scalar xd = xposn - xpos;
                const scalar yd = yposn - ypos;
                Kokkos::parallel_for(Kokkos::RangePolicy<typename playout_type::RegionLayout_t::view_type::execution_space>(0, bunch.getLocalNum()), KOKKOS_LAMBDA(const size_t i){
                    using Kokkos::sqrt;
                    rnp1view(i) = rview(i);
                    rnp1view(i)[0] = rview(i)[0] + xd;
                    rnp1view(i)[1] = rview(i)[1] + yd;
                    //if(_iter == 0){
                    //    std::cout << rnp1view(i) << " from " << rview(i) << std::endl;
                    //}
                });
            }
            case FDTDParticleUpdateRule::STATIONARY:{
                
            }break;
        }
        this->JN_mp->operator=(0.0);
        bunch.R_np12 = (bunch.R + bunch.R_np1) * 0.5;
        bunch.R_nm12 = (bunch.R + bunch.R_nm1) * 0.5;
        bunch.Q.scatter(*this->JN_mp, bunch.R_nm12, bunch.R_np12, scalar(1.0) / (dt * hr_m[0] * hr_m[1] * hr_m[2]));

        Kokkos::deep_copy(bunch.R_nm1.getView(), bunch.R.getView());
        Kokkos::deep_copy(bunch.R.getView(), bunch.R_np1.getView());

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
        bunch.destroy(invalid, invalid_count);

        // evaluate E and B fields at N
        
        //std::cout << "Energy: " << field_evaluation() << " " << this->absorbed__energy << "\n";

        // store potentials at N in Nm1, and Np1 in N
        if(field_update_m == FDTDFieldUpdateRule::DO){
            Kokkos::deep_copy(aNm1_m.getView(), aN_m.getView());
            Kokkos::deep_copy(aN_m.getView(), aNp1_m.getView());
            Kokkos::deep_copy(phiNm1_m.getView(), phiN_m.getView());
            Kokkos::deep_copy(phiN_m.getView(), phiNp1_m.getView());
        }
        ++iteration;
    };

    template <typename Tfields, unsigned Dim, class M, class C>
    double FDTDSolver<Tfields, Dim, M, C>::field_evaluation(){
        // magnetic field is the curl of the vector potential
        // we take the average of the potential at N and N+1
        auto Aview = this->aN_m.getView();
        auto Ap1view = this->aNp1_m.getView();
        
        (*Bn_mp) = 0.5 * (curl(aN_m) + curl(aNp1_m));

        // electric field is the time derivative of the vector potential
        // minus the gradient of the scalar potential
        (*En_mp) = -(aNp1_m - aN_m) / dt - grad(phiN_m);
        if(radiation_mp){
            (*radiation_mp) = ippl::cross(*En_mp, *Bn_mp);
        }
        //return 0.0;
        auto Bview = Bn_mp->getView();
        auto Eview = En_mp->getView();
        tracer_bunch.E_gather.gather(*(this->En_mp), tracer_bunch.R);
        tracer_bunch.B_gather.gather(*(this->Bn_mp), tracer_bunch.R);
        auto tbev = tracer_bunch.E_gather.getView();
        auto tbbv = tracer_bunch.B_gather.getView();
        auto tnbv = tracer_bunch.outward_normal.getView();
        Tfields radiation_on_boundary = 0.0;
        Kokkos::parallel_reduce(tracer_bunch.getLocalNum(), KOKKOS_LAMBDA(size_t i, Tfields& ref){
            ref += dot_prod(tnbv(i), cross_prod(tbev(i), tbbv(i)));
        }, radiation_on_boundary);
        radiation_on_boundary *= 4.0 * M_PI * 0.5 * 0.5 / tracer_bunch.getLocalNum();
        LOG("Boundary radiation: " << radiation_on_boundary);
        absorbed__energy += dt * radiation_on_boundary;
        LOG("Cumulative radiation: " << this->absorbed__energy);
        Kokkos::fence();
        ippl::Vector<Tfields, Dim> Jsum(0.0);
        auto jview = JN_mp->getView();
        Kokkos::parallel_reduce(ippl::getRangePolicy(jview, 0), KOKKOS_LAMBDA(size_t i, size_t j, size_t k, ippl::Vector<Tfields, Dim>& ref){
            ref += jview(i,j,k);
        }, Jsum);
        Jsum *= hr_m[0] * hr_m[1] * hr_m[2];
        //std::cerr << "vol_avg Q = " << rhoN_mp->sum() * hr_m[0] * hr_m[1] * hr_m[2] << "\n";
        //std::cerr << "vol_avg J = " << std::sqrt(dot_prod(Jsum, Jsum)) << "\n";
        LOG("Particle pos: " << bunch.R.getView()(0)[0] << " " << bunch.R.getView()(0)[1]);
        return 0.0;
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

        bunch.create(pcount_m);
        bunch.Q = 1.0;
        size_t tracer_count = 500 * 500 + 4;
        {
            size_t tcisqrt = (size_t)(std::sqrt((double)tracer_count));
            if(tcisqrt * tcisqrt != tracer_count){
                tracer_count = tcisqrt * tcisqrt;
                LOG("Updated tracer count to an integer square: " << tracer_count);
            }
        }
        tracer_bunch.create(tracer_count);
        auto tbrv = tracer_bunch.R.getView();
        auto tbnv = tracer_bunch.outward_normal.getView();
        Kokkos::Array<long int, 2> begin, end;
        begin[0] = 0;
        begin[1] = 0;
        end[0] = std::sqrt(tracer_count);
        end[1] = std::sqrt(tracer_count);
        const double limit = std::sqrt(tracer_count);
        constexpr Tfields radius = 0.5;
        ippl::Vector<Tfields, 3> center{0.5, 0.5, 0.5};
        Kokkos::parallel_for(ippl::createRangePolicy(begin, end), KOKKOS_LAMBDA(size_t i, size_t j){
            using Kokkos::sin;
            using Kokkos::cos;
            using Kokkos::acos;
            using Kokkos::sqrt;
            const Tfields u = Tfields(i) / limit * M_PI * 2.0;
            const Tfields v = Tfields(j) / (limit - 1);
            Tfields phi = u;
            Tfields theta = acos(2 * v - 1);
            ippl::Vector<Tfields, 3> offs{cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)};
            
            tbnv(i * size_t(limit) + j) = offs;
            offs *= radius;
            tbrv(i * size_t(limit) + j) = center + offs;

        });
    };
}  // namespace ippl
