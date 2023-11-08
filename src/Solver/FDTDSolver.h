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

#ifndef FDTD_SOLVER_H_
#define FDTD_SOLVER_H_

#include "Types/Vector.h"

#include "Solver/BoundaryDispatch.h"
#include "Field/Field.h"

#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"
template <typename _scalar, class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    using scalar = _scalar;
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
        this->addAttribute(mass);
        this->addAttribute(gamma_beta);
        this->addAttribute(R_np1);
        this->addAttribute(E_gather);
        this->addAttribute(B_gather);
    }

    ~Bunch() {}
    
    using charge_container_type   = ippl::ParticleAttrib             <scalar    >;
    using velocity_container_type = ippl::ParticleAttrib<ippl::Vector<scalar, 3>>;
    using vector_container_type   = ippl::ParticleAttrib<ippl::Vector<scalar, 3>>;
    charge_container_type Q;
    charge_container_type mass;
    velocity_container_type gamma_beta;
    typename ippl::ParticleBase<PLayout>::particle_position_type R_np1;
    vector_container_type E_gather;
    vector_container_type B_gather;
};
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
    template <typename Tfields, unsigned Dim, class M = UniformCartesian<Tfields, Dim>,
              class C = typename M::DefaultCentering>
    class FDTDSolver {
        using scalar = Tfields;
    public:
        // define a type for scalar field (e.g. charge density field)
        // define a type for vectors
        // define a type for vector field
        typedef Field<Tfields, Dim, M, C> Field_t;
        typedef Vector<Tfields, Dim> Vector_t;
        typedef Field<Vector_t, Dim, M, C> VField_t;

        ippl::Vector<ippl::FieldBC, Dim> bconds;
        using memory_space = typename Field_t::memory_space;

        // define type for field layout
        typedef FieldLayout<Dim> FieldLayout_t;

        // type for communication buffers
        using buffer_type = Communicate::buffer_type<memory_space>;

        // constructor and destructor
        FDTDSolver(Field_t* charge, VField_t* current, VField_t* E, VField_t* B,
                   scalar timestep = 0.05, VField_t* radiation = nullptr, bool seed_ = false);
        ~FDTDSolver();

        // finite differences time domain solver for potentials (A and phi)
        void solve();

        // evaluates E and B fields using computed potentials
        Tfields field_evaluation();

        // gaussian pulse
        KOKKOS_FUNCTION scalar gaussian(size_t it, size_t i, size_t j, size_t k)const noexcept;

        // initialization of FDTD solver
        void initialize();

        template<typename callable>
        void fill_initialcondition(callable c);
        template<typename callable>
        Tfields volumetric_integral(callable c);

    public:
        // mesh and layout objects
        M* mesh_mp;
        FieldLayout_t* layout_mp;

        // computational domain
        NDIndex<Dim> domain_m;

        // mesh spacing and mesh size
        Vector_t hr_m;
        Vector<int, Dim> nr_m;

        // size of timestep
        scalar dt;

        // seed flag
        bool seed;

        // iteration number for gaussian seed
        size_t iteration = 0;

        // fields containing reference to charge and current
        Field_t* rhoN_mp;
        VField_t* JN_mp;

        //NULLABLE!
        VField_t* radiation;

        // scalar and vector potentials at n-1, n, n+1 times
        Field_t phiNm1_m;
        Field_t phiN_m;
        Field_t phiNp1_m;
        VField_t aNm1_m;
        VField_t aN_m;
        VField_t aNp1_m;

        // E and B fields
        VField_t* En_mp;
        VField_t* Bn_mp;
        scalar total_energy;
        using accumulation_type = scalar;
        accumulation_type absorbed__energy;

        using playout_type = ippl::ParticleSpatialLayout<scalar, 3>;
        using bunch_type = ::Bunch<scalar, playout_type>;
        playout_type pl;
        ::Bunch<scalar, ippl::ParticleSpatialLayout<scalar, 3>> bunch;
        size_t particle_count;

        // buffer for communication
        detail::FieldBufferData<Tfields> fd_m;
    };
}  // namespace ippl

#include "FDTDSolver.hpp"

#endif
