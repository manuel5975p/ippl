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

//#include "Solver/BoundaryDispatch.h"
#include "Field/Field.h"

#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"



template <typename _scalar, class PLayout>
struct  Bunch : public ippl::ParticleBase<PLayout> {
    using scalar = _scalar;

    // Constructor for the Bunch class, taking a PLayout reference
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        // Add attributes to the particle bunch
        this->addAttribute(Q);          // Charge attribute
        this->addAttribute(mass);       // Mass attribute
        this->addAttribute(gamma_beta); // Gamma-beta attribute (product of relativistiv gamma and beta)
        this->addAttribute(R_np1);      // Position attribute for the next time step
        this->addAttribute(R_np12);      // Position attribute for the next time step
        this->addAttribute(R_nm1);      // Position attribute for the next time step
        this->addAttribute(R_nm12);      // Position attribute for the next time step
        this->addAttribute(E_gather);   // Electric field attribute for particle gathering
        this->addAttribute(B_gather);   // Magnetic field attribute for particle gathering
    }

    // Destructor for the Bunch class
    ~Bunch() {}

    // Define container types for various attributes
    using charge_container_type   = ippl::ParticleAttrib<scalar>;
    using velocity_container_type = ippl::ParticleAttrib<ippl::Vector<scalar, 3>>;
    using vector_container_type   = ippl::ParticleAttrib<ippl::Vector<scalar, 3>>;
    using vector4_container_type   = ippl::ParticleAttrib<ippl::Vector<scalar, 4>>;

    // Declare instances of the attribute containers
    charge_container_type Q;          // Charge container
    charge_container_type mass;       // Mass container
    velocity_container_type gamma_beta; // Gamma-beta container
    typename ippl::ParticleBase<PLayout>::particle_position_type R_np1; // Position container for the next time step
    typename ippl::ParticleBase<PLayout>::particle_position_type R_np12; // Position container half a timestep in the future, only temporarily correct
    typename ippl::ParticleBase<PLayout>::particle_position_type R_nm1; // Position container for the previous time step
    typename ippl::ParticleBase<PLayout>::particle_position_type R_nm12; // Position container half a timestep in the past, only temporarily correct
    vector_container_type E_gather;   // Electric field container for particle gathering
    vector_container_type B_gather;   // Magnetic field container for particle gathering

};
template <typename _scalar, class PLayout>
struct TracerBunch : public ippl::ParticleBase<PLayout> {
    using scalar = _scalar;

    // Constructor for the Bunch class, taking a PLayout reference
    TracerBunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        // Add attributes to the particle bunch
        //this->addAttribute(Q);          // Charge attribute
        //this->addAttribute(mass);       // Mass attribute
        //this->addAttribute(gamma_beta); // Gamma-beta attribute (product of relativistiv gamma and beta)
        //this->addAttribute(R_np1);      // Position attribute for the next time step
        this->addAttribute(E_gather);   // Electric field attribute for particle gathering
        this->addAttribute(B_gather);   // Magnetic field attribute for particle gathering
        this->addAttribute(outward_normal);   // Magnetic field attribute for particle gathering
    }

    // Destructor for the Bunch class
    ~TracerBunch() {}

    using vector_container_type   = ippl::ParticleAttrib<ippl::Vector<scalar, 3>>;

    vector_container_type E_gather;      // Electric field container for particle gathering
    vector_container_type B_gather;      // Magnetic field container for particle gathering
    vector_container_type outward_normal;// Outward normals for particle gathering
};
template<typename _scalar, class PLayout>
_scalar bunch_energy(const Bunch<_scalar, PLayout>& bantsch);
template<typename bunch_type>
concept bunch_updater = requires(bunch_type t){
    {sizeof(bunch_type) >= 1};
};
namespace ippl {
    enum struct FDTDBoundaryCondition{
        ABC_MUR,
        ABC_FALLAHI,
        PERIODIC
    };

    //TODO: Maybe switch to std::function
    enum struct FDTDParticleUpdateRule{
        LORENTZ, CIRCULAR_ORBIT, DIPOLE_ORBIT, STATIONARY
    };
    enum struct FDTDFieldUpdateRule{
        DONT, DO
    };

    template <typename Tfields, unsigned Dim, class M = UniformCartesian<double, Dim>,
              class C = typename M::DefaultCentering>
    class FDTDSolver {
    public:
        // define a type for scalar field (e.g. charge density field)
        // define a type for vectors
        // define a type for vector field
        typedef Field<Tfields, Dim, M, C> Field_t;
        typedef Vector<Tfields, Dim> Vector_t;
        typedef Field<Vector_t, Dim, M, C> VField_t;
        using memory_space = typename Field_t::memory_space;

        // define type for field layout
        typedef FieldLayout<Dim> FieldLayout_t;

        // type for communication buffers
        //using buffer_type = Communicate::buffer_type<memory_space>;

        // constructor and destructor
        FDTDSolver(Field_t& charge, VField_t& current, VField_t& E, VField_t& B, size_t pcount, FDTDBoundaryCondition bcond = FDTDBoundaryCondition::PERIODIC,
                   FDTDParticleUpdateRule pur = FDTDParticleUpdateRule::LORENTZ, FDTDFieldUpdateRule fur = FDTDFieldUpdateRule::DO, double timestep = 0.05, bool seed_ = false,  VField_t* radiation = nullptr);
        ~FDTDSolver();

        // finite differences time domain solver for potentials (A and phi)
        void solve();

        // evaluates E and B fields using computed potentials
        double field_evaluation();

        template<typename callable>
        void fill_initialcondition(callable c);

        // gaussian pulse
        double gaussian(size_t it, size_t i, size_t j, size_t k)const noexcept;

        // initialization of FDTD solver
        void initialize();

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
        double dt;

        // seed flag
        bool seed;

        // iteration number for gaussian seed
        size_t iteration = 0;

        // fields containing reference to charge and current
        Field_t* rhoN_mp;
        VField_t* JN_mp;

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
        VField_t* radiation_mp;

        FDTDBoundaryCondition bconds_m;
        FDTDParticleUpdateRule particle_update_m;
        FDTDFieldUpdateRule field_update_m;
        using playout_type = ippl::ParticleSpatialLayout<Tfields, 3>;
        using bunch_type = ::Bunch<Tfields, playout_type>;
        using tracer_bunch_type = ::TracerBunch<Tfields, playout_type>;

        size_t pcount_m;
        playout_type pl;
        bunch_type bunch;
        tracer_bunch_type tracer_bunch;
        double total_energy;
        double absorbed__energy;
        // buffer for communication
        detail::FieldBufferData<Tfields> fd_m;
    };
}  // namespace ippl

#include "FDTDSolver.hpp"

#endif
