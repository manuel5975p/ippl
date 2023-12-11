//
// TestFDTDSolver
// This programs tests the FDTD electromagnetic solver with a
// sinusoidal pulse at the center, and absorbing boundaries.
//   Usage:
//     srun ./TestFDTDSolver <nx> <ny> <nz> <timesteps> --info 5
//     nx        = No. cell-centered points in the x-direction
//     ny        = No. cell-centered points in the y-direction
//     nz        = No. cell-centered points in the z-direction
//     timesteps = No. of timesteps
//     (the timestep size is computed using the CFL condition)
//
//     Example:
//       srun ./TestFDTDSolver 25 25 25 150 --info 5
//
// Copyright (c) 2023, Sonali Mayani,
// Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include "Ippl.h"

#include <Kokkos_Macros.hpp>
#include <cstdlib>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"
#include <Kokkos_Random.hpp>
#include "Solver/FDTDSolver.h"
//#include "Solver/BoundaryDispatch.h"

KOKKOS_INLINE_FUNCTION double sine(double n, double dt) {
    return 100 * std::sin(n * dt);
}
template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {
    using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type = typename T::value_type;
    // Output View for the random numbers
    view_type Rview, Rn1view, GBview;
    ippl::NDRegion<value_type, Dim> inside;
    // The GeneratorPool
    GeneratorPool rand_pool;

    // Initialize all members
    generate_random(view_type x_,view_type xn1_, view_type v_, ippl::NDRegion<value_type, Dim> reg, GeneratorPool rand_pool_)
        :Rview(x_)
        ,Rn1view(xn1_)
        ,GBview(v_)
        ,inside(reg)
        ,rand_pool(rand_pool_){}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        //value_type u;
        for (unsigned d = 0; d < Dim; ++d) {
            typename T::value_type w = inside[d].max() - inside[d].min();
            if(d == 0){
                Rview (i)[d] = rand_gen.normal(inside[d].min() + 0.8 * w, 0.03 * w);
                Rn1view (i)[d] = rand_gen.normal(inside[d].min() + 0.8 * w, 0.03 * w);
            }
            else{
                Rview (i)[d] = rand_gen.normal(inside[d].min() + 0.5 * w, 0.03 * w);
                Rn1view (i)[d] = rand_gen.normal(inside[d].min() + 0.5 * w, 0.03 * w);
            }
            GBview(i)[d] = 0; //Irrelefant
        }
        GBview(i)[1] = 1;
        
        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};
void dumpVTK(ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>,
                         ippl::UniformCartesian<double, 3>::DefaultCentering>& E,
             int nx, int ny, int nz, int iteration, double dx, double dy, double dz) {
    using Mesh_t      = ippl::UniformCartesian<double, 3>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<ippl::Vector<double, 3>, 3, Mesh_t, Centering_t> VField_t;
    typename VField_t::view_type::host_mirror_type host_view = E.getHostMirror();

    std::stringstream fname;
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, E.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(7);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << "TestFDTD" << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

    vtkout << "VECTORS E-Field float" << endl;
    for (int x = 0; x < nx + 2; x++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int z = 0; z < nz + 2; z++) {
                vtkout << host_view(x, y, z)[0] << "\t" << host_view(x, y, z)[1] << "\t"
                       << host_view(x, y, z)[2] << endl;
            }
        }
    }
}

void dumpVTK(ippl::Field<double, 3, ippl::UniformCartesian<double, 3>,
                         ippl::UniformCartesian<double, 3>::DefaultCentering>& rho,
             int nx, int ny, int nz, int iteration, double dx, double dy, double dz) {
    using Mesh_t      = ippl::UniformCartesian<double, 3>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<double, 3, Mesh_t, Centering_t> Field_t;
    typename Field_t::view_type::host_mirror_type host_view = rho.getHostMirror();

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, rho.getView());

    Inform vtkout(NULL, fname.str().c_str(), Inform::OVERWRITE);
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << "TestFDTD" << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

    vtkout << "SCALARS Rho float" << endl;
    vtkout << "LOOKUP_TABLE default" << endl;
    for (int z = 0; z < nz + 2; z++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int x = 0; x < nx + 2; x++) {
                vtkout << host_view(x, y, z) << endl;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        const unsigned int Dim = 3;

        // get the gridsize from the user
        ippl::Vector<int, Dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

        // get the total simulation time from the user
        const unsigned int iterations = std::atof(argv[4]);

        using Mesh_t      = ippl::UniformCartesian<double, Dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        typedef ippl::Field<double, Dim, Mesh_t, Centering_t> Field_t;
        typedef ippl::Field<ippl::Vector<double, Dim>, Dim, Mesh_t, Centering_t> VField_t;

        // domain
        ippl::NDIndex<Dim> owned;
        for (unsigned i = 0; i < Dim; i++) {
            owned[i] = ippl::Index(nr[i]);
        }

        // specifies decomposition; here all dimensions are parallel

        //ippl::e_cube_tag decomp[Dim];
        //for (unsigned int d = 0; d < Dim; d++) {
        //    decomp[d] = ippl::IS_PARALLEL;
        //}

        // unit box
        //bool periodic = false;

        //TODO: Put this in everywhere
        using scalar = double;
        scalar dx                        = 1.0 / nr[0];
        scalar dy                        = 1.0 / nr[1];
        scalar dz                        = 1.0 / nr[2];

        ippl::Vector<scalar, Dim> hr     = {dx, dy, dz};
        ippl::Vector<scalar, Dim> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);

        // CFL condition lambda = c*dt/h < 1/sqrt(d) = 0.57 for d = 3
        // we set a more conservative limit by choosing lambda = 0.5
        // we take h = minimum(dx, dy, dz)
        const scalar c = 1.0;  // 299792458.0;
        scalar dt      = std::min({dx, dy, dz}) * 0.5 / c;

        // all parallel layout, standard domain, normal axis order
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // define the R (rho) field
        Field_t rho;
        rho.initialize(mesh, layout);
        rho = 0.0;
        //lambda_dispatch(rho, 1, 
        //KOKKOS_LAMBDA(size_t i, size_t j, size_t k, boundary_occlusion occ){
        //    //std::printf("%ld, %ld, %ld, %s\n", i, j, k, to_string(occ).c_str());
        //},
        //KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
        //    
        //});
        // define the Vector field E (LHS)
        VField_t fieldE, fieldB, radiation;
        fieldE.initialize(mesh, layout);
        fieldB.initialize(mesh, layout);
        radiation.initialize(mesh, layout);
        fieldE = 0.0;
        fieldB = 0.0;
        radiation = 0.0;

        // define current = 0
        VField_t current;
        current.initialize(mesh, layout);
        current = 0.0;

        // turn on the seeding (gaussian pulse) - if set to false, sine pulse is added on rho
        bool seed = false;

        // define an FDTDSolver object
        
        using s_t = ippl::FDTDSolver<double, Dim>; 
        s_t solver(rho, current, fieldE, fieldB, 1000, ippl::FDTDBoundaryCondition::ABC_FALLAHI, ippl::FDTDParticleUpdateRule::CIRCULAR_ORBIT, ippl::FDTDFieldUpdateRule::DO, dt, seed, &radiation);
        auto srview = solver.bunch.R.getView();
        auto srn1view = solver.bunch.R_nm1.getView();
        auto gbrview = solver.bunch.gamma_beta.getView();
        Kokkos::Random_XorShift64_Pool<> rand_pool((size_t)(42312 + 100 * ippl::Comm->rank()));
        typename s_t::playout_type::RegionLayout_t const& rlayout = solver.pl.getRegionLayout();
        typename s_t::playout_type::RegionLayout_t::view_type::host_mirror_type regions_view = rlayout.gethLocalRegions();
        
        Kokkos::parallel_for(
            Kokkos::RangePolicy<typename s_t::playout_type::RegionLayout_t::view_type::execution_space>(0, solver.bunch.getLocalNum()),
            generate_random<ippl::Vector<scalar, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                solver.bunch.R.getView(),
                solver.bunch.R_nm1.getView(),
                solver.bunch.gamma_beta.getView(),
                regions_view(ippl::Comm->rank()),
                rand_pool
            )
            //KOKKOS_LAMBDA(size_t idx){
            //    srview(idx) = ippl::Vector<scalar, Dim>{0.8, 0.5, 0.5};
            //    srn1view(idx) = ippl::Vector<scalar, Dim>{0.8, 0.5, 0.5};
            //    gbrview(idx) = ippl::Vector<scalar, Dim>{0.0, 0.0, 0.0};
            //    
            //}
        );

        solver.phiN_m = 0;
        solver.phiNm1_m = 0;
        solver.fill_initialcondition(KOKKOS_LAMBDA(scalar x, scalar y, scalar z){
            (void)x;(void)y;(void)z;
            return ippl::Vector<scalar, Dim>{0, x * 0.0f, 0};
        });
        if (!seed && false) {
            // add pulse at center of domain
            auto view_rho    = rho.getView();
            auto view_A      = solver.aN_m.getView();
            auto view_Am1      = solver.aNm1_m.getView();
            const int nghost = rho.getNghost();
            auto ldom        = layout.getLocalNDIndex();

            Kokkos::parallel_for(
                "Assign gaussian cylinder", ippl::getRangePolicy(view_A, 2),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    // define the physical points (cell-centered)
                    double x = (ig + 0.5) * hr[0] + origin[0];
                    double y = (jg + 0.5) * hr[1] + origin[1];
                    double z = (kg + 0.5) * hr[2] + origin[2];

                    //if ((x == 0.5) && (y == 0.5) && (z == 0.5))
                    view_A  (i, j, k)[1] = 0.0 * Kokkos::exp(-60.0 * ((x - 0.5) * (x - 0.5)));
                    view_Am1(i, j, k)[1] = 0.0 * Kokkos::exp(-60.0 * ((x - 0.5) * (x - 0.5)));

                    (void)x;(void)y;(void)z;
                    (void)ig;(void)jg;(void)kg; //Suppress warnings lol
            });
        }

        msg << "Timestep number = " << 0 << " , time = " << 0 << endl;
        solver.solve();

        // time-loop
        for (unsigned int it = 1; it < iterations; ++it) {
            msg << "Timestep number = " << it << " , time = " << it * dt << endl;

            /*if (!seed) {
                // add pulse at center of domain
                auto view_rho    = rho.getView();
                const int nghost = rho.getNghost();
                auto ldom        = layout.getLocalNDIndex();

                Kokkos::parallel_for(
                    "Assign sine source at center", ippl::getRangePolicy(view_rho, nghost),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                        const int ig = i + ldom[0].first() - nghost;
                        const int jg = j + ldom[1].first() - nghost;
                        const int kg = k + ldom[2].first() - nghost;

                        // define the physical points (cell-centered)
                        double x = (ig + 0.5) * hr[0] + origin[0];
                        double y = (jg + 0.5) * hr[1] + origin[1];
                        double z = (kg + 0.5) * hr[2] + origin[2];

                        if ((x == 0.5) && (y == 0.5) && (z == 0.5))
                            view_rho(i, j, k) = sine(it, dt);
                });
            }*/

            solver.solve();
            //std::cout << msg.getOutputLevel() << "\n";
            if(msg.getOutputLevel() >= 5){
                dumpVTK(fieldB, nr[0], nr[1], nr[2], it, hr[0], hr[1], hr[2]);
            }
        }
    }
    ippl::finalize();

    return 0;
}
