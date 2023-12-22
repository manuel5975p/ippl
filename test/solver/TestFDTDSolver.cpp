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

#include <Kokkos_Core_fwd.hpp>
#include "Ippl.h"

#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>
#include <cstddef>
#include <cstdlib>
#include <fstream>

#include "Utility/ParameterList.h"

#include "PoissonSolvers/PoissonCG.h"
#include "Solver/FDTDSolver.h"
// #include "Solver/BoundaryDispatch.h"

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
    generate_random(view_type x_, view_type xn1_, view_type v_, ippl::NDRegion<value_type, Dim> reg,
                    GeneratorPool rand_pool_)
        : Rview(x_)
        , Rn1view(xn1_)
        , GBview(v_)
        , inside(reg)
        , rand_pool(rand_pool_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        // value_type u;
        for (unsigned d = 0; d < Dim; ++d) {
            typename T::value_type w = inside[d].max() - inside[d].min();
            if (d == 1) {
                value_type posd = rand_gen.normal(inside[d].min() + 0.6 * w, 0.03 * w);
                Rview(i)[d]     = posd;
                Rn1view(i)[d]   = posd;
            } else {
                value_type posd = rand_gen.normal(inside[d].min() + 0.5 * w, 0.03 * w);
                Rview(i)[d]     = posd;
                Rn1view(i)[d]   = posd;
            }
            GBview(i)[d] = 0;  // Irrelefant
        }
        // GBview(i)[0] = 1;
        // GBview(i)[2] = 0.1;

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

template <unsigned Dim, typename _scalar = double>
struct fdtd_initer {
    using s_t = ippl::FDTDSolver<_scalar, Dim>;
    std::unique_ptr<s_t> m_solver;

    using Mesh_t      = ippl::UniformCartesian<_scalar, Dim>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<_scalar, Dim, Mesh_t, Centering_t> Field_t;
    typedef ippl::Field<ippl::Vector<_scalar, Dim>, Dim, Mesh_t, Centering_t> VField_t;
    using scalar = _scalar;
    Field_t rho;
    VField_t current;
    VField_t fieldE, fieldB, radiation;
    constexpr static unsigned dim = Dim;
    ippl::Vector<int, Dim> nr;
    ippl::Vector<scalar, Dim> hr;
    std::unique_ptr<Mesh_t> mesh;
    std::unique_ptr<ippl::FieldLayout<Dim>> layout;
    typename s_t::Field_t::BConds_t m_ic_scalar_bcs;
    fdtd_initer(int N, size_t PC) {
        

        // domain
        ippl::NDIndex<Dim> owned;
        nr = ippl::Vector<int, Dim>(N);
        for (unsigned i = 0; i < Dim; i++) {
            owned[i] = ippl::Index(nr[i]);
            hr[i]    = 1.0 / nr[i];
        }

        bool seed                        = false;
        ippl::Vector<scalar, Dim> origin = {0.0, 0.0, 0.0};
        mesh = std::make_unique<Mesh_t>(owned, hr, origin);
        const scalar c = 1.0;  // 299792458.0;
        scalar dt      = (*std::min(hr.begin(), hr.end())) * 0.5 / c;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        layout = std::make_unique<ippl::FieldLayout<Dim>>(MPI_COMM_WORLD, owned, isParallel);
        

        rho.initialize(*mesh, *layout);
        rho = 0.0;
        fieldE.initialize(*mesh, *layout);
        fieldB.initialize(*mesh, *layout);
        radiation.initialize(*mesh, *layout);
        current.initialize(*mesh, *layout);
        current = 0.0;

        m_solver = std::make_unique<s_t>(rho, current, fieldE, fieldB, PC,
                                         ippl::FDTDBoundaryCondition::ABC_FALLAHI,
                                         ippl::FDTDParticleUpdateRule::XLINE,
                                         ippl::FDTDFieldUpdateRule::DO, dt, seed, &radiation);
        setupBunch(PC);
        initialConditionPhi();
    }
    void setupBunch(size_t pc){
        m_solver->bunch.setParticleBC(ippl::BC::PERIODIC);
        auto srview=    m_solver->bunch.R.getView();
        auto srn1view = m_solver->bunch.R_nm1.getView();
        auto gbrview  = m_solver->bunch.gamma_beta.getView();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<
                typename s_t::playout_type::RegionLayout_t::view_type::execution_space>(
                0, m_solver->bunch.getLocalNum()),
            // generate_random<ippl::Vector<scalar, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
            //     solver.bunch.R.getView(),
            //     solver.bunch.R_nm1.getView(),
            //     solver.bunch.gamma_beta.getView(),
            //     regions_view(ippl::Comm->rank()),
            //     rand_pool
            //)
            KOKKOS_LAMBDA(size_t idx) {
                srview(idx) =
                    ippl::Vector<scalar, Dim>{(idx) / scalar(pc), 0.50234, 0.50341523765};
                srn1view(idx) =
                    ippl::Vector<scalar, Dim>{(idx) / scalar(pc), 0.50234, 0.50341523765};
                gbrview(idx) = ippl::Vector<scalar, Dim>{0.0, 0.0, 0.0};
            });
        Kokkos::fence();
    }
    void initialConditionPhi() {
        
        auto& ic_scalar_bcs = m_ic_scalar_bcs;
        const auto hr = this->hr;
        m_solver->bunch.Q.scatterVolumetricallyCorrect(rho, m_solver->bunch.R);
        auto bcsetter_single = [&ic_scalar_bcs, hr]<size_t Idx>(const std::index_sequence<Idx>&) {
            ic_scalar_bcs[Idx] = std::make_shared<ippl::ZeroFace<Field_t>>(Idx);
            return 0;
        };
        auto bcsetter = [bcsetter_single]<size_t... Idx>(const std::index_sequence<Idx...>&) {
            int x = (bcsetter_single(std::index_sequence<Idx>{}) ^ ...);
            (void)x;
        };
        bcsetter(std::make_index_sequence<Dim * 2>{});
        ippl::ParameterList list;
        list.add("use_heffte_defaults", true);
        list.add("output_type", ippl::PoissonCG<VField_t, Field_t>::SOL);

        Field_t urho   = rho.deepCopy();
        Field_t phi_ic = m_solver->phiN_m.deepCopy();
        rho.setFieldBC(ic_scalar_bcs);
        phi_ic.setFieldBC(ic_scalar_bcs);

        ippl::PoissonCG<Field_t, Field_t> initial_phi_solver(phi_ic, rho);
        initial_phi_solver.mergeParameters(list);
        try {
            initial_phi_solver.solve();
        } catch (const IpplException& e) {
            LOG("Exception yoten: " << e.what());
        }
        Kokkos::deep_copy(m_solver->phiN_m.getView(), phi_ic.getView());
        Kokkos::deep_copy(m_solver->phiNm1_m.getView(), phi_ic.getView());
    }
    void doNSteps(size_t nsteps){
        for (unsigned int it = 0; it < nsteps; ++it){
            m_solver->solve();
        }
    }
};
int main(int argc, char* argv[]){
    ippl::initialize(argc, argv);
    {
        constexpr unsigned Dim = 3;
        fdtd_initer<Dim> fdtd(3, 100);
        fdtd.doNSteps(100);
    }
    ippl::finalize();

    return 0;
}
int main2(int argc, char* argv[]) {
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

        // ippl::e_cube_tag decomp[Dim];
        // for (unsigned int d = 0; d < Dim; d++) {
        //     decomp[d] = ippl::IS_PARALLEL;
        // }

        // unit box
        // bool periodic = false;

        // TODO: Put this in everywhere
        using scalar = double;
        scalar dx    = 1.0 / nr[0];
        scalar dy    = 1.0 / nr[1];
        scalar dz    = 1.0 / nr[2];

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
        // lambda_dispatch(rho, 1,
        // KOKKOS_LAMBDA(size_t i, size_t j, size_t k, boundary_occlusion occ){
        //     //std::printf("%ld, %ld, %ld, %s\n", i, j, k, to_string(occ).c_str());
        // },
        // KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
        //
        // });
        //  define the Vector field E (LHS)
        VField_t fieldE, fieldB, radiation;
        fieldE.initialize(mesh, layout);
        fieldB.initialize(mesh, layout);
        radiation.initialize(mesh, layout);
        fieldE    = 0.0;
        fieldB    = 0.0;
        radiation = 0.0;

        // define current = 0
        VField_t current;
        current.initialize(mesh, layout);
        current = 0.0;
        // turn on the seeding (gaussian pulse) - if set to false, sine pulse is added on rho
        bool seed = false;

        // define an FDTDSolver object

        using s_t     = ippl::FDTDSolver<double, Dim>;
        size_t pcount = 10000;
        s_t solver(rho, current, fieldE, fieldB, pcount, ippl::FDTDBoundaryCondition::ABC_FALLAHI,
                   ippl::FDTDParticleUpdateRule::XLINE, ippl::FDTDFieldUpdateRule::DO, dt, seed,
                   &radiation);
        solver.bunch.setParticleBC(ippl::BC::PERIODIC);
        std::ofstream rad_output("radiation.txt");
        std::ofstream p0pos_output("partpos.txt");
        solver.output_stream[ippl::trackableOutput::boundaryRadiation] = &rad_output;
        solver.output_stream[ippl::trackableOutput::p0pos]             = &p0pos_output;
        auto srview= solver.bunch.R.getView();
        auto srn1view = solver.bunch.R_nm1.getView();
        auto gbrview  = solver.bunch.gamma_beta.getView();
        Kokkos::Random_XorShift64_Pool<> rand_pool((size_t)(42312 + 100 * ippl::Comm->rank()));
        typename s_t::playout_type::RegionLayout_t const& rlayout = solver.pl.getRegionLayout();
        typename s_t::playout_type::RegionLayout_t::view_type::host_mirror_type regions_view =
            rlayout.gethLocalRegions();

        Kokkos::parallel_for(
            Kokkos::RangePolicy<
                typename s_t::playout_type::RegionLayout_t::view_type::execution_space>(
                0, solver.bunch.getLocalNum()),
            // generate_random<ippl::Vector<scalar, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
            //     solver.bunch.R.getView(),
            //     solver.bunch.R_nm1.getView(),
            //     solver.bunch.gamma_beta.getView(),
            //     regions_view(ippl::Comm->rank()),
            //     rand_pool
            //)
            KOKKOS_LAMBDA(size_t idx) {
                srview(idx) =
                    ippl::Vector<scalar, Dim>{(idx) / scalar(pcount), 0.50234, 0.50341523765};
                srn1view(idx) =
                    ippl::Vector<scalar, Dim>{(idx) / scalar(pcount), 0.50234, 0.50341523765};
                gbrview(idx) = ippl::Vector<scalar, Dim>{0.0, 0.0, 0.0};
            });
        Kokkos::fence();
        rho = 0.0;
        solver.bunch.Q.scatterVolumetricallyCorrect(rho, solver.bunch.R);
        s_t::Field_t::BConds_t ic_scalar_bcs;
        {
            auto bcsetter_single = [&ic_scalar_bcs,
                                    hr]<size_t Idx>(const std::index_sequence<Idx>&) {
                ic_scalar_bcs[Idx] = std::make_shared<ippl::ZeroFace<Field_t>>(Idx);
                return 0;
            };
            auto bcsetter = [bcsetter_single]<size_t... Idx>(const std::index_sequence<Idx...>&) {
                int x = (bcsetter_single(std::index_sequence<Idx>{}) ^ ...);
                (void)x;
            };
            bcsetter(std::make_index_sequence<Dim * 2>{});
            ippl::ParameterList list;
            list.add("use_heffte_defaults", true);
            list.add("output_type", ippl::PoissonCG<VField_t, Field_t>::SOL);

            Field_t urho   = rho.deepCopy();
            Field_t phi_ic = solver.phiN_m.deepCopy();
            rho.setFieldBC(ic_scalar_bcs);
            phi_ic.setFieldBC(ic_scalar_bcs);

            ippl::PoissonCG<Field_t, Field_t> initial_phi_solver(phi_ic, rho);
            initial_phi_solver.mergeParameters(list);
            try {
                initial_phi_solver.solve();
            } catch (const IpplException& e) {
                LOG("Exception yoten: " << e.what());
            }
            Kokkos::deep_copy(solver.phiN_m.getView(), phi_ic.getView());
            Kokkos::deep_copy(solver.phiNm1_m.getView(), phi_ic.getView());
        }
        solver.setBoundaryConditions(ippl::Vector<ippl::FDTDBoundaryCondition, Dim>{
            ippl::FDTDBoundaryCondition::PERIODIC, ippl::FDTDBoundaryCondition::ABC_MUR,
            ippl::FDTDBoundaryCondition::ABC_MUR});
        // solver.phiN_m = 0;
        // solver.phiNm1_m = 0;
        // solver.fill_initialcondition(KOKKOS_LAMBDA(scalar x, scalar y, scalar z){
        //     (void)x;(void)y;(void)z;
        //     return ippl::Vector<scalar, Dim>{0, x * 7.0, 0};
        // });
        if (!seed && false) {
            // add pulse at center of domain
            auto view_rho    = rho.getView();
            auto view_A      = solver.aN_m.getView();
            auto view_Am1    = solver.aNm1_m.getView();
            const int nghost = rho.getNghost();
            auto ldom        = layout.getLocalNDIndex();

            Kokkos::parallel_for(
                "Assign gaussian cylinder", ippl::getRangePolicy(view_A, 0),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    // define the physical points (cell-centered)
                    double x = (ig + 0.5) * hr[0] + origin[0];
                    double y = (jg + 0.5) * hr[1] + origin[1];
                    double z = (kg + 0.5) * hr[2] + origin[2];

                    // if ((x == 0.5) && (y == 0.5) && (z == 0.5))
                    view_A(i, j, k)[1]   = Kokkos::exp(-60.0 * ((x - 0.5) * (x - 0.5)));
                    view_Am1(i, j, k)[1] = Kokkos::exp(-60.0 * ((x - 0.5) * (x - 0.5)));

                    (void)x;
                    (void)y;
                    (void)z;
                    (void)ig;
                    (void)jg;
                    (void)kg;  // Suppress warnings lol
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
            // std::cout << msg.getOutputLevel() << "\n";
            if (msg.getOutputLevel() >= 5) {
                dumpVTK(radiation, nr[0], nr[1], nr[2], it, hr[0], hr[1], hr[2]);
            }
        }
    }
    ippl::finalize();

    return 0;
}
