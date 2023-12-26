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

#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include "Ippl.h"

#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>
#include <cstddef>
#include <cstdlib>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <fstream>

#include "Utility/ParameterList.h"

#include "PoissonSolvers/PoissonCG.h"
#include "PoissonSolvers/FFTOpenPoissonSolver.h"
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
    using Vector_t = typename s_t::Vector_t;
    private:
    size_t required_steps;
    size_t particle_count;
    public:
    fdtd_initer(int N, size_t PC, scalar end_time) {
        
        particle_count = PC;
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
        required_steps = std::ceil(end_time / dt);
        dt = end_time / required_steps;
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
                    ippl::Vector<scalar, Dim>{(idx) / scalar(pc), 0.5, 0.5};
                srn1view(idx) =
                    ippl::Vector<scalar, Dim>{(idx) / scalar(pc), 0.5, 0.5};
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
        ippl::Vector<ippl::FDTDBoundaryCondition, Dim> fdtdbcs(ippl::FDTDBoundaryCondition::ABC_MUR);
        fdtdbcs[0] = ippl::FDTDBoundaryCondition::PERIODIC;
        m_solver->setBoundaryConditions(fdtdbcs);
    }
    void doRequiredSteps(){
        for (unsigned int it = 0; it < required_steps; ++it){
            m_solver->solve();
            //std::cout << ippl::Info->getOutputLevel() << "\n";
            if(ippl::Info->getOutputLevel() == 5 && (it % (required_steps / 20)) == 0){
                dumpVTK(fieldB, nr[0], nr[1], nr[2], it, hr[0], hr[1], hr[2]);
            }
        }
        evaluation();
    }
    void evaluation(){
        typename s_t::tracer_bunch_type eval(m_solver->pl);
        const auto pc = this->particle_count;
        eval.create(pc);
        auto srview = eval.R.getView();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<
                typename s_t::playout_type::RegionLayout_t::view_type::execution_space>(
                0, m_solver->bunch.getLocalNum()),
            KOKKOS_LAMBDA(size_t idx) {
                srview(idx) =
                    ippl::Vector<scalar, Dim>{0.500004513241, (idx) / scalar(pc) ,0.500001204513241};
            }
        );
        eval.E_gather.gather(fieldE, eval.R);
        eval.B_gather.gather(fieldB, eval.R);
        typename Kokkos::View<Vector_t*>::host_mirror_type Eview_hmirror = Kokkos::create_mirror_view(eval.E_gather.getView());
        typename Kokkos::View<Vector_t*>::host_mirror_type Bview_hmirror = Kokkos::create_mirror_view(eval.B_gather.getView());
        typename Kokkos::View<Vector_t*>::host_mirror_type Rview_hmirror = Kokkos::create_mirror_view(eval.R.getView());
        Kokkos::deep_copy(Eview_hmirror, eval.E_gather.getView());
        Kokkos::deep_copy(Bview_hmirror, eval.B_gather.getView());
        Kokkos::deep_copy(Rview_hmirror, eval.R.getView());
        std::ofstream ostr("dB.txt");
        for(size_t i = 0;i < particle_count;i++){
            auto v = Rview_hmirror(i);
            auto B = Bview_hmirror(i);
            v -= 0.5;
            v[0] = 0.0;
            double dist = Kokkos::sqrt(dot_prod(v, v));
            double bmag = Kokkos::sqrt(dot_prod(B, B));

            ostr << dist << " " << bmag << "\n";
        }

    }
};

int main(int argc, char* argv[]){
    ippl::initialize(argc, argv);
    {
        constexpr unsigned Dim = 3;
        int N = 30;
        if(argc > 1){
            N = std::atoi(argv[1]);
        }
        fdtd_initer<Dim> fdtd(N, N * 10, 15.0);
        fdtd.doRequiredSteps();
    }
    ippl::finalize();

    return 0;
}