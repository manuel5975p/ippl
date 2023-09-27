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

#include <cstdlib>
#include <fstream>
#include <functional>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "Solver/FDTDSolver.h"

KOKKOS_INLINE_FUNCTION double sine(double n, double dt) {
    return 100 * std::sin(n * dt);
}
KOKKOS_INLINE_FUNCTION double gauss(double x, double mean, double stddev) {
    //return std::sin(x * M_PI * 2.0 * 1.0);
    //return 100.0 * std::exp(-(x - mean) * (x - mean) / (stddev * stddev)) * x;
    return (1.0 + x - mean) * 100.0 * std::exp(-(x - mean) * (x - mean) / (stddev * stddev)) * x;
    //return 100.0 * (std::max(0.0, 1.0 - std::abs(x - mean) / stddev));
}

void dumpVTK(ippl::Field<ippl::Vector<double, 3>, 3, ippl::UniformCartesian<double, 3>,
                         ippl::UniformCartesian<double, 3>::DefaultCentering>& E,
             int nx, int ny, int nz, int iteration, double dx, double dy, double dz) {
    using Mesh_t      = ippl::UniformCartesian<double, 3>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<ippl::Vector<double, 3>, 3, Mesh_t, Centering_t> VField_t;
    typename VField_t::view_type::host_mirror_type host_view = E.getHostMirror();

    std::stringstream fname;
    constexpr char endl = '\n';
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, E.getView());

    std::ofstream vtkout(fname.str());
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

    vtkout << "VECTORS E-Field float" << endl;
    for (int z = 0; z < nz + 2; z++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int x = 0; x < nx + 2; x++) {
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

    std::ofstream vtkout(fname.str());
    constexpr char endl = '\n';
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
        const double time_simulated = std::atof(argv[4]);
        if(time_simulated <= 0){
            std::cerr << "Time must be > 0\n";
            goto exit;
        }
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
        ippl::e_dim_tag decomp[Dim];
        for (unsigned int d = 0; d < Dim; d++) {
            decomp[d] = ippl::PARALLEL;
        }

        // unit box
        double dx                        = 1.0 / nr[0];
        double dy                        = 1.0 / nr[1];
        double dz                        = 1.0 / nr[2];
        ippl::Vector<double, Dim> hr     = {dx, dy, dz};
        ippl::Vector<double, Dim> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);
        // CFL condition lambda = c*dt/h < 1/sqrt(d) = 0.57 for d = 3
        // we set a more conservative limit by choosing lambda = 0.5
        // we take h = minimum(dx, dy, dz)
        const double c = 1.0;  // 299792458.0;
        double dt      = std::min({dx, dy, dz}) * 0.5 / c;
        unsigned int iterations = std::ceil(time_simulated / dt);
        dt = time_simulated / (double)iterations;
        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<Dim> layout(owned, decomp);

        // define the R (rho) field
        Field_t rho;
        rho.initialize(mesh, layout);
        //lambda_dispatch(rho, 1, 
        //KOKKOS_LAMBDA(size_t i, size_t j, size_t k, boundary_occlusion occ){
        //    //std::printf("%ld, %ld, %ld, %s\n", i, j, k, to_string(occ).c_str());
        //},
        //KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
        //    
        //});
        // define the Vector field E (LHS)
        VField_t fieldE, fieldB;
        fieldE.initialize(mesh, layout);
        fieldB.initialize(mesh, layout);
        fieldE = 0.0;
        fieldB = 0.0;

        // define current = 0
        VField_t current;
        current.initialize(mesh, layout);
        
        current = 0.0;

        // turn on the seeding (gaussian pulse) - if set to false, sine pulse is added on rho
        bool seed = false;

        // define an FDTDSolver object
        ippl::FDTDSolver<double, Dim> solver(&rho, &current, &fieldE, &fieldB, dt, seed);
        solver.bconds[0] = ippl::PERIODIC_FACE;
        solver.bconds[1] = ippl::PERIODIC_FACE;
        solver.bconds[2] = ippl::PERIODIC_FACE;

        /*
        std::cout << nr[0] << " " << current.getView().extent(0) << "\n";
        std::cout << nr[1] << " " << current.getView().extent(1) << "\n";
        std::cout << nr[2] << " " << current.getView().extent(2) << "\n";
        std::cout << nr[0] << " " << solver.aN_m.getView().extent(0) << "\n";
        std::cout << nr[1] << " " << solver.aN_m.getView().extent(1) << "\n";
        std::cout << nr[2] << " " << solver.aN_m.getView().extent(2) << "\n";
        return 0;
        */

        if (!seed) {
            // add pulse at center of domain
            auto view_rho    = rho.getView();
            const int nghost = rho.getNghost();
            auto ldom        = layout.getLocalNDIndex();
            auto view_a      = solver.aN_m.getView();
            auto view_an1    = solver.aNm1_m.getView();
            solver.fill_initialcondition(
                KOKKOS_LAMBDA(double x, double y, double z) {
                    ippl::Vector<double, 3> ret(0.0);
                    ret[2] = gauss(/*std::hypot(x - 0.5, y - 0.5, z - 0.5)*/ y - 0.5, 0.0, 0.1);
                    (void)x;
                    (void)y;
                    (void)z;
                    return ret;
            });
        }
        msg << "Timestep number = " << 0 << " , time = " << 0 << endl;
        dumpVTK(solver.aN_m, nr[0], nr[1], nr[2], 0, hr[0], hr[1], hr[2]);
        solver.solve();
        dumpVTK(solver.aN_m, nr[0], nr[1], nr[2], 1, hr[0], hr[1], hr[2]);
        // time-loop
        for (unsigned int it = 1; it < iterations; ++it) {
            msg << "Timestep number = " << it << " , time = " << it * dt << endl;

            if (!seed) {
                // add pulse at center of domain
                auto view_rho    = rho.getView();
                auto view_a    = solver.aN_m.getView();
                const int nghost = rho.getNghost();
                auto ldom        = layout.getLocalNDIndex();
            }

            solver.solve();
            
            //dumpVTK(solver.aN_m, nr[0], nr[1], nr[2], it + 1, hr[0], hr[1], hr[2]);
        }
        if (!seed) {
            // add pulse at center of domain
            
            auto view_rho    = rho.getView();
            const int nghost = rho.getNghost();
            auto ldom        = layout.getLocalNDIndex();
            auto view_a      = solver.aN_m.getView();
            auto view_b      = fieldB.getView();
            auto view_e      = fieldE.getView();
            
            auto view_an1    = solver.aNm1_m.getView();
            //Kokkos::View<double***> energy_density("Energies", view_a.extent(0), view_a.extent(1),
            //                                   view_a.extent(2));
        
            double error_accumulation = 0.0;
            const double volume = (1.0 / (nr[0] - 6)) * (1.0 / (nr[1] - 6)) * (1.0 / (nr[2] - 6));
            /*Kokkos::parallel_reduce(
                "Assign sinusoidal source at center", ippl::getRangePolicy(view_a, 3),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& ref) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    // define the physical points (cell-centered)
                    double x = (ig + 0.5) * hr[0] + origin[0];
                    double y = (jg + 0.5) * hr[1] + origin[1];
                    double z = (kg + 0.5) * hr[2] + origin[2];
                    //std::cout << std::to_string(y) + " " + std::to_string(gauss(y, 0.5, 0.25)) + "\n";
                    if(i > 0 && j > 0 && k > 0 && i < view_a.extent(0) - 1 && j < view_a.extent(1) - 1 && k < view_a.extent(2) - 1){
                        //ref += (dot_prod(view_b(i, j, k), view_b(i, j, k)) + dot_prod(view_e(i, j, k), view_e(i, j, k))) * volume;
                        ref += std::abs(view_a(i, j, k)[2]) * volume;
                        //ref += std::abs(view_a(i, j, k)[2] - gauss(y, 0.5, 0.1)) * volume;
                        //view_an1(i, j, k)[2] - gauss(y, 0.5, 0.1);
                    }
                    (void)x;
                    (void)y;
                    (void)z;
                    //if ((x == 0.5) && (y == 0.5) && (z == 0.5)){}
                    //if(jg == nr[1] - 2){
                        //view_rho(i, j, k) = sine(0, dt);
                    //}
            }, error_accumulation);*/
            error_accumulation = solver.volumetric_integral(KOKKOS_LAMBDA(const int i, const int j, const int k, double x, double y, double z){
                return std::abs(view_a(i, j, k)[2] - gauss(/*std::hypot(x - 0.5, y - 0.5, z - 0.5)*/ y - 0.5, 0.0, 0.1));
            });
            std::cout << "TOTAL ERROR: " << error_accumulation << std::endl;
        }
    }
    exit:
    ippl::finalize();

    return 0;
}
