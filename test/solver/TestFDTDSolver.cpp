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
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <fstream>
#include "Field/BcTypes.h"
#include "Types/Vector.h"

#include "Particle/ParticleAttrib.h"
#include "Solver/FDTDSolver.h"


template<typename T, typename R>
KOKKOS_INLINE_FUNCTION auto sine(T n, R dt) {
    using Kokkos::sin;
    return 100.0 * sin(n * dt);
}
KOKKOS_INLINE_FUNCTION auto gauss(double x, double mean, double stddev) {
    (void)x;
    (void)mean;
    (void)stddev;
    //return std::sin(x * M_PI * 2.0 * 1.0);
    return std::exp(-(x - mean) * (x - mean) / (stddev * stddev));
    //return (1.0 + x - mean) * 100.0 * std::exp(-(x - mean) * (x - mean) / (stddev * stddev)) * x;
    //return 100.0 * (std::max(0.0, 1.0 - std::abs(x - mean) / stddev));
}
template<typename T, unsigned Dim>
std::string povstring(const ippl::Vector<T, Dim>& v){
    std::stringstream sstr;
    sstr << '<';
    for(size_t i = 0;i < Dim;i++){
        sstr << v[i];
        if(Dim && (i < Dim - 1)){
            sstr << ',';
        }
    }
    sstr << '>';
    return sstr.str();
}
template<typename scalar>
void dumpPOV(const typename ippl::FDTDSolver<scalar, 3>::bunch_type& pbunch,  ippl::Field<ippl::Vector<scalar, 3>, 3, ippl::UniformCartesian<scalar, 3>,
                         typename ippl::UniformCartesian<scalar, 3>::DefaultCentering>& E,
             int nx, int ny, int nz, int iteration, scalar dx, scalar dy, scalar dz) {
         using Mesh_t      = ippl::UniformCartesian<scalar, 3>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<ippl::Vector<scalar, 3>, 3, Mesh_t, Centering_t> VField_t;
    typename VField_t::view_type::host_mirror_type host_view = E.getHostMirror();

    std::stringstream fname;
    constexpr char endl = '\n';
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".pov";

    Kokkos::deep_copy(host_view, E.getView());

    std::ofstream vtkout(fname.str());
    vtkout << R"(
#include "colors.inc"
#include "glass.inc"
#include "metals.inc"
#version 3.7;
global_settings {
  assumed_gamma 1.0
  max_trace_level 15
  photons {
    count 10000
  }
  radiosity{
    pretrace_start 0.04
    pretrace_end 0.01
    count 100
    recursion_limit 5
    nearest_count 10
    error_bound 0.3
  }
}
camera {
  location    <0,-10,-10>
  right       x*image_width/image_height
  look_at     <0, 0, 0>
  angle       50
}
light_source {
  <5, 10, 0>
  color rgb <1,1,1>
  photons {
    reflection on
    refraction on
  }
}
    )";
    for (int x = 0; x < nx + 2; x++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int z = 0; z < nz + 2; z++) {
                vtkout << "sphere{\n";
                vtkout << povstring(ippl::Vector<double, 3>{(double)x, (double)y, (double)z}) << ", 0.2\n";
                vtkout << "texture {pigment{rgb <0,1,0>}}\n";
                vtkout << "}\n";
                //vtkout << host_view(x, y, z)[0] << "\t" << host_view(x, y, z)[1] << "\t"
                //       << host_view(x, y, z)[2] << endl;
            }
        }
    }
}
template<typename scalar>
Kokkos::View<ippl::Vector<scalar, 3>***> collect_field_on_zero(ippl::Field<ippl::Vector<scalar, 3>, 3, ippl::UniformCartesian<scalar, 3>,
                         typename ippl::UniformCartesian<scalar, 3>::DefaultCentering>& E, int nx, int ny, int nz){
    
    ippl::NDIndex<3U> lindex = E.getLayout().getLocalNDIndex();
    std::vector<std::pair<ippl::NDIndex<3>, std::vector<scalar>>> ret;
    
    //unsigned volume = 1;
    //for(int i = 0;i < 3;i++){
    //    volume *= (lindex[i].last() - lindex[i].first());
    //}
    std::vector<ippl::Vector<scalar, 3>> local_field_dump;

    //int lindex_xextent = lindex[0].last() - lindex[0].first() + 1;
    //if(lindex_xextent != E.getFieldRangePolicy(1).m_upper[0] - E.getFieldRangePolicy(1).m_lower[0]){
    //    std::cout << lindex_xextent << " " << E.getFieldRangePolicy(0).m_upper[0] - E.getFieldRangePolicy(0).m_lower[0] << std::endl;
    //    throw 5;
    //}
    //if(E.getFieldRangePolicy(1).m_upper[0] - E.getFieldRangePolicy(1).m_lower[0] != nx){std::cout << "NX = " << nx << ", DIFF = " << E.getFieldRangePolicy(1).m_upper[0] - E.getFieldRangePolicy(1).m_lower[0] << std::endl;throw 5;}
    for(int64_t i = E.getFieldRangePolicy(0).m_lower[0];i < E.getFieldRangePolicy(0).m_upper[0];i++){
        for(int64_t j = E.getFieldRangePolicy(0).m_lower[1];j < E.getFieldRangePolicy(0).m_upper[1];j++){
            for(int64_t k = E.getFieldRangePolicy(0).m_lower[2];k < E.getFieldRangePolicy(0).m_upper[2];k++){
                local_field_dump.push_back(E.getView()(i,j,k));
            }
        }
    }
    size_t local_field_size = local_field_dump.size();
    std::vector<ippl::NDIndex<3U>> field_regions(ippl::Comm->size());
    std::vector<size_t> field_sizes(ippl::Comm->size(), 0);

    // Gather the counts of data from all processes to the root
    MPI_Gather(&local_field_size, 1, MPI_UNSIGNED_LONG, field_sizes.data(), 1, MPI_UNSIGNED_LONG, 0, ippl::Comm->getCommunicator());
    MPI_Gather(&lindex, sizeof(ippl::NDIndex<3U>), MPI_UNSIGNED_CHAR, field_regions.data(), sizeof(ippl::NDIndex<3U>), MPI_UNSIGNED_CHAR, 0, ippl::Comm->getCommunicator());
    size_t total_field_size_on_rank_0 = 0;
    std::vector<ippl::Vector<scalar, 3>> complete_field_collection;
    int* rcnts = new int[ippl::Comm->size()];
    int* displs = new int[ippl::Comm->size()];
    size_t displ = 0;
    if(ippl::Comm->rank() == 0){
        for(size_t i = 0;i < field_sizes.size();i++){
            total_field_size_on_rank_0 += field_sizes[i]*3;
            displs[i] = displ;
            rcnts[i] = field_sizes[i] * 3;
            displ += field_sizes[i] * 3;
        }
            
        complete_field_collection.resize(total_field_size_on_rank_0 * 3);
    }
    
    //std::cerr << ippl::Comm->size() << "\n";
    MPI_Gatherv(local_field_dump.data(), local_field_dump.size() * 3, std::is_same_v<std::remove_all_extents_t<scalar>, double> ? MPI_DOUBLE : MPI_FLOAT, complete_field_collection.data(), rcnts, displs, std::is_same_v<std::remove_all_extents_t<scalar>, double> ? MPI_DOUBLE : MPI_FLOAT, 0, ippl::Comm->getCommunicator());
    Kokkos::View<ippl::Vector<scalar, 3>***> rank0collect;
    if(ippl::Comm->rank() == 0){
        rank0collect = Kokkos::View<ippl::Vector<scalar, 3>***>("", nx, ny, nz);
        for(int i = 0;i < ippl::Comm->size();i++){
            std::vector<ippl::Vector<scalar, 3>> rebuild(complete_field_collection.begin() + displs[i] / 3, complete_field_collection.begin() + displs[i] / 3 + rcnts[i] / 3);
            typename Kokkos::View<ippl::Vector<scalar, 3>*>::host_mirror_type hmirror("", rebuild.size());
            Kokkos::View<ippl::Vector<scalar, 3>*> rebuild_view("", rebuild.size());
            for(size_t ih = 0;ih < rebuild.size();ih++){
                hmirror(ih) = rebuild[ih];
            }
            Kokkos::deep_copy(rebuild_view, hmirror);
            Kokkos::View<ippl::Vector<scalar, 3>***> viu("",
                1 + field_regions[i][0].last() - field_regions[i][0].first(),
                1 + field_regions[i][1].last() - field_regions[i][1].first(),
                1 + field_regions[i][2].last() - field_regions[i][2].first()
            );
            Kokkos::Array<long int, 3> from{
                field_regions[i][0].first(),
                field_regions[i][1].first(),
                field_regions[i][2].first()
            };
            Kokkos::Array<long int, 3> to{
                field_regions[i][0].last() + 1,
                field_regions[i][1].last() + 1,
                field_regions[i][2].last() + 1
            };
            //for(size_t _i = 0;_i < viu.extent(0);_i++){
            //    for(size_t j = 0;j < viu.extent(1);j++){
            //        for(size_t k = 0;k < viu.extent(2);k++){
            //            viu(_i, j, k) = rebuild[_i * viu.extent(1) * viu.extent(2) + j * viu.extent(2) + k];
            //        }
            //    }
            //}
            using exec_space = typename decltype(viu)::execution_space;

            Kokkos::parallel_for(ippl::createRangePolicy<3, exec_space>(from, to), KOKKOS_LAMBDA(int i, int j, int k){
                int i_r = i - from[0];
                int j_r = j - from[1];
                int k_r = k - from[2];
                rank0collect(i,j,k) = rebuild_view(i_r * viu.extent(1) * viu.extent(2) + j_r * viu.extent(2) + k_r);
            });
            //Kokkos::View<ippl::Vector<double, 3>***> dst = Kokkos::subview(rank0collect, 
            //    Kokkos::make_pair(field_regions[i][0].first(), field_regions[i][0].last()),
            //    Kokkos::make_pair(field_regions[i][1].first(), field_regions[i][1].last()),
            //    Kokkos::make_pair(field_regions[i][2].first(), field_regions[i][2].last())
            //);
            //Kokkos::deep_copy(dst, viu);
        }

    }
    //std::cout << "Gathered\n";
    delete[] rcnts;
    delete[] displs;
    //std::stringstream sstr;
    //sstr << ippl::Comm->rank() << ": Volume = " << volume << " = " << (lindex[0].last() - lindex[0].first()) << " * " << (lindex[1].last() - lindex[1].first()) << " * " << (lindex[2].last() - lindex[2].first()) << " * " << "\n";
    //std::cout << sstr.str();
    
    return rank0collect;
}

template<typename view_value_type>
Kokkos::View<view_value_type*> collect_linear_view_on_zero(Kokkos::View<view_value_type*> src_view, size_t locsize){
    //return Kokkos::View<view_value_type*>("", 0);
    size_t local_field_size = locsize;
    std::vector<size_t> field_sizes(ippl::Comm->size(), 0);

    //std::cout << "pg" << std::endl;// Gather the counts of data from all processes to the root
    MPI_Gather(&local_field_size, 1, MPI_UNSIGNED_LONG, field_sizes.data(), 1, MPI_UNSIGNED_LONG, 0, ippl::Comm->getCommunicator());
    if(!ippl::Comm->rank()){
        size_t ts = 0;
        for(auto fs : field_sizes)ts += fs;
        std::cout << "Total: " << ts << "\n";
    }
    //std::cout << "ag" << std::endl;
    typename Kokkos::View<view_value_type*>::host_mirror_type hmirror = Kokkos::create_mirror_view(src_view);
    std::vector<view_value_type> local_view_dump(local_field_size);
    for(size_t i = 0;i < local_field_size;i++){
        local_view_dump[i] = hmirror[i];
    }
    std::vector<int> rcnts(ippl::Comm->size());
    std::vector<int> displs(ippl::Comm->size());
    std::vector<view_value_type> complete_field_collection;
    size_t total_field_size_on_rank_0 = 0;
    if(ippl::Comm->rank() == 0){
        int displ = 0;
        for(size_t i = 0;i < field_sizes.size();i++){
            total_field_size_on_rank_0 += field_sizes[i];
            displs[i] = displ;
            rcnts[i] = field_sizes[i] * sizeof(view_value_type);
            displ += field_sizes[i] * sizeof(view_value_type);
        }
            
        complete_field_collection.resize(total_field_size_on_rank_0);
    }
    MPI_Gatherv(local_view_dump.data(), local_view_dump.size() * sizeof(view_value_type), MPI_CHAR, complete_field_collection.data(), rcnts.data(), displs.data(), MPI_CHAR, 0, ippl::Comm->getCommunicator());
    Kokkos::View<view_value_type*> ret("", complete_field_collection.size());
    typename Kokkos::View<view_value_type*>::host_mirror_type ret_hmirror("", complete_field_collection.size());
    if(ippl::Comm->rank() == 0){
        for(size_t i = 0;i < complete_field_collection.size();i++){
            ret_hmirror(i) = complete_field_collection[i];
        }
        Kokkos::deep_copy(ret, ret_hmirror);
    }
    return ret;

}
template<typename scalar>
void dumpVTK(const typename ippl::FDTDSolver<scalar, 3>::bunch_type& pbunch, ippl::Field<ippl::Vector<scalar, 3>, 3, ippl::UniformCartesian<scalar, 3>,
                        typename ippl::UniformCartesian<scalar, 3>::DefaultCentering>& E,
             int nx, int ny, int nz, int iteration, scalar dx, scalar dy, scalar dz) {

    Kokkos::View<ippl::Vector<scalar, 3U> ***> collected_view;
    Kokkos::View<ippl::Vector<scalar, 3>*> pbunchgview;
        
    if(ippl::Comm->size() > 1){
        collected_view = collect_field_on_zero(E, nx, ny, nz);
        pbunchgview = collect_linear_view_on_zero(pbunch.R.getView(), pbunch.getLocalNum());
    }
    else{
        pbunchgview = pbunch.R.getView();
        collected_view = Kokkos::View<ippl::Vector<scalar, 3U> ***>("", nx, ny, nz);
        auto eview = E.getView();
        Kokkos::parallel_for(ippl::getRangePolicy(E.getView(), 1), KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
            collected_view(i-1,j-1,k-1) = eview(i,j,k);
        });
        Kokkos::fence();
    }
    if(ippl::Comm->rank())return;
    //std::cout << "Extents: " << collected_view.extent(0) << ", " << collected_view.extent(1) << ", " << collected_view.extent(2) << std::endl;
    using Mesh_t      = ippl::UniformCartesian<scalar, 3>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<ippl::Vector<scalar, 3>, 3, Mesh_t, Centering_t> VField_t;
    typename VField_t::view_type::host_mirror_type host_view = E.getHostMirror();

    std::stringstream fname;
    constexpr char endl = '\n';
    fname << "data/ef_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, E.getView());

    std::ofstream vtkout(fname.str());
    vtkout.precision(4);
    //vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0" << endl;
    vtkout << "TestFDTD" << endl;
    vtkout << "ASCII" << endl;
    vtkout << "DATASET STRUCTURED_POINTS" << endl;
    //vtkout << "DIMENSIONS " << nx + 3 << " " << ny + 3 << " " << nz + 3 << endl;
    vtkout << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " " << nz + 1 << endl;
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << endl;
    vtkout << "SPACING " << dx << " " << dy << " " << dz << endl;
    vtkout << "CELL_DATA " << (nx) * (ny) * (nz) << endl;
    //vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << endl;

    vtkout << "VECTORS E-Field float" << endl;
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                //vtkout << host_view(x, y, z)[0] << "\t" << host_view(x, y, z)[1] << "\t"
                //       << host_view(x, y, z)[2] << endl;
                vtkout << collected_view(x, y, z)[0] << "\t" << collected_view(x, y, z)[1] << "\t"
                       << collected_view(x, y, z)[2] << endl;
            }
        }
    }
    
    vtkout << "----\n";
    for (size_t z = 0; z < pbunchgview.extent(0);z++) {
        vtkout << pbunchgview(z)[0] << " " << pbunchgview(z)[1] << " " << pbunchgview(z)[2] << "\n";
    }
}
template<typename scalar>
void dumpVTK(ippl::Field<scalar, 3, ippl::UniformCartesian<scalar, 3>,
                         typename ippl::UniformCartesian<scalar, 3>::DefaultCentering>& rho,
             int nx, int ny, int nz, int iteration, scalar dx, scalar dy, scalar dz) {
    using Mesh_t      = ippl::UniformCartesian<scalar, 3>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<scalar, 3, Mesh_t, Centering_t> Field_t;
    typename Field_t::view_type::host_mirror_type host_view = rho.getHostMirror();

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, rho.getView());
    Kokkos::View<scalar***> sv = Kokkos::subview(host_view, Kokkos::make_pair(3,5), Kokkos::make_pair(3,5), Kokkos::make_pair(3,5));
    //Kokkos::subview()
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
    for (int x = 0; x < nx + 2; x++) {
        for (int y = 0; y < ny + 2; y++) {
            for (int z = 0; z < nz + 2; z++) {
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
        using scalar = float;
        // get the total simulation time from the user
        const scalar time_simulated = std::atof(argv[4]);
        if(time_simulated <= 0){
            std::cerr << "Time must be > 0\n";
            goto exit;
        }
        using Mesh_t      = ippl::UniformCartesian<scalar, Dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        typedef ippl::Field<scalar, Dim, Mesh_t, Centering_t> Field_t;
        typedef ippl::Field<ippl::Vector<scalar, Dim>, Dim, Mesh_t, Centering_t> VField_t;

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
        scalar dx                        = scalar(1.0) / nr[0];
        scalar dy                        = scalar(1.0) / nr[1];
        scalar dz                        = scalar(1.0) / nr[2];
        ippl::Vector<scalar, Dim> hr     = {dx, dy, dz};
        ippl::Vector<scalar, Dim> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);
        
        // CFL condition lambda = c*dt/h < 1/sqrt(d) = 0.57 for d = 3
        // we set a more conservative limit by choosing lambda = 0.5
        // we take h = minimum(dx, dy, dz)
        const scalar c = 1.0;  // 299792458.0;
        scalar dt      = std::min({dx, dy, dz}) * 0.125 / c;
        unsigned int iterations = std::ceil(time_simulated / dt);
        dt = time_simulated / (scalar)iterations;
        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<Dim> layout(owned, decomp);
        if(false){
            Field_t halo_test_field;        

            halo_test_field.initialize(mesh, layout);

            auto _tv = halo_test_field.getView();   
            const scalar ronk = ippl::Comm->rank();
            Kokkos::parallel_for(halo_test_field.getFieldRangePolicy(), KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                _tv(i, j, k) = ronk;
            });
            Kokkos::fence();
            if(ippl::Comm->rank() == 0){
                std::cout << "BV: " << _tv(5, 5, _tv.extent(0) - 1) << "\n";
            }
            ippl::Comm->barrier();
            halo_test_field.fillHalo();
            Kokkos::fence();
            if(ippl::Comm->rank() == 0){
                std::cout << "BV: " << _tv(5, 5, _tv.extent(0) - 1) << "\n";
            }
        }
        //Define particle layout and bunch
        
        //bunch.create(100);
        //bunch.setParticleBC(ippl::NO);
        //bunch.Q = 1.0;
        //bunch.R = 0.5 + hr[0] * 0.5;
        //std::cout << bunch.R.getView()(0) << "\n";
        //std::cout << bunch.Q.getView()(0) << "\n";
        //std::cin.get();

        // define the R (rho) field
        Field_t rho;
        rho.initialize(mesh, layout);
        // define the Vector field E (LHS)
        VField_t fieldE, fieldB;
        fieldE.initialize(mesh, layout);
        fieldB.initialize(mesh, layout);
        fieldE = 0.0;
        fieldB = 0.0;

        // define current = 0
        VField_t current;
        current.initialize(mesh, layout);

        
        //std::stringstream sstr;
        //sstr << ippl::Comm->rank() << ": " << current.getLayout().getLocalNDIndex()[0].first() << " to " << current.getLayout().getLocalNDIndex()[0].last() << "\n";
        //sstr << ippl::Comm->rank() << ": " << current.getLayout().getLocalNDIndex()[1].first() << " to " << current.getLayout().getLocalNDIndex()[1].last() << "\n";
        //sstr << ippl::Comm->rank() << ": " << current.getLayout().getLocalNDIndex()[2].first() << " to " << current.getLayout().getLocalNDIndex()[2].last() << "\n";
        //std::cout << sstr.str() << std::endl;
        //return 0;
        current = 0.0;

        // turn on the seeding (gaussian pulse) - if set to false, sine pulse is added on rho
        bool seed = false;

        // define an FDTDSolver object
        ippl::FDTDSolver<scalar, Dim> solver(&rho, &current, &fieldE, &fieldB, dt, seed);
        
        //if(!ippl::Comm->rank()){
        //    std::cout << solver.aN_m.getLayout().getDomain().last() << "\n";
        //}
        //return 0;
        
        solver.bconds[0] = ippl::MUR_ABC_1ST;
        solver.bconds[1] = ippl::MUR_ABC_1ST;
        solver.bconds[2] = ippl::MUR_ABC_1ST;
        //solver.bconds[0] = ippl::PERIODIC_FACE;
        //solver.bconds[1] = ippl::PERIODIC_FACE;
        //solver.bconds[2] = ippl::PERIODIC_FACE;
        decltype(solver)::bunch_type bunch_buffer(solver.pl);
        solver.pl.update(solver.bunch, bunch_buffer);

        /*
        std::cout << nr[0] << " " << current.getView().extent(0) << "\n";
        std::cout << nr[1] << " " << current.getView().extent(1) << "\n";
        std::cout << nr[2] << " " << current.getView().extent(2) << "\n";
        std::cout << nr[0] << " " << solver.aN_m.getView().extent(0) << "\n";
        std::cout << nr[1] << " " << solver.aN_m.getView().extent(1) << "\n";
        std::cout << nr[2] << " " << solver.aN_m.getView().extent(2) << "\n";
        return 0;
        */
        //bunch.Q.scatter(rho, bunch.R);
        //std::cout << rho.getView()(nr[0] / 2, nr[1] / 2, nr[2] / 2) << "\n";
        //std::cin.get();
        if (!seed) {

            // add pulse at center of domain
            auto view_rho    = rho.getView();
            //const int nghost = rho.getNghost();
            //auto ldom        = layout.getLocalNDIndex();
            auto view_a      = solver.aN_m.getView();
            auto view_an1    = solver.aNm1_m.getView();
            if(false)
            solver.fill_initialcondition(
                KOKKOS_LAMBDA(scalar x, scalar y, scalar z) {
                    ippl::Vector<scalar, 3> ret(0.0);
                    ret[2] = gauss(/*std::hypot(x - 0.5, y - 0.5, z - 0.5)*/ y, 0.2, 0.05);
                    //(void)x;
                    //(void)y;
                    //(void)z;
                    return ret;
            });
        }
        msg << "Timestep number = " << 0 << " , time = " << 0 << endl;
        //dumpVTK(solver.bunch, fieldB, nr[0], nr[1], nr[2], 0, hr[0], hr[1], hr[2]);
        solver.solve();
        //dumpVTK(solver.bunch, fieldB, nr[0], nr[1], nr[2], 1, hr[0], hr[1], hr[2]);
        // time-loop
        for (unsigned int it = 1; it < iterations; ++it) {
            msg << "Timestep number = " << it << " , time = " << it * dt << endl;

            if (!seed) {
                // add pulse at center of domain
                //auto view_rho    = rho.getView();
                //auto view_a    = solver.aN_m.getView();
                //const int nghost = rho.getNghost();
                //auto ldom        = layout.getLocalNDIndex();
            }

            solver.solve();
            //
            
            //dumpVTK(solver.bunch, fieldB, nr[0], nr[1], nr[2], it + 1, hr[0], hr[1], hr[2]);
            //if(it == iterations / 2){
            //    dumpPOV(solver.bunch, fieldB, nr[0], nr[1], nr[2], it + 1, hr[0], hr[1], hr[2]);
            //}
        }
        //dumpVTK(solver.bunch, fieldB, nr[0], nr[1], nr[2], iterations + 1, hr[0], hr[1], hr[2]);
        if (!seed) {
            // add pulse at center of domain
            
            [[maybe_unused]] auto view_rho    = rho.getView();
            [[maybe_unused]] const int nghost = rho.getNghost();
            [[maybe_unused]] auto ldom        = layout.getLocalNDIndex();
            [[maybe_unused]] auto view_a      = solver.aN_m.getView();
            [[maybe_unused]] auto view_b      = fieldB.getView();
            [[maybe_unused]] auto view_e      = fieldE.getView();
            
            auto view_an1    = solver.aNm1_m.getView();
            //Kokkos::View<double***> energy_density("Energies", view_a.extent(0), view_a.extent(1),
            //                                   view_a.extent(2));
        
            scalar error_accumulation = 0.0;
            //const double volume = (1.0 / (nr[0] - 6)) * (1.0 / (nr[1] - 6)) * (1.0 / (nr[2] - 6));
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
            error_accumulation = solver.volumetric_integral(KOKKOS_LAMBDA(const int i, const int j, const int k, scalar x, scalar y, scalar z){
                return std::abs(view_a(i, j, k)[2] - gauss(/*std::hypot(x - 0.5, y - 0.5, z - 0.5)*/ y - 0.5, 0.0, 0.1));
                (void)i;(void)x;
                (void)j;(void)y;
                (void)k;(void)z;
            });
            std::cout << "TOTAL ERROR: " << error_accumulation << std::endl;
        }
    }
    exit:
    ippl::finalize();

    return 0;
}
