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
#include <json.hpp>
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
template<typename scalar, unsigned Dim>
ippl::Vector<scalar, Dim> getVector(const nlohmann::json& j){
    //assert(j.is_array());
    
    if(j.is_array()){
        assert(j.size() == Dim);
        ippl::Vector<scalar, Dim> ret;
        for(unsigned i = 0;i < Dim;i++)
            ret[i] = (scalar)j[i];
        return ret;
    }
    else{
        std::cerr << "Warning: Obtaining Vector from scalar json\n";
        return ippl::Vector<scalar, Dim>((scalar)j);
    }
}
template<size_t N, typename T>
struct DefaultedStringLiteral {
    constexpr DefaultedStringLiteral(const char (&str)[N], const T val) : value(val) {
        std::copy_n(str, N, key);
    }
    
    T value;
    char key[N];
};

template<size_t N>
struct StringLiteral {
    constexpr StringLiteral(const char (&str)[N]) {
        std::copy_n(str, N, value);
    }
    
    char value[N];
    constexpr DefaultedStringLiteral<N, int> operator>>(int t)const noexcept{
        return DefaultedStringLiteral<N, int>(value, t);
    }
    constexpr size_t size()const noexcept{return N - 1;}
};
template<StringLiteral lit>
constexpr size_t chash(){
    size_t hash = 5381;
    int c;

    for(size_t i = 0;i < lit.size();i++){
        c = lit.value[i];
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }

    return hash;
}
size_t chash(const char* val) {
    size_t hash = 5381;
    int c;

    while ((c = *val++)) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }

    return hash;
}
size_t chash(const std::string& _val) {
    size_t hash = 5381;
    const char* val = _val.c_str();
    int c;

    while ((c = *val++)) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }

    return hash;
}
template<typename view_value_type>
Kokkos::View<view_value_type*> collect_linear_view_on_zero(Kokkos::View<view_value_type*> src_view, size_t locsize){
    //return Kokkos::View<view_value_type*>("", 0);
    size_t local_field_size = locsize;
    std::vector<size_t> field_sizes(ippl::Comm->size(), 0);

    //std::cout << "pg" << std::endl;// Gather the counts of data from all processes to the root
    MPI_Gather(&local_field_size, 1, MPI_UNSIGNED_LONG, field_sizes.data(), 1, MPI_UNSIGNED_LONG, 0, ippl::Comm->getCommunicator());
    //if(!ippl::Comm->rank()){
    //    size_t ts = 0;
    //    for(auto fs : field_sizes)ts += fs;
    //    LOG("Total: " << ts);
    //}
    
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
    ippl::Vector<scalar, Dim> extents;
    std::unique_ptr<Mesh_t> mesh;
    std::unique_ptr<ippl::FieldLayout<Dim>> layout;
    nlohmann::json custom_options;
    typename s_t::Field_t::BConds_t m_ic_scalar_bcs;
    using Vector_t = typename s_t::Vector_t;
    private:
    size_t required_steps;
    size_t particle_count;
    public:
    fdtd_initer(ippl::Vector<int, Dim> _res, ippl::Vector<_scalar, Dim> _ext, size_t PC, scalar end_time, const nlohmann::json& _opt) : extents(_ext), custom_options(_opt) {
        
        particle_count = PC;
        // domain
        ippl::NDIndex<Dim> owned;
        nr = _res;
        scalar dt = _ext[0] / _res[0];
        for (unsigned i = 0; i < Dim; i++) {
            owned[i] = ippl::Index(nr[i]);
            hr[i]    = _ext[i] / _res[i];
            dt = std::min(dt, hr[i]);
        }

        bool seed                        = false;
        ippl::Vector<scalar, Dim> origin = {0.0, 0.0, 0.0};
        mesh = std::make_unique<Mesh_t>(owned, hr, origin);
        const scalar c = 1.0;  // 299792458.0;
        dt *= 0.5 / c;
        if(custom_options.contains("timestep-ratio")){
            dt *= (scalar)custom_options["timestep-ratio"];
        }
        required_steps = std::ceil(end_time / dt);
        
        dt = end_time / required_steps;
        std::cout << "picking dt = " << dt << "\n";
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
        ippl::FDTDParticleUpdateRule pur;
        ippl::FDTDFieldUpdateRule fur;
        //std::cerr << std::string(custom_options["bunch"]["update-rule"]["type"]) << "\n";
        switch(chash(std::string(custom_options["bunch"]["update-rule"]["type"]))){
            case chash<"circular-orbit">():
                pur = ippl::FDTDParticleUpdateRule::CIRCULAR_ORBIT;
            break;
            case chash<"dipole-orbit">():
                pur = ippl::FDTDParticleUpdateRule::DIPOLE_ORBIT;
            break;
            case chash<"lorentz">():
                pur = ippl::FDTDParticleUpdateRule::LORENTZ;
            break;
            case chash<"stationary">():
                pur = ippl::FDTDParticleUpdateRule::STATIONARY;
            break;
            default:
                assert(false);
                pur = ippl::FDTDParticleUpdateRule::XLINE;
        }
        switch(chash(std::string(custom_options["field"]["update-rule"]))){
            case chash<"do">():
                fur = ippl::FDTDFieldUpdateRule::DO;
            break;
            case chash<"dont">():
            case chash<"don't">():
                fur = ippl::FDTDFieldUpdateRule::DONT;
            break;
            default:
                assert(false);
                fur = ippl::FDTDFieldUpdateRule::DO;
        }

        m_solver = std::make_unique<s_t>(rho, current, fieldE, fieldB, PC,
                                         ippl::FDTDBoundaryCondition::ABC_FALLAHI,
                                         pur,
                                         fur, dt, seed, &radiation);
        setupBunch(PC);
        initialConditionPhi();

        //initialConditionA();
        m_solver->setBoundaryConditions(ippl::Vector<ippl::FDTDBoundaryCondition, 3>(ippl::FDTDBoundaryCondition::ABC_FALLAHI));
    }
    void initialConditionA(){
        auto av = m_solver->aN_m.getView();
        auto am1v = m_solver->aNm1_m.getView();
        auto& layout = m_solver->layout_mp;
        const ippl::NDIndex<Dim> lDom       = layout->getLocalNDIndex();
        auto nr = this->nr;
        auto extents = this->extents;
        const unsigned nghost = m_solver->aN_m.getNghost();
        double strength = custom_options["field"]["strength"];
        Kokkos::parallel_for(ippl::getRangePolicy(av), KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
            ippl::Vector<size_t, 3> args = ippl::Vector<size_t, 3>{i, j, k} - lDom.first() + nghost;
            av(i, j, k) =   ippl::Vector<scalar, 3>{scalar(0), scalar(args[2]) / nr[2] * extents[2] * (scalar)strength, scalar(0)};
            av(i, j, k) =   ippl::Vector<scalar, 3>(0.0);
            am1v(i, j, k) = ippl::Vector<scalar, 3>{scalar(0), scalar(args[2]) / nr[2] * extents[2] * (scalar)strength, scalar(0)};
            //am1v(i, j, k) = ippl::Vector<scalar, 3>{scalar(0), scalar(args[2]) / nr[2] * extents[2] * (scalar)strength, scalar(0)};
            am1v(i, j, k) =   ippl::Vector<scalar, 3>(0.0);
        });
    }
    void setupBunch(size_t pc){
        (void)pc;
        m_solver->bunch.setParticleBC(ippl::BC::PERIODIC);

        //Not required, since done by m_solver constructor
        //m_solver->bunch.create(pc);
        auto srview   = m_solver->bunch.R.getView();
        auto srn1view = m_solver->bunch.R_nm1.getView();
        auto gbrview  = m_solver->bunch.gamma_beta.getView();

        Vector_t bunchpos     = getVector<scalar, Dim>(custom_options["bunch"]["bunch-initialization"]["position"]);// + extents * 0.5; //Mithra is centered
        Vector_t direction    = getVector<scalar, Dim>(custom_options["bunch"]["bunch-initialization"]["direction"]);
        Vector_t pos_var      = getVector<scalar, Dim>(custom_options["bunch"]["bunch-initialization"]["sigma-position"]);
        Vector_t momentum_var = getVector<scalar, Dim>(custom_options["bunch"]["bunch-initialization"]["sigma-momentum"]);
        Vector_t gamma_means  = getVector<scalar, Dim>(custom_options["bunch"]["bunch-initialization"]["particle-gammabetas"]);

        double externalStrength = custom_options["field"]["strength"];
        m_solver->externalMagneticScale = externalStrength;
        Kokkos::Random_XorShift64_Pool<> rand_pool(12345);
        scalar dt = m_solver->dt;
        Kokkos::parallel_for(
            Kokkos::RangePolicy<
                typename s_t::playout_type::RegionLayout_t::view_type::execution_space>(
                0, m_solver->bunch.getLocalNum()),
                KOKKOS_LAMBDA(size_t idx){
                    using Kokkos::sqrt;
                    auto state = rand_pool.get_state();
                    Vector_t gammabeta;
                    for(unsigned i = 0;i < Dim;i++){
                        srview(idx)[i] = state.normal(bunchpos[i], pos_var[i]);
                        const scalar gammabeta_i = state.normal(gamma_means[i], momentum_var[i]) * direction[i];
                        //const scalar gammabeta_i = sqrt(gamma_i * gamma_i - 1.0);
                        //std::cout << gammabeta_i << " GAMA\n";
                        gbrview(idx)[i] = gammabeta_i;
                        gammabeta[i] = gammabeta_i;
                    }
                    srn1view(idx) = srview(idx) - dt * gammabeta / (sqrt(1.0 + dot_prod(gammabeta, gammabeta)));
                    rand_pool.free_state(state);
                }
                // generate_random<ippl::Vector<scalar, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                //     solver.bunch.R.getView(),
                //     solver.bunch.R_nm1.getView(),
                //     solver.bunch.gamma_beta.getView(),
                //     regions_view(ippl::Comm->rank()),
                //     rand_pool
                //)
                //KOKKOS_LAMBDA(size_t idx) {
                //    srview(idx) =
                //        ippl::Vector<scalar, Dim>{(idx) / scalar(pc), 0.5, 0.5};
                //    srn1view(idx) =
                //        ippl::Vector<scalar, Dim>{(idx) / scalar(pc), 0.5, 0.5};
                //    gbrview(idx) = ippl::Vector<scalar, Dim>{0.0, 0.0, 0.0};
                //}
            );
        Kokkos::fence();
    }
    void initialConditionPhi() {
        
        auto& ic_scalar_bcs = m_ic_scalar_bcs;
        const auto hr = this->hr;

        m_solver->pl.update(m_solver->bunch);
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
        //ippl::Vector<ippl::FDTDBoundaryCondition, Dim> fdtdbcs(ippl::FDTDBoundaryCondition::ABC_MUR);
        //fdtdbcs[0] = ippl::FDTDBoundaryCondition::PERIODIC;
        //m_solver->setBoundaryConditions(fdtdbcs);
    }
    void doRequiredSteps(){
        std::unique_ptr<std::ofstream> brad;
        std::unique_ptr<std::ofstream> crad;
        std::unique_ptr<std::ofstream> lrad;
        std::unique_ptr<std::ofstream> pp0;
        
        
        if(ippl::Comm->rank() == 0 && custom_options["output"].contains("track")){
            //std::cerr << "contains track\n";
            if(custom_options["output"]["track"].contains("radiation")){
                brad = std::make_unique<std::ofstream>((std::string)custom_options["output"]["track"]["radiation"]);
                m_solver->output_stream[ippl::trackableOutput::boundaryRadiation] = brad.get();
            }
            if(custom_options["output"]["track"].contains("cumulative-radiation")){
                crad = std::make_unique<std::ofstream>((std::string)custom_options["output"]["track"]["cumulative-radiation"]);
                m_solver->output_stream[ippl::trackableOutput::cumulativeRadiation] = crad.get();
                //std::cerr << "Added boundaryradiation tracker\n";
            }
            if(custom_options["output"]["track"].contains("particle-position")){
                pp0 = std::make_unique<std::ofstream>((std::string)custom_options["output"]["track"]["particle-position"]);
                m_solver->output_stream[ippl::trackableOutput::p0pos] = pp0.get();
                //std::cerr << "Added boundaryradiation tracker\n";
            }
        }
        for (unsigned int it = 0; it < required_steps; ++it){
            m_solver->solve();
            //std::cout << ippl::Info->getOutputLevel() << "\n";
            size_t count = 50;
            if(custom_options.contains("output") && custom_options["output"].contains("count")){
                count = custom_options["output"]["count"];
            }
            auto rsteps = std::max((size_t)1, (required_steps / std::max(1ul, count)));
            if(count && ippl::Info->getOutputLevel() == 5 && (it % rsteps) == 0){
                switch(chash(custom_options["output"]["type"])){
                    case chash<"B">():
                    case chash<"magnet">():
                    case chash<"magnetic">():
                    dumpVTK(fieldB, nr[0], nr[1], nr[2], it, hr[0], hr[1], hr[2]);
                    break;
                    case chash<"E">():
                    case chash<"electric">():
                    dumpVTK(fieldE, nr[0], nr[1], nr[2], it, hr[0], hr[1], hr[2]);
                    break;
                    case chash<"rad">():
                    case chash<"radiation">():
                    dumpVTK(radiation, nr[0], nr[1], nr[2], it, hr[0], hr[1], hr[2]);
                    break;
                }
            }
        }
        evaluation();
    }
    void evaluation(){
        return;
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

template <char... C> 
constexpr StringLiteral<sizeof...(C)> operator "" _option(){
    return StringLiteral<sizeof...(C)>({C...});
}
/*template<size_t N>
DefaultedStringLiteral<N, int> operator>>(const char (&str)[N], int t){
    return DefaultedStringLiteral<N, int>(str, t);
}*/
template<size_t Index, DefaultedStringLiteral... lits>
struct options_imp;
template<size_t Index, DefaultedStringLiteral lit1, DefaultedStringLiteral lit2, DefaultedStringLiteral... lits>
struct options_imp<Index, lit1, lit2, lits...>{
    constexpr static DefaultedStringLiteral value = lit1;
    constexpr static size_t index = Index;
    using next = options_imp<Index + 1, lit2, lits...>;
    template<size_t I>
    constexpr static auto get(){
        if constexpr(I == Index){
            return value;
        }
        else if constexpr(I > Index){
            return next::template get<I>();
        }
    }
};
template<size_t Index, DefaultedStringLiteral lit>
struct options_imp<Index, lit>{
    constexpr static size_t index = Index;
};
template<DefaultedStringLiteral... lit>
struct options{
    using datatype = options_imp<0, lit...>;
    options(){
        //auto values = {lit...};
    }
};
template<DefaultedStringLiteral L>
struct P{
    P(){
        std::cout << L.value << "\n";
    }
};

auto jdefault(const nlohmann::json& v, const std::string& k, auto dv){
    if(v.contains(k)){
        return decltype(dv)(v[k]);
    }
    std::cerr << "Defaulting option " << k << " to " << dv << "\n";
    return dv;
}

int main(int argc, char* argv[]){
    ippl::initialize(argc, argv);
    {
        //lorentz_frame<double>  frame(ippl::Vector<double, 3>{0.5,0.7,0.9});
        //lorentz_frame<double> iframe(ippl::Vector<double, 3>{-0.5,-0.7,-0.9});
        //matrix<double, 4, 4> lorenz  =  frame.unprimedToPrimed();
        //matrix<double, 4, 4> lorenz2 = iframe.primedToUnprimed();
        //std::cout << lorenz << "\n\n";
        //std::cout << lorenz2 << "\n";
        //using vec3 = ippl::Vector<double, 3>;
        //Kokkos::pair<ippl::Vector<double, 3>, ippl::Vector<double, 3>> eb;
        //eb.first = vec3(0);
        //eb.second = vec3{0,0,1};
        //std::cout << frame.transform_EB(eb).first << "\n";

        //lorenz.inverse();

        //options<("mesh-lengths"_option)> option_getter;
        //std::cout << decltype(option_getter)::datatype::get<0>().value << "\n";
        std::ifstream cfile("config.json");
        nlohmann::json j;
        cfile >> j;
        constexpr unsigned Dim = 3;
        ippl::Vector<int, Dim> res = getVector<int, Dim>(j["mesh"]["resolution"]);
        ippl::Vector<double, Dim> ext = ippl::Vector<double, 3>{j["mesh"]["extents"][0], j["mesh"]["extents"][1], j["mesh"]["extents"][2]};
        size_t pc = j["bunch"]["bunch-initialization"]["number-of-particles"];
        double time = j["mesh"]["total-time"];
        std::cerr << "Res: " << res << "\n";
        std::cerr << "Ext: " << ext << "\n";
        fdtd_initer<Dim> fdtd(res, ext, pc, time, j);
        fdtd.doRequiredSteps();
    }
    ippl::finalize();

    return 0;
}