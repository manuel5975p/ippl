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
#include <cstring>
const char* from_last_slash(const char* x){
    size_t len = std::strlen(x);
    const char* end = x + len;
    while(*(end - 1) != '/')--end;
    return end;
}

#define LOG(X) std::cout << from_last_slash(__FILE__) << ':' << __LINE__ << ": " << X << "\n"
#include <iostream>
#define FRAST3D_IMPLEMENTATION
#include "rast3d.hpp"
//#include <quadmath.h>
#define gqf(X) __float128 X(__float128 x)noexcept{ \
    return X##q(x);\
}
namespace Kokkos{




/*gqf(sqrt);
gqf(sin);
gqf(floor);
gqf(cos);
gqf(exp);
__float128 max(__float128 x, __float128 y)noexcept{ 
    return fmaxq(x, y);
}
__float128 min(__float128 x, __float128 y)noexcept{ 
    return fminq(x, y);
}
}
std::ostream& operator<<(std::ostream& ostr, __float128 x){
    return ostr << (double)x;
}*/
#include "Ippl.h"
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <cstdlib>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <fstream>
#include "Field/BcTypes.h"
#include "Types/Vector.h"

#include "Particle/ParticleAttrib.h"
#include "Solver/FDTDSolver.h"

constexpr double turbo_cm[256][3] = {  {0.18995,0.07176,0.23217},  {0.19483,0.08339,0.26149},  {0.19956,0.09498,0.29024},  {0.20415,0.10652,0.31844},  {0.20860,0.11802,0.34607},  {0.21291,0.12947,0.37314},  {0.21708,0.14087,0.39964},  {0.22111,0.15223,0.42558},  {0.22500,0.16354,0.45096},  {0.22875,0.17481,0.47578},  {0.23236,0.18603,0.50004},  {0.23582,0.19720,0.52373},  {0.23915,0.20833,0.54686},  {0.24234,0.21941,0.56942},  {0.24539,0.23044,0.59142},  {0.24830,0.24143,0.61286},  {0.25107,0.25237,0.63374},  {0.25369,0.26327,0.65406},  {0.25618,0.27412,0.67381},  {0.25853,0.28492,0.69300},  {0.26074,0.29568,0.71162},  {0.26280,0.30639,0.72968},  {0.26473,0.31706,0.74718},  {0.26652,0.32768,0.76412},  {0.26816,0.33825,0.78050},  {0.26967,0.34878,0.79631},  {0.27103,0.35926,0.81156},  {0.27226,0.36970,0.82624},  {0.27334,0.38008,0.84037},  {0.27429,0.39043,0.85393},  {0.27509,0.40072,0.86692},  {0.27576,0.41097,0.87936},  {0.27628,0.42118,0.89123},  {0.27667,0.43134,0.90254},  {0.27691,0.44145,0.91328},  {0.27701,0.45152,0.92347},  {0.27698,0.46153,0.93309},  {0.27680,0.47151,0.94214},  {0.27648,0.48144,0.95064},  {0.27603,0.49132,0.95857},  {0.27543,0.50115,0.96594},  {0.27469,0.51094,0.97275},  {0.27381,0.52069,0.97899},  {0.27273,0.53040,0.98461},  {0.27106,0.54015,0.98930},  {0.26878,0.54995,0.99303},  {0.26592,0.55979,0.99583},  {0.26252,0.56967,0.99773},  {0.25862,0.57958,0.99876},  {0.25425,0.58950,0.99896},  {0.24946,0.59943,0.99835},  {0.24427,0.60937,0.99697},  {0.23874,0.61931,0.99485},  {0.23288,0.62923,0.99202},  {0.22676,0.63913,0.98851},  {0.22039,0.64901,0.98436},  {0.21382,0.65886,0.97959},  {0.20708,0.66866,0.97423},  {0.20021,0.67842,0.96833},  {0.19326,0.68812,0.96190},  {0.18625,0.69775,0.95498},  {0.17923,0.70732,0.94761},  {0.17223,0.71680,0.93981},  {0.16529,0.72620,0.93161},  {0.15844,0.73551,0.92305},  {0.15173,0.74472,0.91416},  {0.14519,0.75381,0.90496},  {0.13886,0.76279,0.89550},  {0.13278,0.77165,0.88580},  {0.12698,0.78037,0.87590},  {0.12151,0.78896,0.86581},  {0.11639,0.79740,0.85559},  {0.11167,0.80569,0.84525},  {0.10738,0.81381,0.83484},  {0.10357,0.82177,0.82437},  {0.10026,0.82955,0.81389},  {0.09750,0.83714,0.80342},  {0.09532,0.84455,0.79299},  {0.09377,0.85175,0.78264},  {0.09287,0.85875,0.77240},  {0.09267,0.86554,0.76230},  {0.09320,0.87211,0.75237},  {0.09451,0.87844,0.74265},  {0.09662,0.88454,0.73316},  {0.09958,0.89040,0.72393},  {0.10342,0.89600,0.71500},  {0.10815,0.90142,0.70599},  {0.11374,0.90673,0.69651},  {0.12014,0.91193,0.68660},  {0.12733,0.91701,0.67627},  {0.13526,0.92197,0.66556},  {0.14391,0.92680,0.65448},  {0.15323,0.93151,0.64308},  {0.16319,0.93609,0.63137},  {0.17377,0.94053,0.61938},  {0.18491,0.94484,0.60713},  {0.19659,0.94901,0.59466},  {0.20877,0.95304,0.58199},  {0.22142,0.95692,0.56914},  {0.23449,0.96065,0.55614},  {0.24797,0.96423,0.54303},  {0.26180,0.96765,0.52981},  {0.27597,0.97092,0.51653},  {0.29042,0.97403,0.50321},  {0.30513,0.97697,0.48987},  {0.32006,0.97974,0.47654},  {0.33517,0.98234,0.46325},  {0.35043,0.98477,0.45002},  {0.36581,0.98702,0.43688},  {0.38127,0.98909,0.42386},  {0.39678,0.99098,0.41098},  {0.41229,0.99268,0.39826},  {0.42778,0.99419,0.38575},  {0.44321,0.99551,0.37345},  {0.45854,0.99663,0.36140},  {0.47375,0.99755,0.34963},  {0.48879,0.99828,0.33816},  {0.50362,0.99879,0.32701},  {0.51822,0.99910,0.31622},  {0.53255,0.99919,0.30581},  {0.54658,0.99907,0.29581},  {0.56026,0.99873,0.28623},  {0.57357,0.99817,0.27712},  {0.58646,0.99739,0.26849},  {0.59891,0.99638,0.26038},  {0.61088,0.99514,0.25280},  {0.62233,0.99366,0.24579},  {0.63323,0.99195,0.23937},  {0.64362,0.98999,0.23356},  {0.65394,0.98775,0.22835},  {0.66428,0.98524,0.22370},  {0.67462,0.98246,0.21960},  {0.68494,0.97941,0.21602},  {0.69525,0.97610,0.21294},  {0.70553,0.97255,0.21032},  {0.71577,0.96875,0.20815},  {0.72596,0.96470,0.20640},  {0.73610,0.96043,0.20504},  {0.74617,0.95593,0.20406},  {0.75617,0.95121,0.20343},  {0.76608,0.94627,0.20311},  {0.77591,0.94113,0.20310},  {0.78563,0.93579,0.20336},  {0.79524,0.93025,0.20386},  {0.80473,0.92452,0.20459},  {0.81410,0.91861,0.20552},  {0.82333,0.91253,0.20663},  {0.83241,0.90627,0.20788},  {0.84133,0.89986,0.20926},  {0.85010,0.89328,0.21074},  {0.85868,0.88655,0.21230},  {0.86709,0.87968,0.21391},  {0.87530,0.87267,0.21555},  {0.88331,0.86553,0.21719},  {0.89112,0.85826,0.21880},  {0.89870,0.85087,0.22038},  {0.90605,0.84337,0.22188},  {0.91317,0.83576,0.22328},  {0.92004,0.82806,0.22456},  {0.92666,0.82025,0.22570},  {0.93301,0.81236,0.22667},  {0.93909,0.80439,0.22744},  {0.94489,0.79634,0.22800},  {0.95039,0.78823,0.22831},  {0.95560,0.78005,0.22836},  {0.96049,0.77181,0.22811},  {0.96507,0.76352,0.22754},  {0.96931,0.75519,0.22663},  {0.97323,0.74682,0.22536},  {0.97679,0.73842,0.22369},  {0.98000,0.73000,0.22161},  {0.98289,0.72140,0.21918},  {0.98549,0.71250,0.21650},  {0.98781,0.70330,0.21358},  {0.98986,0.69382,0.21043},  {0.99163,0.68408,0.20706},  {0.99314,0.67408,0.20348},  {0.99438,0.66386,0.19971},  {0.99535,0.65341,0.19577},  {0.99607,0.64277,0.19165},  {0.99654,0.63193,0.18738},  {0.99675,0.62093,0.18297},  {0.99672,0.60977,0.17842},  {0.99644,0.59846,0.17376},  {0.99593,0.58703,0.16899},  {0.99517,0.57549,0.16412},  {0.99419,0.56386,0.15918},  {0.99297,0.55214,0.15417},  {0.99153,0.54036,0.14910},  {0.98987,0.52854,0.14398},  {0.98799,0.51667,0.13883},  {0.98590,0.50479,0.13367},  {0.98360,0.49291,0.12849},  {0.98108,0.48104,0.12332},  {0.97837,0.46920,0.11817},  {0.97545,0.45740,0.11305},  {0.97234,0.44565,0.10797},  {0.96904,0.43399,0.10294},  {0.96555,0.42241,0.09798},  {0.96187,0.41093,0.09310},  {0.95801,0.39958,0.08831},  {0.95398,0.38836,0.08362},  {0.94977,0.37729,0.07905},  {0.94538,0.36638,0.07461},  {0.94084,0.35566,0.07031},  {0.93612,0.34513,0.06616},  {0.93125,0.33482,0.06218},  {0.92623,0.32473,0.05837},  {0.92105,0.31489,0.05475},  {0.91572,0.30530,0.05134},  {0.91024,0.29599,0.04814},  {0.90463,0.28696,0.04516},  {0.89888,0.27824,0.04243},  {0.89298,0.26981,0.03993},  {0.88691,0.26152,0.03753},  {0.88066,0.25334,0.03521},  {0.87422,0.24526,0.03297},  {0.86760,0.23730,0.03082},  {0.86079,0.22945,0.02875},  {0.85380,0.22170,0.02677},  {0.84662,0.21407,0.02487},  {0.83926,0.20654,0.02305},  {0.83172,0.19912,0.02131},  {0.82399,0.19182,0.01966},  {0.81608,0.18462,0.01809},  {0.80799,0.17753,0.01660},  {0.79971,0.17055,0.01520},  {0.79125,0.16368,0.01387},  {0.78260,0.15693,0.01264},  {0.77377,0.15028,0.01148},  {0.76476,0.14374,0.01041},  {0.75556,0.13731,0.00942},  {0.74617,0.13098,0.00851},  {0.73661,0.12477,0.00769},  {0.72686,0.11867,0.00695},  {0.71692,0.11268,0.00629},  {0.70680,0.10680,0.00571},  {0.69650,0.10102,0.00522},  {0.68602,0.09536,0.00481},  {0.67535,0.08980,0.00449},  {0.66449,0.08436,0.00424},  {0.65345,0.07902,0.00408},  {0.64223,0.07380,0.00401},  {0.63082,0.06868,0.00401},  {0.61923,0.06367,0.00410},  {0.60746,0.05878,0.00427},  {0.59550,0.05399,0.00453},  {0.58336,0.04931,0.00486},  {0.57103,0.04474,0.00529},  {0.55852,0.04028,0.00579},  {0.54583,0.03593,0.00638},  {0.53295,0.03169,0.00705},  {0.51989,0.02756,0.00780},  {0.50664,0.02354,0.00863},  {0.49321,0.01963,0.00955},  {0.47960,0.01583,0.01055} };
inline Color jet(double t){
    Color ret;
    int index = t * 256;
    index = std::min(std::max(index, 0), 255);
    ret.x = turbo_cm[index][0] * 255.0;
    ret.y = turbo_cm[index][1] * 255.0;
    ret.z = turbo_cm[index][2] * 255.0;
    ret.w = 255;
    return ret;
}
template<typename T, typename R>
KOKKOS_INLINE_FUNCTION auto sine(T n, R dt) {
    using Kokkos::sin;
    return 100.0 * sin(n * dt);
}
template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION auto gauss(T1 x, T2 mean, T3 stddev) {
    (void)x;
    (void)mean;
    (void)stddev;
    using Kokkos::exp;
    return exp(-(x - mean) * (x - mean) / (stddev * stddev));
}

template<typename T1, unsigned Dim, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION auto gauss(ippl::Vector<T1, Dim> x, T2 mean, T3 stddev) {
    (void)x;
    (void)mean;
    (void)stddev;
    using Kokkos::exp;
    T1 accum = 0.0;
    for(unsigned i = 0;i < Dim;i++){
        accum += (x[i] - mean) * (x[i] - mean);
    }
    return exp(-accum / (stddev * stddev));
}
template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {
    using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type = typename T::value_type;
    // Output View for the random numbers
    view_type Rview, GBview;
    ippl::NDRegion<value_type, Dim> inside;
    // The GeneratorPool
    GeneratorPool rand_pool;

    // Initialize all members
    generate_random(view_type x_, view_type v_, ippl::NDRegion<value_type, Dim> reg, GeneratorPool rand_pool_)
        :Rview(x_)
        ,GBview(v_)
        ,inside(reg)
        ,rand_pool(rand_pool_){}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        value_type u;
        for (unsigned d = 0; d < Dim; ++d) {
            
            Rview(i)[d] = (inside[d].min()+inside[d].max()) * 0.5;//rand_gen.drand(inside[d].min(),inside[d].max());
            GBview(i)[d] = rand_gen.drand(-100.0,100.0);
            GBview(i)[d] /= Kokkos::abs(GBview(i)[d]);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};
template<typename T, unsigned Dim>
std::string povstring(const ippl::Vector<T, Dim>& v){
    std::stringstream sstr;
    sstr.precision(4);
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
    Kokkos::View<ippl::Vector<scalar, 3>***> Eview = E.getView();
    typename Kokkos::View<ippl::Vector<scalar, 3>***>::host_mirror_type Eview_hmirror = Kokkos::create_mirror_view(Eview);
    Kokkos::deep_copy(Eview_hmirror, Eview);
    //int lindex_xextent = lindex[0].last() - lindex[0].first() + 1;
    //if(lindex_xextent != E.getFieldRangePolicy(1).m_upper[0] - E.getFieldRangePolicy(1).m_lower[0]){
    //    std::cout << lindex_xextent << " " << E.getFieldRangePolicy(0).m_upper[0] - E.getFieldRangePolicy(0).m_lower[0] << std::endl;
    //    throw 5;
    //}
    //if(E.getFieldRangePolicy(1).m_upper[0] - E.getFieldRangePolicy(1).m_lower[0] != nx){std::cout << "NX = " << nx << ", DIFF = " << E.getFieldRangePolicy(1).m_upper[0] - E.getFieldRangePolicy(1).m_lower[0] << std::endl;throw 5;}
    for(int64_t i = E.getFieldRangePolicy(0).m_lower[0];i < E.getFieldRangePolicy(0).m_upper[0];i++){
        for(int64_t j = E.getFieldRangePolicy(0).m_lower[1];j < E.getFieldRangePolicy(0).m_upper[1];j++){
            for(int64_t k = E.getFieldRangePolicy(0).m_lower[2];k < E.getFieldRangePolicy(0).m_upper[2];k++){
                local_field_dump.push_back(Eview_hmirror(i,j,k));
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
    
    std::vector<int> rcnts(ippl::Comm->size());
    std::vector<int> displs(ippl::Comm->size());

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
    MPI_Gatherv(local_field_dump.data(), local_field_dump.size() * 3, std::is_same_v<std::remove_all_extents_t<scalar>, double> ? MPI_DOUBLE : MPI_FLOAT, complete_field_collection.data(), rcnts.data(), displs.data(), std::is_same_v<std::remove_all_extents_t<scalar>, double> ? MPI_DOUBLE : MPI_FLOAT, 0, ippl::Comm->getCommunicator());
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
        //LOG("Total: " << ts);
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
    Kokkos::View<ippl::Vector<scalar, 3>*> pbunch_pos_gathered;
    Kokkos::View<ippl::Vector<scalar, 3>*> pbunch_gammabeta_gathered;
        
    if(ippl::Comm->size() > 1){
        collected_view = collect_field_on_zero(E, nx, ny, nz);
        pbunch_pos_gathered = collect_linear_view_on_zero(pbunch.R.getView(), pbunch.getLocalNum());
        pbunch_gammabeta_gathered = collect_linear_view_on_zero(pbunch.gamma_beta.getView(), pbunch.getLocalNum());
    }
    else{
        pbunch_pos_gathered = pbunch.R.getView();
        collected_view = Kokkos::View<ippl::Vector<scalar, 3U> ***>("", nx, ny, nz);
        auto eview = E.getView();
        Kokkos::parallel_for(ippl::getRangePolicy(E.getView(), 1), KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
            collected_view(i-1,j-1,k-1) = eview(i,j,k);
        });
        
        Kokkos::fence();
    }
    //std::cout << "Lokal: " << pbunch.getLocalNum() << "\n";
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
    for (size_t z = 0; z < pbunch_pos_gathered.extent(0);z++) {
        vtkout << pbunch_pos_gathered(z)[0] << " " << pbunch_pos_gathered(z)[1] << " " << pbunch_pos_gathered(z)[2] << "\n";
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
        using scalar = double;
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

        //std::cout << std::is_conti<Field_t> << "\n";
        //goto exit;

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
        unsigned int iterations = std::ceil((double)(time_simulated / dt));
        dt = time_simulated / (scalar)iterations;
        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<Dim> layout(owned, decomp);

        //Define particle layout and bunch
        

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
        VField_t radiation;
        radiation.initialize(mesh, layout);
        
        current = 0.0;

        // turn on the seeding (gaussian pulse) - if set to false, sine pulse is added on rho
        bool seed = false;

        // define an FDTDSolver object
        ippl::FDTDSolver<scalar, Dim> solver(&rho, &current, &fieldE, &fieldB, dt, /*Particle count*/ 0, &radiation,seed);
        
        using s_t = ippl::FDTDSolver<scalar, Dim>;
        s_t::VField_t::BConds_t vector_bcs;
        s_t::Field_t::BConds_t  scalar_bcs;

        typename s_t::playout_type::RegionLayout_t const& rlayout = solver.pl.getRegionLayout();
        typename s_t::playout_type::RegionLayout_t::view_type regions_view = rlayout.gethLocalRegions();

        Kokkos::Random_XorShift64_Pool<> rand_pool((size_t)(42 + 100 * ippl::Comm->rank()));
        Kokkos::parallel_for(
            solver.bunch.getLocalNum(),
            generate_random<ippl::Vector<scalar, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                solver.bunch.R.getView(),
                solver.bunch.gamma_beta.getView(),
                regions_view(ippl::Comm->rank()),
                rand_pool
            )
        );
        auto bcsetter_single = [&vector_bcs, &scalar_bcs, hr, dt]<size_t Idx>(const std::index_sequence<Idx>&){
            //vector_bcs[Idx] = std::make_shared<ippl::MurABC1st<s_t::VField_t, Idx>>(Idx, hr, 1.0, dt);
            //scalar_bcs[Idx] = std::make_shared<ippl::MurABC1st<s_t::Field_t , Idx>>(Idx, hr, 1.0, dt);
            vector_bcs[Idx] = std::make_shared<ippl::PeriodicFace<s_t::VField_t>>(Idx);
            scalar_bcs[Idx] = std::make_shared<ippl::PeriodicFace<s_t:: Field_t>>(Idx);
            return 0;
        };
        auto bcsetter = [bcsetter_single]<size_t... Idx>(const std::index_sequence<Idx...>&){
            int x = (bcsetter_single(std::index_sequence<Idx>{}) ^ ...);
            (void) x;
        };
        bcsetter(std::make_index_sequence<Dim * 2>{});
        solver.aN_m.setFieldBC(vector_bcs);
        solver.aNp1_m.setFieldBC(vector_bcs);
        solver.aNm1_m.setFieldBC(vector_bcs);
        solver.phiN_m.setFieldBC(scalar_bcs);
        solver.phiNp1_m.setFieldBC(scalar_bcs);
        solver.phiNm1_m.setFieldBC(scalar_bcs);
        solver.bunch.setParticleBC(ippl::NO);
        
        solver.bconds[0] = ippl::MUR_ABC_1ST;
        solver.bconds[1] = ippl::MUR_ABC_1ST;
        solver.bconds[2] = ippl::MUR_ABC_1ST;
        //decltype(solver)::bunch_type bunch_buffer(solver.pl);
        //solver.pl.update(solver.bunch, bunch_buffer);
        if (!seed) {

            // add pulse at center of domain
            auto view_rho    = rho.getView();
            //const int nghost = rho.getNghost();
            //auto ldom        = layout.getLocalNDIndex();
            auto view_a      = solver.aN_m.getView();
            auto view_an1    = solver.aNm1_m.getView();
            //if(false)
            solver.fill_initialcondition(
                KOKKOS_LAMBDA(scalar x, scalar y, scalar z) {
                    ippl::Vector<scalar, 3> ret(0.0);
                    //std::cout << x << " x\n";
                    ret[2] = -1.0 * gauss(ippl::Vector<scalar, 3> {x, y, 0.2}, 0.2, 0.05);
                    (void)x;
                    (void)y;
                    (void)z;
                    return ret;
            });
        }
        
        msg << "Timestep number = " << 0 << " , time = " << 0 << endl;
        //solver.aN_m.setFieldBC()
        //dumpVTK(solver.bunch, radiation, nr[0], nr[1], nr[2], 0, hr[0], hr[1], hr[2]);
        //std::cout << "Before first step: " << solver.field_evaluation() << " ";
        //std::cout << solver.total_energy << "\n";
        solver.solve();
        std::cout.precision(10);
        LOG("After first step: " << solver.total_energy);
        //goto exit;
        //dumpVTK(solver.bunch, fieldB, nr[0], nr[1], nr[2], 1, hr[0], hr[1], hr[2]);
        // time-loop
        std::vector<scalar> energies;
        constexpr unsigned int ww = 1280, wh = 720;
        InitWindow(ww, wh);
        
        Mesh sphere_mesh = GenMeshSphere(0.3f, 12, 12);
        for (unsigned int it = 1; it < iterations; ++it) {
            if(ippl::Comm->rank() == 0)
                LOG("Timestep number: " << it);
            //msg << "Timestep number = " << it << " , time = " << it * dt << endl;

            if (!seed) {
                // add pulse at center of domain
                //auto view_rho    = rho.getView();
                //auto view_a    = solver.aN_m.getView();
                //const int nghost = rho.getNghost();
                //auto ldom        = layout.getLocalNDIndex();
            }

            solver.solve();
            //std::cout << solver.total_energy << "\n";
            //if(it > iterations / 6)
            energies.push_back(solver.total_energy);
            
            if(true || it % 10 == 0){
                Vector3<float> campos{(float)(-150.0 * std::cos(7.0 * (it * dt))), float(80.0 * (it * dt)), (float)(-150.0 * std::sin(7.0 * (it * dt)))};
                Vector3<float> center{nr[0] / 2.0f, nr[1] / 2.0f, nr[2] / 2.0f};
                Vector3<float> look = center - campos;
                
                camera cam(campos, look);
                //LOG(look << "\n");
                //LOG(cam.look_dir() << "\n");
                matrix_stack.push(cam.matrix(ww, wh));
                ClearBackground(Color{0, 15, 20, 255});
                auto bview = fieldB.getView();
                auto bunch_r_view = solver.bunch.R.getView();
                scalar maxnorm = -1.0;
                Kokkos::parallel_reduce(ippl::getRangePolicy(fieldB.getView()), KOKKOS_LAMBDA(size_t i, size_t j, size_t k, scalar& ref){
                    ippl::Vector<scalar, 3> bv = bview(i, j, k);
                    using Kokkos::max;
                    ref = max(bv.norm(), ref);
                }, maxnorm);
                scalar maxnorm_ar = -1.0;
                MPI_Allreduce(&maxnorm, &maxnorm_ar, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
                //DrawBillboardLineEx(Vector3<float>{0,0,0}, Vector3<float>{0,1,0}, 0.05f, Color{255,255,0,255});
                auto lindex_lower = solver.layout_mp->getLocalNDIndex().first();
                auto lindex_upper = solver.layout_mp->getLocalNDIndex().last();
                constexpr float lt = 0.01;
                int rankm3 = ippl::Comm->rank() % 8;
                Color lc{
                    (unsigned char)((rankm3 & 1) * 255),
                    (unsigned char)(((rankm3 >> 1) & 1) * 255),
                    (unsigned char)(((rankm3 >> 2) & 1) * 255), 255
                };

                auto ld = [lindex_lower, lindex_upper, lt](int xo, int yo, int zo, int xo2, int yo2, int zo2, Color lc){
                    ippl::Vector<int, 3U> v[2] = {lindex_lower, lindex_upper};
                    DrawBillboardLineEx(
                        Vector3<float>{(float)v[xo][0]  + !!xo ,(float)v[yo][1]  + !!yo, (float)v[zo][2]  + !!zo},
                        Vector3<float>{(float)v[xo2][0] + !!xo2,(float)v[yo2][1] + !!yo2,(float)v[zo2][2] + !!zo2}, lt, lc
                    );
                };
                //Draw local cube wireframe
                ld(0,0,0,1,0,0,lc);
                ld(0,0,0,0,1,0,lc);
                ld(0,0,0,0,0,1,lc);
                ld(1,0,0,1,1,0,lc);
                ld(1,0,0,1,0,1,lc);
                ld(0,1,0,1,1,0,lc);
                ld(0,1,0,0,1,1,lc);
                ld(0,0,1,0,1,1,lc);
                ld(0,0,1,1,0,1,lc);
                ld(1,1,0,1,1,1,lc);
                ld(1,0,1,1,1,1,lc);
                ld(0,1,1,1,1,1,lc);

                Kokkos::parallel_for(ippl::getRangePolicy(fieldB.getView()), KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                    ippl::Vector<scalar, 3> bv = bview(i, j, k) * (scalar(10.0) / maxnorm_ar);
                    i += lindex_lower[0];
                    j += lindex_lower[1];
                    k += lindex_lower[2];
                    DrawBillboardLineEx(Vector3<float>{(float)i,(float)j,(float)k}, Vector3<float>{(float)(i + bv[0]),(float)(j + bv[1]),(float)(k + bv[2])}, 0.01f, jet(bv.norm()));
                });
                Kokkos::parallel_for(solver.bunch.getLocalNum(), KOKKOS_LAMBDA(size_t i){
                    ippl::Vector<scalar, 3> bv = bunch_r_view(i);
                    Matrix4<float> trf(1.0);
                    trf[12] = bv[0] / hr[0] + 1.5;
                    trf[13] = bv[1] / hr[1] + 1.5;
                    trf[14] = bv[2] / hr[2] + 1.5;
                    DrawMesh(sphere_mesh, trf); 
                });

                {
                    int size = ippl::Comm->size();
                    int rank = ippl::Comm->rank();
                    for (int stride = 1; stride < size; stride *= 2) {
                        if (rank % (2 * stride) == 0) {
                            int partner = rank + stride;
                            if (partner < size) {
                                // Receive data from partner
                                framebuffer received(current_fb->resolution.x, current_fb->resolution.y);
                                MPI_Recv(
                                    received.color_buffer.data.get(), 
                                    received.color_buffer.width * received.color_buffer.height * sizeof(framebuffer::color_t),
                                    MPI_BYTE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE
                                );
                                MPI_Recv(
                                    received.depth_buffer.data.get(), 
                                    received.depth_buffer.width * received.color_buffer.height * sizeof(framebuffer::depth_t),
                                    MPI_BYTE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE
                                );
                                // Update localArray using the reduction function
                                depthblend_framebuffers(*current_fb, received);
                            }
                        } else {
                            int partner = rank - stride;
                            // Send data to partner
                            MPI_Send(
                                current_fb->color_buffer.data.get(),
                                current_fb->resolution.x * current_fb->resolution.y * sizeof(framebuffer::color_t),
                                MPI_BYTE, partner, 0, MPI_COMM_WORLD
                            );
                            MPI_Send(
                                current_fb->depth_buffer.data.get(),
                                current_fb->resolution.x * current_fb->resolution.y * sizeof(framebuffer::depth_t),
                                MPI_BYTE, partner, 1, MPI_COMM_WORLD
                            );
                            break;  // Break to avoid participating in further communication in this iteration
                        }
                    }
                    if(rank == 0){
                        std::string fn = std::to_string(it + 1);
                        while(fn.size() < 4)fn = '0' + fn;
                        outputPNG(*current_fb, "iout" + fn + ".png");

                    }
                    //dumpVTK(solver.bunch, fieldB, nr[0], nr[1], nr[2], it + 1, hr[0], hr[1], hr[2]);
                }
                matrix_stack.pop();
            }
            //if(it == iterations / 2){
            //    dumpPOV(solver.bunch, fieldB, nr[0], nr[1], nr[2], it + 1, hr[0], hr[1], hr[2]);
            //}
        }
        std::sort(energies.begin(), energies.end());
        std::cout << energies.front() << " to " << energies.back() << "\n"; 
        std::cout << "ERROR: " << std::abs(energies.back() - energies.front()) / (std::abs(energies.back()) + std::abs(energies.front())) << std::endl;
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
        
            //scalar error_accumulation = solver.volumetric_integral(KOKKOS_LAMBDA(const int i, const int j, const int k, scalar x, scalar y, scalar z){
            //    return std::abs(view_a(i, j, k)[2] - gauss(/*std::hypot(x - 0.5, y - 0.5, z - 0.5)*/ y - 0.5, 0.0, 0.1));
            //    (void)i;(void)x;
            //    (void)j;(void)y;
            //    (void)k;(void)z;
            //});
            //std::cout << "TOTAL ERROR: " << error_accumulation << std::endl;
        }
    }
    exit:
    ippl::finalize();

    return 0;
}
