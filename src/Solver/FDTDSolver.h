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

#include <ostream>
#include "Types/Vector.h"

//#include "Solver/BoundaryDispatch.h"
#include "Field/Field.h"

#include "FieldLayout/FieldLayout.h"
#include "Meshes/UniformCartesian.h"
#include "Utility/unit.hpp"

//constexpr double electron_charge = 1.0;
//constexpr double electron_mass = 1.0;
constexpr double unit_length_in_meters = 1.616255e-35;
constexpr double unit_charge_in_electron_charges = 11.71;
//constexpr double unit_charge_in_electron_charges = 3.303;
constexpr double unit_mass_in_kg = 2.176434e-8;
constexpr double kg_in_unit_masses = 1.0 / unit_mass_in_kg;
constexpr double unit_time_in_seconds = 5.391247e-44;
constexpr double meter_in_unit_lengths = 1.0 / unit_length_in_meters;
constexpr double electron_charge_in_unit_charges = 1.0 / unit_charge_in_electron_charges;
constexpr double second_in_unit_times = 1.0 / unit_time_in_seconds;

constexpr double electron_mass_in_kg = 9.1093837015e-31;
constexpr double electron_mass_in_unit_masses = electron_mass_in_kg * kg_in_unit_masses;
template <typename T>
T squaredNorm(const ippl::Vector<T, 3>& a) {
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}
template <typename T>
auto sq(T x) {
    return x * x;
}
template <typename... Args>
void castToVoid(Args&&... args) {
    (void)(std::tuple<Args...>{args...});
}
#define CAST_TO_VOID(...) castToVoid(__VA_ARGS__)
template <typename T, unsigned Dim>
T dot_prod(const ippl::Vector<T, Dim>& a, const ippl::Vector<T, Dim>& b) {
    T ret = 0.0;
    for (unsigned i = 0; i < Dim; i++) {
        ret += a[i] * b[i];
    }
    return ret;
}
template <typename T>
KOKKOS_INLINE_FUNCTION ippl::Vector<T, 3> cross_prod(const ippl::Vector<T, 3>& a,
                                                     const ippl::Vector<T, 3>& b) {
    ippl::Vector<T, 3> ret{0.0, 0.0, 0.0};
    ret[0] = a[1] * b[2] - a[2] * b[1];
    ret[1] = a[2] * b[0] - a[0] * b[2];
    ret[2] = a[0] * b[1] - a[1] * b[0];
    return ret;
}
/**
 * @brief Compiletime-sized matrix type
 * 
 * @tparam T Scalar
 * @tparam m Rows
 * @tparam n Columns
 */
template<typename T, int m, int n>
struct matrix{
    /**
     * @brief Column major
     * 
     */
    ippl::Vector<ippl::Vector<T, m>, n> data;
    constexpr static bool squareMatrix = (m == n && m > 0);

    constexpr KOKKOS_INLINE_FUNCTION matrix(T diag)
    requires(m == n){
        for(unsigned i = 0;i < n;i++){
            for(unsigned j = 0;j < n;j++){
                data[i][j] = diag * T(i == j);
            }
        }
    }
    constexpr KOKKOS_INLINE_FUNCTION matrix() = default;
    KOKKOS_INLINE_FUNCTION constexpr static matrix zero(){
        matrix<T, m, n> ret;
        for(unsigned i = 0;i < n;i++){
            for(unsigned j = 0;j < n;j++){
                ret.data[i][j] = 0;
            }
        }
        return ret;
    };

    KOKKOS_INLINE_FUNCTION T operator()(int i, int j)const noexcept{
        return data[j][i];
    }
    KOKKOS_INLINE_FUNCTION T& operator()(int i, int j)noexcept{
        return data[j][i];
    }

    KOKKOS_INLINE_FUNCTION matrix<T, m, n> operator+(const matrix<T, m, n>& other) const {
        matrix<T, m, n> result;
        for (int i = 0; i < n; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    // Implement matrix subtraction
    KOKKOS_INLINE_FUNCTION matrix<T, m, n> operator-(const matrix<T, m, n>& other) const {
        matrix<T, m, n> result;
        for (int i = 0; i < n; ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // Implement matrix-vector multiplication
    template<unsigned int other_m>
    KOKKOS_INLINE_FUNCTION ippl::Vector<T, m> operator*(const ippl::Vector<T, other_m>& vec) const {
        static_assert((int)other_m == n);
        ippl::Vector<T, m> result;
        for (int i = 0; i < n; ++i) {
            result += vec[i] * data[i];
        }
        return result;
    }
    template<int otherm, int othern>
        requires(n == otherm)
    KOKKOS_INLINE_FUNCTION matrix<T, m, othern> operator*(const matrix<T, otherm, othern>& otherMat) const noexcept {
        matrix<T, m, othern> ret(0);
        for(int i = 0;i < m;i++){
            for(int j = 0;j < othern;j++){
                for(int k = 0;k < n;k++){
                    ret(i, j) += (*this)(i, k) * otherMat(k, j);
                }
            }
        }
        return ret;
    }
    KOKKOS_INLINE_FUNCTION void addCol(int i, int j, T alpha = 1.0){
        data[j] += data[i] * alpha;
    }
    KOKKOS_INLINE_FUNCTION matrix<T, m, n> inverse()const noexcept
        requires (squareMatrix){
        constexpr int N = m;
        
        matrix<T, m, n> ret(1.0);
        matrix<T, m, n> dis(*this);

        for(int i = 0;i < N;i++){
            for(int j = i + 1;j < N;j++){
                T alpha = -dis(i, j) / dis(i, i);
                dis.addCol(i, j, alpha);
                dis(i, j) = 0;
                ret.addCol(i, j, alpha);
            }
        }
        for(int i = N - 1;i >= 0;i--){
            for(int j = i - 1;j >= 0;j--){
                T alpha = -dis(i, j) / dis(i, i);
                dis.addCol(i, j, alpha);
                dis(i, j) = 0;
                ret.addCol(i, j, alpha);
            }
        }
        for(int i = 0;i < N;i++){
            T d = dis(i, i);
            T oneod = T(1) / d;
            dis.data[i] *= oneod;
            ret.data[i] *= oneod;
        }

        return ret;
    }
    
    template<typename stream_t>
    friend stream_t& operator<<(stream_t& str, const matrix<T, m, n>& mat){
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j++){
                str << mat.data[j][i] << " ";
            }
            str << "\n";
        }
        return str;
    }
};
template<typename T, unsigned N>
ippl::Vector<T, N> vscale(const ippl::Vector<T, N>& v, T alpha){
    ippl::Vector<T, N> ret;
    for(unsigned i = 0;i < N;i++){
        ret[i] = v[i] * alpha;
    }
    return ret;
}
template<typename scalar>
KOKKOS_INLINE_FUNCTION ippl::Vector<scalar, 4> prepend_t(const ippl::Vector<scalar, 3>& x, scalar t = 0){
    return ippl::Vector<scalar, 4>{t, x[0], x[1], x[2]};
}
template<typename scalar>
KOKKOS_INLINE_FUNCTION ippl::Vector<scalar, 3> strip_t(const ippl::Vector<scalar, 4>& x){
    return ippl::Vector<scalar, 3>{x[1], x[2], x[3]};
}
template<typename T>
struct LorentzFrame{
    constexpr static T c = 1.0;
    using scalar = T;
    using Vector3 = ippl::Vector<T, 3>;
    ippl::Vector<T, 3> beta_m;
    ippl::Vector<T, 3> gammaBeta_m;
    T gamma_m;
    KOKKOS_INLINE_FUNCTION LorentzFrame(const ippl::Vector<T, 3>& gammaBeta){
        using Kokkos::sqrt;
        beta_m = gammaBeta / Kokkos::sqrt(1 + dot_prod(gammaBeta, gammaBeta));
        gamma_m = Kokkos::sqrt(1 + dot_prod(gammaBeta, gammaBeta));
        gammaBeta_m = gammaBeta;
    }
    template<char axis>
    static LorentzFrame uniaxialGamma(T gamma){
        static_assert(axis == 'x' || axis == 'y' || axis == 'z', "Only xyz axis suproted");
        assert(gamma >= 1.0 && "Gamma must be >= 1");
        using Kokkos::sqrt;
        
        T beta = sqrt(gamma * gamma - 1) / gamma;
        Vector3 arg{0,0,0};
        arg[axis - 'x'] = gamma * beta;
        return LorentzFrame<T>(arg);
    }
    KOKKOS_INLINE_FUNCTION matrix<T, 4, 4> unprimedToPrimed()const noexcept{
        T betaMagsq = dot_prod(beta_m, beta_m);
        using Kokkos::abs;
        if(abs(betaMagsq) < 1e-8){
            return matrix<T, 4, 4>(T(1));
        }
        ippl::Vector<T, 3> betaSquared = beta_m * beta_m;

        matrix<T, 4, 4> ret;

        ret.data[0] = ippl::Vector<T, 4>{ gamma_m, -gammaBeta_m[0], -gammaBeta_m[1], -gammaBeta_m[2]};
        ret.data[1] = ippl::Vector<T, 4>{-gammaBeta_m[0], 1 + (gamma_m - 1) * betaSquared[0] / betaMagsq, (gamma_m - 1) * beta_m[0] * beta_m[1] / betaMagsq, (gamma_m - 1) * beta_m[0] * beta_m[2] / betaMagsq};
        ret.data[2] = ippl::Vector<T, 4>{-gammaBeta_m[1], (gamma_m - 1) * beta_m[0] * beta_m[1] / betaMagsq, 1 + (gamma_m - 1) * betaSquared[1] / betaMagsq, (gamma_m - 1) * beta_m[1] * beta_m[2] / betaMagsq};
        ret.data[3] = ippl::Vector<T, 4>{-gammaBeta_m[2], (gamma_m - 1) * beta_m[0] * beta_m[2] / betaMagsq, (gamma_m - 1) * beta_m[1] * beta_m[2] / betaMagsq, 1 + (gamma_m - 1) * betaSquared[2] / betaMagsq};

        return ret;
    }
    KOKKOS_INLINE_FUNCTION matrix<T, 4, 4> primedToUnprimed()const noexcept{
        return unprimedToPrimed().inverse();
    }

    KOKKOS_INLINE_FUNCTION Vector3 transformV(const Vector3& unprimedV)const noexcept{
        T factor = T(1.0) / (1.0 - dot_prod(unprimedV, beta_m));
        Vector3 ret = vscale(unprimedV, 1.0 / gamma_m);
        ret -= beta_m;
        ret += vscale(beta_m, dot_prod(unprimedV, beta_m) * (gamma_m / (gamma_m + 1)));
        return vscale(ret, factor);
    }


    KOKKOS_INLINE_FUNCTION Vector3 transformGammabeta(const Vector3& gammabeta)const noexcept{
        using Kokkos::sqrt;
        T gamma = sqrt(T(1) + dot_prod(gammabeta, gammabeta));
        Vector3 beta = gammabeta;
        beta /= gamma;
        Vector3 betatrf = transformV(beta);
        betatrf *= sqrt(1 - dot_prod(betatrf, betatrf));
        return betatrf;
    }

    KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> transform_EB(const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& unprimedEB)const noexcept{
        using Kokkos::sqrt;
        Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> ret;
        Vector3 vnorm = vscale(beta_m, (1.0 / sqrt(dot_prod(beta_m, beta_m))));
        //std::cout << "FDTDSolver::289:" << dot_prod(vnorm, vnorm) << "\n";
        ret.first  = vscale(ippl::Vector<T, 3>(unprimedEB.first  + cross_prod(beta_m, unprimedEB.second)), gamma_m) - vscale(vnorm, (gamma_m - 1) * (dot_prod(unprimedEB.first,  vnorm)));
        ret.second = vscale(ippl::Vector<T, 3>(unprimedEB.second - cross_prod(beta_m, unprimedEB.first )), gamma_m) - vscale(vnorm, (gamma_m - 1) * (dot_prod(unprimedEB.second, vnorm)));
        return ret;
    }
    KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> inverse_transform_EB(const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& primedEB)const noexcept{
        ippl::Vector<T, 3> mgb(gammaBeta_m * -1.0);
        return LorentzFrame<T>(mgb).transform_EB(primedEB);
    }
    //Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> transform_inverse_EB(const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& primedEB)const noexcept{
    //    using Kokkos::sqrt;
    //    Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> ret;
    //    Vector3 vnorm = beta_m * (1.0 / sqrt(dot_prod(beta_m, beta_m)));
    //    ret.first  = (primedEB.first - cross_prod(beta_m, primedEB.second))*gamma - (gamma_m - 1) * (dot_prod(primedEB.first, vnorm) * vnorm);
    //    ret.second = (primedEB.second + cross_prod(beta_m, primedEB.first))*gamma - (gamma_m - 1) * (dot_prod(primedEB.second, vnorm) * vnorm);
    //    return ret;
    //}
};


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
    template<typename scalar>
    struct undulator_parameters{
        scalar lambda; //MITHRA: lambda_u
        scalar K; //Undulator parameter
        scalar length;
        scalar B_magnitude;
        undulator_parameters(scalar K_undulator_parameter, scalar lambda_u, scalar _length) : lambda(lambda_u), K(K_undulator_parameter), length(_length){
            B_magnitude = (2 * M_PI * electron_mass_in_unit_masses * K) / (electron_charge_in_unit_charges * lambda_u);
        }
    };

    //TODO: Maybe switch to std::function
    enum struct FDTDParticleUpdateRule{
        LORENTZ, CIRCULAR_ORBIT, DIPOLE_ORBIT, XLINE, STATIONARY
    };
    enum struct FDTDFieldUpdateRule{
        DONT, DO
    };
    enum struct trackableOutput{
        boundaryRadiation, cumulativeRadiation, p0pos
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
        FDTDSolver(Field_t& charge, VField_t& current, VField_t& E, VField_t& B, size_t pcount, LorentzFrame<Tfields> frameBoost_, undulator_parameters<Tfields> up, FDTDBoundaryCondition bcond = FDTDBoundaryCondition::PERIODIC,
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

        void setBoundaryConditions(const ippl::Vector<FDTDBoundaryCondition, Dim>& bcs);

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

        Vector<FDTDBoundaryCondition, Dim> bconds_m;
        FDTDParticleUpdateRule particle_update_m;
        FDTDFieldUpdateRule field_update_m;
        using playout_type = ippl::ParticleSpatialLayout<Tfields, 3>;
        using bunch_type = ::Bunch<Tfields, playout_type>;
        using tracer_bunch_type = ::TracerBunch<Tfields, playout_type>;

        size_t pcount_m;
        playout_type pl;
        bunch_type bunch;
        tracer_bunch_type tracer_bunch;


        double externalMagneticScale;
        double total_energy;
        double absorbed__energy;
        // buffer for communication
        detail::FieldBufferData<Tfields> fd_m;
        LorentzFrame<Tfields> frameBoost;
        undulator_parameters<Tfields> uparams;
        std::map<trackableOutput, std::ostream*> output_stream;
    };
}  // namespace ippl

#include "FDTDSolver.hpp"

#endif
