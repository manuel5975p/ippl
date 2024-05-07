#include "Ippl.h"

#include "Types/Vector.h"

#include "Field/Field.h"
#include "MaxwellSolvers/FDTD.h"
#include <chrono>
#include <cstdio> // For popen
#define JSON_HAS_RANGES 0 //Merlin compilation
#include <json.hpp>
#include <fstream>
#include <list>
#include <Kokkos_Random.hpp>
#include "Utility/Rendering.hpp"


uint64_t nanoTime(){
    using namespace std;
    using namespace chrono;
    return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
}
//CONFIG
KOKKOS_INLINE_FUNCTION bool isNaN(float x){
    #ifdef __CUDA_ARCH__
    return isnan(x);
    #else
    return std::isnan(x);
    #endif
}
KOKKOS_INLINE_FUNCTION bool isINF(float x){
    #ifdef __CUDA_ARCH__
    return isinf(x);
    #else
    return std::isinf(x);
    #endif
}
KOKKOS_INLINE_FUNCTION bool isNaN(double x){
    #ifdef __CUDA_ARCH__
    return isnan(x);
    #else
    return std::isnan(x);
    #endif
}
KOKKOS_INLINE_FUNCTION bool isINF(double x){
    #ifdef __CUDA_ARCH__
    return isinf(x);
    #else
    return std::isinf(x);
    #endif
}
template<unsigned int Dim, typename callable, typename... Ts>
KOKKOS_INLINE_FUNCTION void serial_for(callable c, ippl::Vector<uint32_t, Dim> from, ippl::Vector<uint32_t, Dim> to, Ts... args){
    if constexpr(sizeof...(Ts) == Dim){
        c(args...);
    }
    else{
        for(uint32_t i = from[sizeof...(Ts)];i < to[sizeof...(Ts)];i++){
            serial_for(c, from, to, args..., i);
        }
    }
}





struct config {
    using scalar = double;

    //using length_unit = funits::length<scalar, funits::planck_base>;
    //using duration_unit = funits::time<scalar, funits::planck_base>;
    // GRID PARAMETERS
    ippl::Vector<uint32_t, 3> resolution;

    ippl::Vector<scalar, 3> extents;
    scalar total_time;
    scalar timestep_ratio;

    scalar length_scale_in_jobfile, temporal_scale_in_jobfile;

        // All in unit_charge, or unit_mass
    scalar charge, mass;

    uint64_t num_particles;
    bool space_charge;

        // BUNCH PARAMETERS
    ippl::Vector<scalar, 3> mean_position;
    ippl::Vector<scalar, 3> sigma_position;
    ippl::Vector<scalar, 3> position_truncations;
    ippl::Vector<scalar, 3> sigma_momentum;
    scalar bunch_gamma;

    scalar undulator_K;
    scalar undulator_period;
    scalar undulator_length;

    uint32_t output_rhythm;
    std::string output_path;
    std::unordered_map<std::string, double> experiment_options;
};
template<typename scalar, unsigned Dim>
ippl::Vector<scalar, Dim> getVector(const nlohmann::json& j){
    if(j.is_array()){
        assert(j.size() == Dim);
        ippl::Vector<scalar, Dim> ret;
        for(unsigned i = 0;i < Dim;i++)
            ret[i] = (scalar)j[i];
        return ret;
    }
    else{
        std::cerr << "Warning: Obtaining Vector from scalar json\n";
        ippl::Vector<scalar, Dim> ret;
        ret.fill((scalar)j);
        return ret;
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
std::string lowercase_singular(std::string str) {
    // Convert string to lowercase
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);

    // Check if the string ends with "s" and remove it if it does
    if (!str.empty() && str.back() == 's') {
        str.pop_back();
    }

    return str;
}
double get_time_multiplier(const nlohmann::json& j){
    std::string length_scale_string = lowercase_singular((std::string)j["mesh"]["time-scale"]);
    double time_factor = 1.0;
    switch (chash(length_scale_string)) {
        case chash<"planck-time">():
        case chash<"plancktime">():
        case chash<"pt">():
        case chash<"natural">():
            time_factor = unit_time_in_seconds;
        break;
        case chash<"picosecond">():
            time_factor = 1e-12;
        break;
        case chash<"nanosecond">():
            time_factor = 1e-9;
        break;
        case chash<"microsecond">():
            time_factor = 1e-6;
        break;
        case chash<"millisecond">():
            time_factor = 1e-3;
        break;
        case chash<"second">():
            time_factor = 1.0;
        break;
        default:
            std::cerr << "Unrecognized time scale: " << (std::string)j["mesh"]["time-scale"] << "\n";
        break;
    }
    return time_factor;
}
double get_length_multiplier(const nlohmann::json& options){
    std::string length_scale_string = lowercase_singular((std::string)options["mesh"]["length-scale"]);
    double length_factor = 1.0;
    switch (chash(length_scale_string)) {
        case chash<"planck-length">():
        case chash<"plancklength">():
        case chash<"pl">():
        case chash<"natural">():
            length_factor = unit_length_in_meters;
        break;
        case chash<"picometer">():
            length_factor = 1e-12;
        break;
        case chash<"nanometer">():
            length_factor = 1e-9;
        break;
        case chash<"micrometer">():
            length_factor = 1e-6;
        break;
        case chash<"millimeter">():
            length_factor = 1e-3;
        break;
        case chash<"meter">():
            length_factor = 1.0;
        break;
        default:
            std::cerr << "Unrecognized length scale: " << (std::string)options["mesh"]["length-scale"] << "\n";
        break;
    }
    return length_factor;
}
config read_config(const char *filepath){
    std::ifstream cfile(filepath);
    nlohmann::json j;
    cfile >> j;
    config::scalar lmult = get_length_multiplier(j);
    config::scalar tmult = get_time_multiplier(j);
    config ret;

    ret.extents[0] = ((config::scalar)j["mesh"]["extents"][0] * lmult) / unit_length_in_meters;
    ret.extents[1] = ((config::scalar)j["mesh"]["extents"][1] * lmult) / unit_length_in_meters;
    ret.extents[2] = ((config::scalar)j["mesh"]["extents"][2] * lmult) / unit_length_in_meters;
    ret.resolution = getVector<uint32_t, 3>(j["mesh"]["resolution"]);

    //std::cerr << (std::string)j["mesh"]["time-scale"] << " " << tmult << " Tumult\n";
    //std::cerr << "Tmult: " << tmult << "\n";
    if(j.contains("timestep-ratio")){
        ret.timestep_ratio = (config::scalar)j["timestep-ratio"];
    }

    else{
        ret.timestep_ratio = 1;
    }
    ret.total_time = ((config::scalar)j["mesh"]["total-time"] * tmult) / unit_time_in_seconds;
    ret.space_charge = (bool)(j["mesh"]["space-charge"]);
    ret.bunch_gamma = (config::scalar)(j["bunch"]["gamma"]);
    if(ret.bunch_gamma < config::scalar(1)){
        std::cerr << "Gamma must be >= 1\n";
        exit(1);
    }
    assert(j.contains("undulator"));
    assert(j["undulator"].contains("static-undulator"));

    ret.undulator_K = j["undulator"]["static-undulator"]["undulator-parameter"];
    ret.undulator_period = ((config::scalar)j["undulator"]["static-undulator"]["period"] * lmult) / unit_length_in_meters;
    ret.undulator_length = ((config::scalar)j["undulator"]["static-undulator"]["length"] * lmult) / unit_length_in_meters;
    assert(!std::isnan(ret.undulator_length));
    assert(!std::isnan(ret.undulator_period));
    assert(!std::isnan(ret.extents[0]));
    assert(!std::isnan(ret.extents[1]));
    assert(!std::isnan(ret.extents[2]));
    assert(!std::isnan(ret.total_time));
    ret.length_scale_in_jobfile = get_length_multiplier(j);
    ret.temporal_scale_in_jobfile = get_time_multiplier(j);
    ret.charge = (config::scalar)j["bunch"]["charge"] * electron_charge_in_unit_charges;
    ret.mass = (config::scalar)j["bunch"]["mass"] * electron_mass_in_unit_masses;
    ret.num_particles = (uint64_t)j["bunch"]["number-of-particles"];
    ret.mean_position  = getVector<config::scalar, 3>(j["bunch"]["position"])                       * lmult / unit_length_in_meters;
    ret.sigma_position = getVector<config::scalar, 3>(j["bunch"]["sigma-position"])                 * lmult / unit_length_in_meters;
    ret.position_truncations = getVector<config::scalar, 3>(j["bunch"]["distribution-truncations"]) * lmult / unit_length_in_meters;
    ret.sigma_momentum = getVector<config::scalar, 3>(j["bunch"]["sigma-momentum"]);
    ret.output_rhythm = j["output"].contains("rhythm") ? uint32_t(j["output"]["rhythm"]) : 0;
    ret.output_path = "../data/";
    if(j["output"].contains("path")){
        ret.output_path = j["output"]["path"];
        if(!ret.output_path.ends_with('/')){
            ret.output_path.push_back('/');
        }
    }
    if(j.contains("experimentation")){
        nlohmann::json je = j["experimentation"];
        for(auto it = je.begin(); it!= je.end();it++){
            ret.experiment_options[it.key()] = double(it.value());
        }
    }
    return ret;
}
template<typename scalar>
using FieldVector = ippl::Vector<scalar, 3>;
template<typename scalar>
struct BunchInitialize {

    /* Type of the bunch which is one of the manual, ellipsoid, cylinder, cube, and 3D-crystal. If it is
     * manual the charge at points of the position vector will be produced.    				*/
    // std::string     			bunchType_;

    /* Type of the distributions (transverse or longitudinal) in the bunch.				*/
    std::string distribution_;

    /* Type of the generator for creating the bunch distribution.					*/
    std::string generator_;

    /* Total number of macroparticles in the bunch.                                                     */
    unsigned int numberOfParticles_;

    /* Total charge of the bunch in pC.                                                                 */
    scalar cloudCharge_;

    /* Initial energy of the bunch in MeV.                                                              */
    scalar initialGamma_;

    /* Initial normalized speed of the bunch.                                                           */
    scalar initialBeta_;

    /* Initial movement direction of the bunch, which is a unit vector.                                 */
    FieldVector<scalar> initialDirection_;

    /* Position of the center of the bunch in the unit of length scale.                           	*/
    // std::vector<FieldVector<scalar> >	position_;
    FieldVector<scalar> position_;

    /* Number of macroparticles in each direction for 3Dcrystal type.                                   */
    FieldVector<unsigned int> numbers_;

    /* Lattice constant in x, y, and z directions for 3D crystal type.                                  */
    FieldVector<scalar> latticeConstants_;

    /* Spread in position for each of the directions in the unit of length scale. For the 3D crystal
     * type, it will be the spread in position for each micro-bunch of the crystal.			*/
    FieldVector<scalar> sigmaPosition_;

    /* Spread in energy in each direction.                                                              */
    FieldVector<scalar> sigmaGammaBeta_;

    /* Store the truncation transverse distance for the electron generation.				*/
    scalar tranTrun_;

    /* Store the truncation longitudinal distance for the electron generation.				*/
    scalar longTrun_;

    /* Name of the file for reading the electrons distribution from.					*/
    std::string fileName_;

    /* The radiation wavelength corresponding to the bunch length outside the undulator			*/
    scalar lambda_;

    /* Bunching factor for the initialization of the bunch.						*/
    scalar bF_;

    /* Phase of the bunching factor for the initialization of the bunch.				*/
    scalar bFP_;

    /* Boolean flag determining the activation of shot-noise.						*/
    bool shotNoise_;

    /* Initial beta vector of the bunch, which is obtained as the product of beta and direction.	*/
    FieldVector<scalar> betaVector_;

    /* Initialize the parameters for the bunch initialization to some first values.                     */
    // BunchInitialize ();
};





//END CONFIG

//LORENTZ FRAME AND UNDULATOR
template<typename T, unsigned axis = 2>
struct UniaxialLorentzframe{
    constexpr static T c = 1.0;
    using scalar = T;
    using Vector3 = ippl::Vector<T, 3>;
    scalar beta_m;
    scalar gammaBeta_m;
    scalar gamma_m;
    KOKKOS_INLINE_FUNCTION static UniaxialLorentzframe from_gamma(const scalar gamma){

        UniaxialLorentzframe ret;
        ret.gamma_m = gamma;
        scalar beta = Kokkos::sqrt(1 - double(1) / (gamma * gamma));
        scalar gammabeta = gamma * beta;
        ret.beta_m = beta;
        ret.gammaBeta_m = gammabeta;
        return ret;
    }
    KOKKOS_INLINE_FUNCTION UniaxialLorentzframe<T, axis> negative()const noexcept{
        UniaxialLorentzframe ret;
        ret.beta_m = -beta_m;
        ret.gammaBeta_m = -gammaBeta_m;
        ret.gamma_m = gamma_m;
        return ret;
    }

    KOKKOS_INLINE_FUNCTION UniaxialLorentzframe() = default;
    KOKKOS_INLINE_FUNCTION UniaxialLorentzframe(const scalar gammaBeta){
        using Kokkos::sqrt;
        gammaBeta_m = gammaBeta;
        beta_m = gammaBeta / sqrt(1 + gammaBeta * gammaBeta);
        gamma_m = sqrt(1 + gammaBeta * gammaBeta);
    }
    KOKKOS_INLINE_FUNCTION void primedToUnprimed(Vector3& arg, scalar time)const noexcept{
        arg[axis] = gamma_m * (arg[axis] + beta_m * time); 
    }
    KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> transform_EB(const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& unprimedEB)const noexcept{
        
        Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> ret;
        ippl::Vector<scalar, 3> betavec{0, 0, beta_m};
        ret.first  = ippl::Vector<T, 3>(unprimedEB.first  + betavec.cross(unprimedEB.second)) * gamma_m;// - (vnorm * (gamma_m - 1) * (unprimedEB.first.dot(vnorm)));
        ret.second = ippl::Vector<T, 3>(unprimedEB.second - betavec.cross(unprimedEB.first )) * gamma_m;// - (vnorm * (gamma_m - 1) * (unprimedEB.second.dot(vnorm)));
        ret.first[axis] -= (gamma_m - 1) * unprimedEB.first[axis];
        ret.second[axis] -= (gamma_m - 1) * unprimedEB.second[axis];
        return ret;
    }
    KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>> inverse_transform_EB(const Kokkos::pair<ippl::Vector<T, 3>, ippl::Vector<T, 3>>& primedEB)const noexcept{
        return negative().transform_EB(primedEB);
    }
};
template<typename scalar>
BunchInitialize<scalar> generate_mithra_config(const config& cfg, const UniaxialLorentzframe<scalar>& /*frame_boost unused*/) {
    using vec3 = ippl::Vector<scalar, 3>;
    scalar frame_gamma = cfg.bunch_gamma / std::sqrt(1 + 0.5 * cfg.undulator_K * cfg.undulator_K);
    BunchInitialize<scalar> init;
    init.generator_ = "random";
    init.distribution_ = "uniform";
    init.initialDirection_ = vec3{0, 0, 1};
    init.initialGamma_ = cfg.bunch_gamma;
    init.initialBeta_ = cfg.bunch_gamma == scalar(1) ? 0 : (sqrt(cfg.bunch_gamma * cfg.bunch_gamma - 1) / cfg.bunch_gamma);
    init.sigmaGammaBeta_ = cfg.sigma_momentum.template cast<scalar>();
    init.sigmaPosition_ = cfg.sigma_position.template cast<scalar>();

    // TODO: Initial bunching factor huh
    init.bF_ = 0.01;
    init.bFP_ = 0;
    init.shotNoise_ = false;
    init.cloudCharge_ = cfg.charge;
    init.lambda_ = cfg.undulator_period / (2 * frame_gamma * frame_gamma);
    init.longTrun_ = cfg.position_truncations[2];
    init.tranTrun_ = cfg.position_truncations[0];
    init.position_ = cfg.mean_position.cast<scalar>();
    init.betaVector_ = ippl::Vector<scalar, 3>{0,0,init.initialBeta_};
    init.numberOfParticles_ = cfg.num_particles;

    init.numbers_ = 0;                      // UNUSED
    init.latticeConstants_ = vec3{0, 0, 0}; // UNUSED

    return init;
}
template <typename Double>
struct Charge {

    Double q;                     /* Charge of the point in the unit of electron charge.	*/
    FieldVector<Double> rnp, rnm; /* Position vector of the charge.			*/
    FieldVector<Double> gb;       /* Normalized velocity vector of the charge.		*/

    /* Double flag determining if the particle is passing the entrance point of the undulator. This flag
     * can be used for better boosting the bunch to the moving frame. We need to consider it to be double,
     * because this flag needs to be communicated during bunch update.					*/
    Double e;

    // Charge();
};
template <typename scalar>
using ChargeVector = std::list<Charge<scalar>>;
template<typename Double>
void initializeBunchEllipsoid (BunchInitialize<Double> bunchInit, ChargeVector<Double> & chargeVector, int rank, int size, int ia){
    /* Correct the number of particles if it is not a multiple of four.					*/
    if ( bunchInit.numberOfParticles_ % 4 != 0 )
      {
        unsigned int n = bunchInit.numberOfParticles_ % 4;
        bunchInit.numberOfParticles_ += 4 - n;
        //printmessage(std::string(__FILE__), __LINE__, std::string("Warning: The number of particles in the bunch is not a multiple of four. ") +
        //    std::string("It is corrected to ") +  std::to_string(bunchInit.numberOfParticles_) );
    }

    /* Save the initially given number of particles.							*/
    unsigned int	Np = bunchInit.numberOfParticles_, i, Np0 = chargeVector.size();

    /* Declare the required parameters for the initialization of charge vectors.                      	*/
    Charge<Double>         	charge; charge.q  = bunchInit.cloudCharge_ / Np;
    FieldVector<Double> gb = bunchInit.initialGamma_ * bunchInit.betaVector_;
    FieldVector<Double> r  (0.0);
    FieldVector<Double> t  (0.0);
    Double            	t0;//, g;
    Double		zmin = 1e100;
    Double		Ne, bF, bFi;
    unsigned int	bmi;
    std::vector<Double>	randomNumbers;

    /* The initialization in group of four particles should only be done if there exists an undulator in
     * the interaction.											*/
    unsigned int	ng = ( bunchInit.lambda_ == 0.0 ) ? 1 : 4;

    /* Check the bunching factor.                                                                     	*/
    if ( bunchInit.bF_ > 2.0 || bunchInit.bF_ < 0.0 )
      {
        //printmessage(std::string(__FILE__), __LINE__, std::string("The bunching factor can not be larger than one or a negative value !!!") );
        //exit(1);
      }

    /* If the generator is random we should make sure that different processors do not produce the same
     * random numbers.											*/
    if 	( bunchInit.generator_ == "random" )
      {
        /* Initialize the random number generator.								*/
        srand ( time(NULL) );
        /* Np / ng * 20 is the maximum number of particles.							*/
        randomNumbers.resize( Np / ng * 20, 0.0);
        for ( unsigned int ri = 0; ri < Np / ng * 20; ri++)
          randomNumbers[ri] = (float)std::min(1 - 1e-7, std::max(1e-7, ((double) rand() ) / RAND_MAX));
      }

    /* Declare the generator function depending on the input.						*/
    auto generate = [&] (unsigned int n, unsigned int m) {
      //if 	( bunchInit.generator_ == "random" )
        return  ( randomNumbers.at( n * 2 * Np/ng + m ) );
      //else
      //  return  ( randomNumbers[ n * 2 * Np/ng + m ] );
    //TODO: Return halton properly
        //return 	( halton(n,m) );
    };

    /* Declare the function for injecting the shot noise.						*/
    auto insertCharge = [&] (Charge<Double> q) {

      for ( unsigned int ii = 0; ii < ng; ii++ )
        {
          /* The random modulation is introduced depending on the shot-noise being activated.		*/
          if ( bunchInit.shotNoise_ )
            {
              /* Obtain the number of beamlet.								*/
              bmi = int( ( charge.rnp[2] - zmin ) / bunchInit.lambda_ );

              /* Obtain the phase and amplitude of the modulation.					*/
              bFi = bF * sqrt( - 2.0 * log( generate( 8 , bmi ) ) );

              q.rnp[2]  = charge.rnp[2] - bunchInit.lambda_ / 4 * ii;

              q.rnp[2] -= bunchInit.lambda_ / M_PI * bFi * sin( 2.0 * M_PI / bunchInit.lambda_ * q.rnp[2] + 2.0 * M_PI * generate( 9 , bmi ) );
            }
          else if ( bunchInit.lambda_ != 0.0)
            {
              q.rnp[2]  = charge.rnp[2] - bunchInit.lambda_ / 4 * ii;

              q.rnp[2] -= bunchInit.lambda_ / M_PI * bunchInit.bF_ * sin( 2.0 * M_PI / bunchInit.lambda_ * q.rnp[2] + bunchInit.bFP_ * M_PI / 180.0 );
            }

          /* Set this charge into the charge vector.							*/
          chargeVector.push_back(q);
        }
    };

    /* If the shot noise is on, we need the minimum value of the bunch z coordinate to be able to
     * calculate the FEL bucket number.									*/
    if ( bunchInit.shotNoise_ )
      {
        for (i = 0; i < Np / ng; i++)
          {
            if ( bunchInit.distribution_ == "uniform" )
              zmin = std::min(   Double( 2.0 * generate(2, i + Np0) - 1.0 ) * bunchInit.sigmaPosition_[2] , zmin );
            else if ( bunchInit.distribution_ == "gaussian" )
              zmin = std::min(  (Double) (bunchInit.sigmaPosition_[2] * sqrt( - 2.0 * log( generate(2, i + Np0) ) ) * sin( 2.0 * M_PI * generate(3, i + Np0) ) ), zmin );
            else
              {
                //printmessage(std::string(__FILE__), __LINE__, std::string("The longitudinal type is not correctly given to the code !!!") );
                exit(1);
              }
          }

        if ( bunchInit.distribution_ == "uniform" )
          for ( ; i < unsigned( Np / ng * ( 1.0 + 2.0 * bunchInit.lambda_ * sqrt( 2.0 * M_PI ) / ( 2.0 * bunchInit.sigmaPosition_[2] ) ) ); i++)
            {
              t0  = 2.0 * bunchInit.lambda_ * sqrt( - 2.0 * log( generate( 2, i + Np0 ) ) ) * sin( 2.0 * M_PI * generate( 3, i + Np0 ) );
              t0 += ( t0 < 0.0 ) ? ( - bunchInit.sigmaPosition_[2] ) : ( bunchInit.sigmaPosition_[2] );

              zmin = std::min(   t0 , zmin );
            }

        //zmin = zmin + bunchInit.position_[ia][2];
        zmin = zmin + bunchInit.position_[2];

        /* Obtain the average number of electrons per FEL beamlet.					*/
        Ne = bunchInit.cloudCharge_ * bunchInit.lambda_ / ( 2.0 * bunchInit.sigmaPosition_[2] );

        /* Set the bunching factor level for the shot noise depending on the given values.		*/
        bF = ( bunchInit.bF_ == 0.0 ) ? 1.0 / sqrt(Ne) : bunchInit.bF_;

        //printmessage(std::string(__FILE__), __LINE__, std::string("The standard deviation of the bunching factor for the shot noise implementation is set to ") + stringify(bF) );
      }

    /* Determine the properties of each charge point and add them to the charge vector.               	*/
    for (i = rank; i < Np / ng; i += size)
      {
        /* Determine the transverse coordinate.								*/
        r[0] = bunchInit.sigmaPosition_[0] * sqrt( - 2.0 * log( generate(0, i + Np0) ) ) * cos( 2.0 * M_PI * generate(1, i + Np0) );
        r[1] = bunchInit.sigmaPosition_[1] * sqrt( - 2.0 * log( generate(0, i + Np0) ) ) * sin( 2.0 * M_PI * generate(1, i + Np0) );

        /* Determine the longitudinal coordinate.							*/
        if ( bunchInit.distribution_ == "uniform" )
          r[2] = ( 2.0 * generate(2, i + Np0) - 1.0 ) * bunchInit.sigmaPosition_[2];
        else if ( bunchInit.distribution_ == "gaussian" )
          r[2] = bunchInit.sigmaPosition_[2] * sqrt( - 2.0 * log( generate(2, i + Np0) ) ) * sin( 2.0 * M_PI * generate(3, i + Np0) );
        else
          {
            //printmessage(std::string(__FILE__), __LINE__, std::string("The longitudinal type is not correctly given to the code !!!") );
            exit(1);
          }

        /* Determine the transverse momentum.								*/
        t[0] = bunchInit.sigmaGammaBeta_[0] * sqrt( - 2.0 * log( generate(4, i + Np0) ) ) * cos( 2.0 * M_PI * generate(5, i + Np0) );
        t[1] = bunchInit.sigmaGammaBeta_[1] * sqrt( - 2.0 * log( generate(4, i + Np0) ) ) * sin( 2.0 * M_PI * generate(5, i + Np0) );
        t[2] = bunchInit.sigmaGammaBeta_[2] * sqrt( - 2.0 * log( generate(6, i + Np0) ) ) * cos( 2.0 * M_PI * generate(7, i + Np0) );

        if ( fabs(r[0]) < bunchInit.tranTrun_ && fabs(r[1]) < bunchInit.tranTrun_ && fabs(r[2]) < bunchInit.longTrun_)
          {
            /* Shift the generated charge to the center position and momentum space.			*/
            //charge.rnp    = bunchInit.position_[ia];
            charge.rnp    = bunchInit.position_;
            charge.rnp   += r;

            charge.gb   = gb;
            charge.gb  += t;
            //std::cout << gb << "\n";
            if(std::isinf(gb[2])){
                std::cerr << "it klonked here\n";
            }

            /* Insert this charge and the mirrored ones into the charge vector.				*/
            insertCharge(charge);
          }
      }

    /* If the longitudinal type of the bunch is uniform a tapered part needs to be added to remove the
     * CSE from the tail of the bunch.									*/
    if ( bunchInit.distribution_ == "uniform" ){
      for ( ; i < unsigned( uint32_t(Np / ng) * ( 1.0 + 2.0 * bunchInit.lambda_ * sqrt( 2.0 * M_PI ) / ( 2.0 * bunchInit.sigmaPosition_[2] ) ) ); i += size)
        {
            
          r[0] = bunchInit.sigmaPosition_[0] * sqrt( - 2.0 * log( generate(0, i + Np0) ) ) * cos( 2.0 * M_PI * generate(1, i + Np0) );
          r[1] = bunchInit.sigmaPosition_[1] * sqrt( - 2.0 * log( generate(0, i + Np0) ) ) * sin( 2.0 * M_PI * generate(1, i + Np0) );

          /* Determine the longitudinal coordinate.							*/
          r[2] = 2.0 * bunchInit.lambda_ * sqrt( - 2.0 * log( generate(2, i + Np0) ) ) * sin( 2.0 * M_PI * generate(3, i + Np0) );
          r[2] += ( r[2] < 0.0 ) ? ( - bunchInit.sigmaPosition_[2] ) : ( bunchInit.sigmaPosition_[2] );

          /* Determine the transverse momentum.								*/
          t[0] = bunchInit.sigmaGammaBeta_[0] * sqrt( - 2.0 * log( generate(4, i + Np0) ) ) * cos( 2.0 * M_PI * generate(5, i + Np0) );
          t[1] = bunchInit.sigmaGammaBeta_[1] * sqrt( - 2.0 * log( generate(4, i + Np0) ) ) * sin( 2.0 * M_PI * generate(5, i + Np0) );
          t[2] = bunchInit.sigmaGammaBeta_[2] * sqrt( - 2.0 * log( generate(6, i + Np0) ) ) * cos( 2.0 * M_PI * generate(7, i + Np0) );
          //std::cerr << "DOING UNIFORM tapering!!!\n";
          if ( fabs(r[0]) < bunchInit.tranTrun_ && fabs(r[1]) < bunchInit.tranTrun_ && fabs(r[2]) < bunchInit.longTrun_)
            {
                //std::cerr << "ACTUALLY DOING UNIFORM tapering!!!\n";
              /* Shift the generated charge to the center position and momentum space.			*/
              charge.rnp   = bunchInit.position_[ia];
              charge.rnp  += r;

              charge.gb  = gb;
              
              charge.gb += t;
              //std::cout << gb[0] << "\n";
              //if(std::isinf(gb.squaredNorm())){
              //    std::cerr << "it klonked here\n";
              //}
              /* Insert this charge and the mirrored ones into the charge vector.			*/
              insertCharge(charge);
            }
        }
    }

    /* Reset the value for the number of particle variable according to the installed number of
     * macro-particles and perform the corresponding changes.                                         	*/
    bunchInit.numberOfParticles_ = chargeVector.size();
}

template<typename Double>
void boost_bunch(ChargeVector<Double>& chargeVectorn_, Double frame_gamma){
    Double frame_beta = std::sqrt((double)frame_gamma * frame_gamma - 1.0) / double(frame_gamma);
    Double zmaxL = -1.0e100, zmaxG;
    for (auto iterQ = chargeVectorn_.begin(); iterQ != chargeVectorn_.end(); iterQ++ )
      {
        Double g  	= std::sqrt(1.0 + iterQ->gb.squaredNorm());
        if(std::isinf(g)){
            std::cerr << __FILE__  << ": " << __LINE__ << " inf gb: " << iterQ->gb << ", g = " << g << "\n";
            abort();
        }
        Double bz 	= iterQ->gb[2] / g;
        iterQ->rnp[2]  *= frame_gamma;
        
        iterQ->gb[2] 	= frame_gamma * g * ( bz - frame_beta );
        
        zmaxL 		= std::max( zmaxL , iterQ->rnp[2] );
      }
    zmaxG = zmaxL;
    struct {
       Double zu_;
       Double beta_;
    } bunch_;
    bunch_.zu_ 		= zmaxG;
    bunch_.beta_ 	= frame_beta;

    /****************************************************************************************************/

    for (auto iterQ = chargeVectorn_.begin(); iterQ != chargeVectorn_.end(); iterQ++ )
      {
        Double g	= std::sqrt(1.0 + iterQ->gb.squaredNorm());
        iterQ->rnp[0]  += iterQ->gb[0] / g * ( iterQ->rnp[2] - bunch_.zu_ ) * frame_beta;
        iterQ->rnp[1]  += iterQ->gb[1] / g * ( iterQ->rnp[2] - bunch_.zu_ ) * frame_beta;
        iterQ->rnp[2]  += iterQ->gb[2] / g * ( iterQ->rnp[2] - bunch_.zu_ ) * frame_beta;
        if(std::isnan(iterQ->rnp[2])){
            std::cerr << iterQ->gb[2] << ", " << g << ", " << iterQ->rnp[2] << ", " << bunch_.zu_  << ", " <<  frame_beta << "\n";
            std::cerr << __FILE__  << ": " << __LINE__ << "   OOOOOF\n";
            abort();
        }
      }
}




template<typename bunch_type, typename scalar>
size_t initialize_bunch_mithra(
    bunch_type& bunch,
    const BunchInitialize<scalar>& bunchInit,
    scalar frame_gamma){

    ChargeVector<scalar> oof;
    initializeBunchEllipsoid(bunchInit, oof, 0, 1, 0);
    for(auto& c : oof){
        if(std::isnan(c.rnp[0]) || std::isnan(c.rnp[1]) || std::isnan(c.rnp[2]))
            std::cout << "Pos before boost: " << c.rnp << "\n";
        if(std::isinf(c.rnp[0]) || std::isinf(c.rnp[1]) || std::isinf(c.rnp[2]))
            std::cout << "Pos before boost: " << c.rnp << "\n";
    }
    boost_bunch(oof, frame_gamma);
    for(auto& c : oof){
         if(std::isnan(c.rnp[0]) || std::isnan(c.rnp[1]) || std::isnan(c.rnp[2])){
            std::cout << "Pos after boost: " << c.rnp << "\n";
            break;
        }
    }
    Kokkos::View<ippl::Vector<scalar, 3>*, Kokkos::HostSpace> positions("", oof.size());
    Kokkos::View<ippl::Vector<scalar, 3>*, Kokkos::HostSpace> gammabetas("", oof.size());
    auto iterQ = oof.begin();
    for (size_t i = 0; i < oof.size(); i++) {
        assert_isreal(iterQ->gb[0]);
        assert_isreal(iterQ->gb[1]);
        assert_isreal(iterQ->gb[2]);
        assert(iterQ->gb[2] != 0.0f);
        scalar g = std::sqrt(1.0 + iterQ->gb.squaredNorm());
        assert_isreal(g);
        scalar bz = iterQ->gb[2] / g;
        assert_isreal(bz);
        (void)bz;
        positions(i) = iterQ->rnp;
        gammabetas(i) = iterQ->gb;
        ++iterQ;
    }
    if(oof.size() > bunch.getLocalNum()){
        bunch.create(oof.size() - bunch.getLocalNum());
    }
    Kokkos::View<ippl::Vector<scalar, 3>*> dpositions ("", oof.size());
    Kokkos::View<ippl::Vector<scalar, 3>*> dgammabetas("", oof.size());
    
    Kokkos::deep_copy(dpositions, positions);
    Kokkos::deep_copy(dgammabetas, gammabetas);
    Kokkos::deep_copy(bunch.R_nm1.getView(), positions);
    Kokkos::deep_copy(bunch.gamma_beta.getView(), gammabetas);
    auto rview = bunch.R.getView(), rm1view = bunch.R_nm1.getView(), gbview = bunch.gamma_beta.getView();;
    Kokkos::parallel_for(oof.size(), KOKKOS_LAMBDA(size_t i){
        rview(i) = dpositions(i);
        rm1view(i) = dpositions(i);
        gbview(i) = dgammabetas(i);
    });
    Kokkos::fence();
    
    return oof.size();
}







//END LORENTZ FRAME AND UNDULATOR AND BUNCH





//PREAMBLE

template<typename scalar1, typename... scalar>
    requires((std::is_floating_point_v<scalar1>))
KOKKOS_INLINE_FUNCTION float gauss(scalar1 mean, scalar1 stddev, scalar... x){
    uint32_t dim = sizeof...(scalar);
    ippl::Vector<scalar1, sizeof...(scalar)> vec{scalar1(x - mean)...};
    for(unsigned d = 0;d < dim;d++){
        vec[d] = vec[d] * vec[d];
    }
    #ifndef __CUDA_ARCH__
    using std::exp;
    #endif
    return exp(-(vec.sum()) / (stddev * stddev)); 
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
template<typename T>
KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<T, 3>> gridCoordinatesOf(const ippl::Vector<T, 3> hr, const ippl::Vector<T, 3> origin, ippl::Vector<T, 3> pos){
    //return pear<ippl::Vector<int, 3>, ippl::Vector<T, 3>>{ippl::Vector<int, 3>{5,5,5}, ippl::Vector<T, 3>{0,0,0}};
    //printf("%.10e, %.10e, %.10e\n", (inverse_spacing * spacing)[0], (inverse_spacing * spacing)[1], (inverse_spacing * spacing)[2]);
    Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<T, 3>> ret;
    //pos -= spacing * T(0.5);
    ippl::Vector<T, 3> relpos = pos - origin;
    ippl::Vector<T, 3> gridpos = relpos / hr;
    ippl::Vector<int, 3> ipos;
    ipos = gridpos.template cast<int>();
    ippl::Vector<T, 3> fracpos;
    for(unsigned k = 0;k < 3;k++){
        fracpos[k] = gridpos[k] - (int)ipos[k];
    }
    //TODO: NGHOST!!!!!!!
    ipos += ippl::Vector<int, 3>(1);
    ret.first = ipos;
    ret.second = fracpos;
    return ret;
}
template<typename view_type, typename scalar>
KOKKOS_FUNCTION void scatterToGrid(const ippl::NDIndex<3>& ldom, view_type& view, ippl::Vector<scalar, 3> hr, ippl::Vector<scalar, 3> orig, const ippl::Vector<scalar, 3>& pos, const scalar value){
    auto [ipos, fracpos] = gridCoordinatesOf(hr, orig, pos);
    ipos -= ldom.first();
    //std::cout << pos << " 's scatter args (will have 1 added): " << ipos << "\n";
    if(
        ipos[0] < 0
        ||ipos[1] < 0
        ||ipos[2] < 0
        ||size_t(ipos[0]) >= view.extent(0) - 1
        ||size_t(ipos[1]) >= view.extent(1) - 1
        ||size_t(ipos[2]) >= view.extent(2) - 1
        ||fracpos[0] < 0
        ||fracpos[1] < 0
        ||fracpos[2] < 0
    ){
        return;
    }
    assert(fracpos[0] >= 0.0f);
    assert(fracpos[0] <= 1.0f);
    assert(fracpos[1] >= 0.0f);
    assert(fracpos[1] <= 1.0f);
    assert(fracpos[2] >= 0.0f);
    assert(fracpos[2] <= 1.0f);
    ippl::Vector<scalar, 3> one_minus_fracpos = ippl::Vector<scalar, 3>(1) - fracpos;
    assert(one_minus_fracpos[0] >= 0.0f);
    assert(one_minus_fracpos[0] <= 1.0f);
    assert(one_minus_fracpos[1] >= 0.0f);
    assert(one_minus_fracpos[1] <= 1.0f);
    assert(one_minus_fracpos[2] >= 0.0f);
    assert(one_minus_fracpos[2] <= 1.0f);
    scalar accum = 0;
    
    for(unsigned i = 0;i < 8;i++){
        scalar weight = 1;
        ippl::Vector<int, 3> ipos_l = ipos;
        for(unsigned d = 0;d < 3;d++){
            weight *= ((i & (1 << d)) ? fracpos[d] : one_minus_fracpos[d]);
            ipos_l[d] += !!(i & (1 << d));
        }
        assert_isreal(value);
        assert_isreal(weight);
        accum += weight;
        Kokkos::atomic_add(&(view(ipos_l[0], ipos_l[1], ipos_l[2])[0]), value * weight);
    }
    assert(abs(accum - 1.0f) < 1e-6f);
}
template<typename view_type, typename scalar>
KOKKOS_FUNCTION void scatterToGrid(const ippl::NDIndex<3>& ldom, view_type& view, ippl::Vector<scalar, 3> hr, ippl::Vector<scalar, 3> orig, const ippl::Vector<scalar, 3>& pos, const ippl::Vector<scalar, 3>& value){
    //std::cout << "Value: " << value << "\n";
    auto [ipos, fracpos] = gridCoordinatesOf(hr, orig, pos);
    ipos -= ldom.first();
    if(
        ipos[0] < 0
        ||ipos[1] < 0
        ||ipos[2] < 0
        ||size_t(ipos[0]) >= view.extent(0) - 1
        ||size_t(ipos[1]) >= view.extent(1) - 1
        ||size_t(ipos[2]) >= view.extent(2) - 1
        ||fracpos[0] < 0
        ||fracpos[1] < 0
        ||fracpos[2] < 0
    ){
        //std::cout << "out of bounds\n";
        return;
    }
    assert(fracpos[0] >= 0.0f);
    assert(fracpos[0] <= 1.0f);
    assert(fracpos[1] >= 0.0f);
    assert(fracpos[1] <= 1.0f);
    assert(fracpos[2] >= 0.0f);
    assert(fracpos[2] <= 1.0f);
    ippl::Vector<scalar, 3> one_minus_fracpos = ippl::Vector<scalar, 3>(1) - fracpos;
    assert(one_minus_fracpos[0] >= 0.0f);
    assert(one_minus_fracpos[0] <= 1.0f);
    assert(one_minus_fracpos[1] >= 0.0f);
    assert(one_minus_fracpos[1] <= 1.0f);
    assert(one_minus_fracpos[2] >= 0.0f);
    assert(one_minus_fracpos[2] <= 1.0f);
    scalar accum = 0;
    
    for(unsigned i = 0;i < 8;i++){
        scalar weight = 1;
        ippl::Vector<int, 3> ipos_l = ipos;
        for(unsigned d = 0;d < 3;d++){
            weight *= ((i & (1 << d)) ? fracpos[d] : one_minus_fracpos[d]);
            ipos_l[d] += !!(i & (1 << d));
        }
        assert_isreal(weight);
        accum += weight;
        Kokkos::atomic_add(&(view(ipos_l[0], ipos_l[1], ipos_l[2])[1]), value[0] * weight);
        Kokkos::atomic_add(&(view(ipos_l[0], ipos_l[1], ipos_l[2])[2]), value[1] * weight);
        Kokkos::atomic_add(&(view(ipos_l[0], ipos_l[1], ipos_l[2])[3]), value[2] * weight);
    }
    assert(abs(accum - 1.0f) < 1e-6f);
}

template<typename view_type, typename scalar>
KOKKOS_INLINE_FUNCTION void scatterLineToGrid(const ippl::NDIndex<3>& ldom, view_type& Jview, ippl::Vector<scalar, 3> hr, ippl::Vector<scalar, 3> origin, const ippl::Vector<scalar, 3>& from, const ippl::Vector<scalar, 3>& to, const scalar factor){ 

    
    Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> from_grid = gridCoordinatesOf(hr, origin, from);
    Kokkos::pair<ippl::Vector<int, 3>, ippl::Vector<scalar, 3>> to_grid   = gridCoordinatesOf(hr, origin, to  );
    //printf("Scatterdest: %.4e, %.4e, %.4e\n", from_grid.second[0], from_grid.second[1], from_grid.second[2]);
    for(int d = 0;d < 3;d++){
        //if(abs(from_grid.first[d] - to_grid.first[d]) > 1){
        //    std::cout <<abs(from_grid.first[d] - to_grid.first[d]) << " violation " << from_grid.first << " " << to_grid.first << std::endl;
        //}
        //assert(abs(from_grid.first[d] - to_grid.first[d]) <= 1);
    }
    //const uint32_t nghost = g.nghost();
    //from_ipos += ippl::Vector<int, 3>(nghost);
    //to_ipos += ippl::Vector<int, 3>(nghost);
    
    if(from_grid.first[0] == to_grid.first[0] && from_grid.first[1] == to_grid.first[1] && from_grid.first[2] == to_grid.first[2]){
        scatterToGrid(ldom, Jview, hr, origin, ippl::Vector<scalar, 3>((from + to) * scalar(0.5)), ippl::Vector<scalar, 3>((to - from) * factor));
        
        return;
    }
    ippl::Vector<scalar, 3> relay;
    const int nghost = 1;
    const ippl::Vector<scalar, 3> orig = origin;
    using Kokkos::max;
    using Kokkos::min;
    for (unsigned int i = 0; i < 3; i++) {
        relay[i] = min(min(from_grid.first[i] - nghost, to_grid.first[i] - nghost) * hr[i] + scalar(1.0) * hr[i] + orig[i],
                       max(max(from_grid.first[i] - nghost, to_grid.first[i] - nghost) * hr[i] + scalar(0.0) * hr[i] + orig[i],
                           scalar(0.5) * (to[i] + from[i])));
    }
    scatterToGrid(ldom, Jview, hr, origin, ippl::Vector<scalar, 3>((from + relay) * scalar(0.5)), ippl::Vector<scalar, 3>((relay - from) * factor));
    scatterToGrid(ldom, Jview, hr, origin, ippl::Vector<scalar, 3>((relay + to) * scalar(0.5))  , ippl::Vector<scalar, 3>((to - relay) * factor));
}

// END PREAMBLE


namespace ippl {
    
    template<typename scalar>
    struct undulator_parameters{
        scalar lambda; //MITHRA: lambda_u
        scalar K; //Undulator parameter
        scalar length;
        scalar B_magnitude;
        undulator_parameters(scalar K_undulator_parameter, scalar lambda_u, scalar _length) : lambda(lambda_u), K(K_undulator_parameter), length(_length){
            B_magnitude = (2 * M_PI * electron_mass_in_unit_masses * K) / (electron_charge_in_unit_charges * lambda_u);
            //std::cout << "Setting bmag: " << B_magnitude << "\n";
        }
        undulator_parameters(const config& cfg): lambda(cfg.undulator_period), K(cfg.undulator_K), length(cfg.undulator_length){
            B_magnitude = (2 * M_PI * electron_mass_in_unit_masses * K) / (electron_charge_in_unit_charges * lambda);
        }
    };

    template<typename scalar>
    struct nondispersive{
        scalar a1;
        scalar a2;
        scalar a4;
        scalar a6;
        scalar a8;
    };
    template <typename _scalar, class PLayout>
    struct  Bunch_eb : public ippl::ParticleBase<PLayout> {
        using scalar = _scalar;

        // Constructor for the Bunch class, taking a PLayout reference
        Bunch_eb(PLayout& playout)
            : ippl::ParticleBase<PLayout>(playout) {
            // Add attributes to the particle bunch
            this->addAttribute(Q);          // Charge attribute
            this->addAttribute(mass);       // Mass attribute
            this->addAttribute(gamma_beta); // Gamma-beta attribute (product of relativistiv gamma and beta)
            this->addAttribute(R_np1);      // Position attribute for the next time step
            this->addAttribute(R_nm1);      // Position attribute for the next time step
            this->addAttribute(EB_gather);   // Electric field attribute for particle gathering
        }

        // Destructor for the Bunch class
        ~Bunch_eb() {}

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
        typename ippl::ParticleBase<PLayout>::particle_position_type R_nm1; // Position container for the previous time step
        ippl::ParticleAttrib<ippl::Vector<ippl::Vector<scalar, 3>, 2>> EB_gather;   // Electric field container for particle gathering

    };
    template<typename scalar>
    struct Undulator{
        undulator_parameters<scalar> uparams;
        scalar distance_to_entry;
        scalar k_u;
        KOKKOS_FUNCTION Undulator(const undulator_parameters<scalar>& p, scalar dte) : uparams(p), distance_to_entry(dte), k_u(2 * M_PI / p.lambda){}
        KOKKOS_INLINE_FUNCTION Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> operator()(const ippl::Vector<scalar, 3>& position_in_lab_frame)const noexcept{
            using Kokkos::sin;
            using Kokkos::sinh;
            using Kokkos::cos;
            using Kokkos::cosh;
            using Kokkos::exp;
            Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> ret;
            ret.first.fill(0);
            ret.second.fill(0);
            if(position_in_lab_frame[2] < distance_to_entry){
                scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
                assert(z_in_undulator < 0);
                scalar scal = exp(-((k_u * z_in_undulator) * (k_u * z_in_undulator) * 0.5));
                ret.second[0] = 0;
                ret.second[1] = uparams.B_magnitude * cosh(k_u * position_in_lab_frame[1]) * z_in_undulator * k_u * scal;
                ret.second[2] = uparams.B_magnitude * sinh(k_u * position_in_lab_frame[1]) * scal;
            }
            else if(position_in_lab_frame[2] > distance_to_entry && position_in_lab_frame[2] < distance_to_entry + uparams.length){
                scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
                assert(z_in_undulator >= 0);
                ret.second[0] = 0;
                ret.second[1] = uparams.B_magnitude * cosh(k_u * position_in_lab_frame[1]) * sin(k_u * z_in_undulator);
                ret.second[2] = uparams.B_magnitude * sinh(k_u * position_in_lab_frame[1]) * cos(k_u * z_in_undulator);
            }
            return ret;
        };
    };
    


    template <typename scalar>
    // clang-format off
    struct FELSimulationState{
        
        //Sorry, can't do more than 3d

        constexpr static unsigned int dim = 3;
        using Vector_t = ippl::Vector<scalar, 3>;
        using value_type = ippl::Vector<scalar, 4>;
        using EB_type = ippl::Vector<ippl::Vector<scalar, 3>, 2>;
        using Mesh_t               = ippl::UniformCartesian<scalar, dim>;

        bool periodic_bc;
        using FourField = ippl::Field<value_type, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        using ThreeField = ippl::Field<Vector_t, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        using VField_t = ippl::Field<value_type, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        using EBField_t = ippl::Field<EB_type   , dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        using view_type = typename VField_t::view_type;
        using ev_view_type = typename EBField_t::view_type;
        using e_view_type = typename ThreeField::view_type;
        using b_view_type = typename ThreeField::view_type;
        //Fields
        ippl::NSFDSolverWithParticles<scalar, absorbing> fieldsAndParticles;
        
        //Discretization options
        Vector_t hr_m;
        ippl::Vector<uint32_t, 3> nr_global;
        ippl::Vector<uint32_t, 3> nr_local;
        //scalar dt;
        config m_config;
        UniaxialLorentzframe<scalar, 2 /*along z*/> ulb;
        undulator_parameters<scalar> uparams;
        Undulator<scalar> undulator;
        /**
         * @brief Construct a new FDTDFieldState object
         * Mesh and resolution parameter are technically redundant
         * @details ulb.gamma_m = cfg.bunch_gamma / std::sqrt(1 + cfg.undulator_K * cfg.undulator_K * 0.5) is the frame's gamma factor
         * @param resolution 
         * @param layout 
         * @param mesch 
         */
        FELSimulationState(FieldLayout<dim>& layout, Mesh_t& mesch, size_t nparticles, config cfg) : fieldsAndParticles(layout, mesch, nparticles), m_config(cfg), ulb(UniaxialLorentzframe<scalar, 2>::from_gamma(cfg.bunch_gamma / std::sqrt(1 + cfg.undulator_K * cfg.undulator_K * 0.5))), uparams(cfg), undulator(uparams, 2.0 * cfg.sigma_position[2] * ulb.gamma_m * ulb.gamma_m){

            hr_m = mesch.getMeshSpacing();
            nr_global = ippl::Vector<uint32_t, 3>{
                uint32_t(layout.getDomain()[0].last() - layout.getDomain()[0].first() + 1),
                uint32_t(layout.getDomain()[1].last() - layout.getDomain()[1].first() + 1),
                uint32_t(layout.getDomain()[2].last() - layout.getDomain()[2].first() + 1)
            };
            nr_local = ippl::Vector<uint32_t, 3>{
                uint32_t(layout.getLocalNDIndex()[0].last() - layout.getLocalNDIndex()[0].first() + 1),
                uint32_t(layout.getLocalNDIndex()[1].last() - layout.getLocalNDIndex()[1].first() + 1),
                uint32_t(layout.getLocalNDIndex()[2].last() - layout.getLocalNDIndex()[2].first() + 1)
            };
            //std::cout << "NR_M_g: " << nr_global << "\n";
            //std::cout << "NR_M_l: " << nr_local << "\n";
        }
        scalar dt()const noexcept{
            return fieldsAndParticles.field_solver.dt;
        }
        void step(){
            //scalar time = fieldsAndParticles.steps_taken * fieldsAndParticles.field_solver.dt;
            auto und = this->undulator;
            auto lb = this->ulb;
            fieldsAndParticles.solve(KOKKOS_LAMBDA(ippl::Vector<scalar, 3> pos, scalar time){
                lb.primedToUnprimed(pos, time);
                auto eb = und(pos);
                return lb.transform_EB(eb);
            });
        }
        void computeRadiation(){

        }
    };
    // clang-format on
}  // namespace ippl
bool writeBMPToFD(FILE* fd, int width, int height, const unsigned char* data) {
    const int channels = 3; // RGB
    const int stride = width * channels;
    std::vector<unsigned char> flippedData(data, data + stride * height);

    // Use stb_image_write to write the BMP image to the file descriptor
    if (!stbi_write_bmp_to_func(
            [](void* context, void* data, int size) {
                FILE* f = reinterpret_cast<FILE*>(context);
                fwrite(data, 1, size, f);
            },
            fd, width, height, channels, flippedData.data())) {
        return false;
    }

    return true;
}
template<typename View, typename T, unsigned Dim>
KOKKOS_INLINE_FUNCTION typename View::value_type gather_helper(const View& v, const ippl::Vector<T, Dim>& pos, const ippl::Vector<T, 3>& origin, const ippl::Vector<T, 3>& hr, const ippl::NDIndex<3>& lDom){
    using vector_type = ippl::Vector<T, 3>;

    vector_type l;
    //vector_type origin = v.get_mesh().getOrigin();
    //auto lDom = v.getLayout().getLocalNDIndex();
    //vector_type hr = v.get_mesh().getMeshSpacing();
    for(unsigned k = 0;k < 3;k++){
        l[k] = (pos[k] - origin[k]) / hr[k] + 1.0; //gather is implemented wrong
    }                     

    ippl::Vector<int, 3> index{int(l[0]), int(l[1]), int(l[2])};
    ippl::Vector<T, 3> whi = l - index;
    ippl::Vector<T, 3> wlo(1.0);
    wlo -= whi;
    //TODO: nghost
    ippl::Vector<size_t, 3> args = index - lDom.first() + 1;
    for(unsigned k = 0;k < 3;k++){
        if(args[k] >= v.extent(k) || args[k] == 0){
            return typename View::value_type(0);
        }
    }
    //std::cout << pos << " 's Gather args (will have 1 subtracted): " << args << "\n";
    return /*{true,*/ ippl::detail::gatherFromField(std::make_index_sequence<(1u << Dim)>{}, v, wlo, whi, args)/*}*/;

}
template<typename scalar>
scalar test_gauss_law(uint32_t n){

    ippl::NDIndex<3> owned(n / 2, n / 2, 2 * n);
    ippl::Vector<uint32_t, 3> nr{n / 2, n / 2, 2 * n};
    ippl::Vector<scalar, 3> extents{meter_in_unit_lengths,meter_in_unit_lengths,meter_in_unit_lengths};
    std::array<bool, 3> isParallel;
    isParallel.fill(false);
    isParallel[2] = true;
        
        // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

        //[-1, 1] box
    ippl::Vector<scalar, 3> hx     = extents / nr.cast<scalar>();
    using vector_type = ippl::Vector<scalar, 3>;
    ippl::Vector<scalar, 3> origin = {scalar(-0.5 * meter_in_unit_lengths), scalar(-0.5 * meter_in_unit_lengths), scalar(-0.5 * meter_in_unit_lengths)};
    ippl::UniformCartesian<scalar, 3> mesh(owned, hx, origin);

    uint32_t pcount = 1 << 20;
    //config cfg{};
    //cfg.space_charge = true;
    ippl::NSFDSolverWithParticles<scalar, ippl::fdtd_bc::absorbing> field_state(layout, mesh, pcount);
    field_state.particles.Q = scalar(coulomb_in_unit_charges) / pcount;
    field_state.particles.mass = scalar(1.0) / pcount; //Irrelefant
    auto pview = field_state.particles.R.getView();
    auto p1view = field_state.particles.R_nm1.getView();

    //constexpr scalar vy = meter_in_unit_lengths / second_in_unit_times;
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    //scalar dt = 0.5 ** std::min_element(hx.begin(), hx.end());
    
    Kokkos::parallel_for(pcount, KOKKOS_LAMBDA(size_t i){
        //bunch.gammaBeta[i].fill(scalar(0));
        auto state = random_pool.get_state();
        pview(i)[0] = state.normal(0.0, 0.01 * meter_in_unit_lengths);
        pview(i)[1] = state.normal(0.0, 0.01 * meter_in_unit_lengths);
        pview(i)[2] = state.normal(0.0, 0.01 * meter_in_unit_lengths);
        p1view(i) = pview(i);
        random_pool.free_state(state);
    });
    Kokkos::fence();
    field_state.J = scalar(0.0);
    field_state.scatterBunch();
    for(size_t i = 0;i < 8*n;i++){
        field_state.field_solver.solve();
    }

    Kokkos::fence();
    auto lDom = field_state.E.getLayout().getLocalNDIndex();
    
    std::ofstream line("gauss_line.txt");
    typename ippl::FELSimulationState<scalar>::e_view_type::host_mirror_type view = Kokkos::create_mirror_view(field_state.E.getView());
    //ippl::Vector<ippl::Vector<scalar, 3>, 2> ebg = gather_helper(view, ippl::Vector<scalar, 3>{0,0,0}, origin, hx, lDom);
    for(unsigned i = 1;i < nr[2];i++){
        vector_type pos = {scalar(0), scalar(0), (scalar)origin[2]};
        pos[2] += hx[2] * scalar(i);
        ippl::Vector<scalar, 3> ebg = gather_helper(view, pos, origin, hx, lDom);
        //line << pos.norm() * unit_length_in_meters << " " << (view(n / 4, n / 4, i)[0].norm()) * unit_electric_fieldstrength_in_voltpermeters << "\n";
        line << pos.norm() * unit_length_in_meters << " " << ebg.norm() * unit_electric_fieldstrength_in_voltpermeters << "\n";
    }
    return 0.0f;
}
template<typename scalar>
scalar test_amperes_law(uint32_t n){

    ippl::NDIndex<3> owned(n / 2, n / 2, 2 * n);
    ippl::Vector<uint32_t, 3> nr{n / 2, n / 2, 2 * n};
    ippl::Vector<scalar, 3> extents{meter_in_unit_lengths,(scalar)(4 * meter_in_unit_lengths),meter_in_unit_lengths};
    std::array<bool, 3> isParallel;
    isParallel.fill(false);
    isParallel[2] = true;
        
        // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

        //[-1, 1] box
    ippl::Vector<scalar, 3> hx;
    for(unsigned d = 0;d < 3;d++){
        hx[d] = extents[d] / (scalar)nr[d];
    }
    using vector_type = ippl::Vector<scalar, 3>;
    ippl::Vector<scalar, 3> origin = {scalar(-0.5 * meter_in_unit_lengths), scalar(-2.0 * meter_in_unit_lengths), scalar(-0.5 * meter_in_unit_lengths)};
    ippl::UniformCartesian<scalar, 3> mesh(owned, hx, origin);

    uint32_t pcount = 1 << 20;
    ippl::NSFDSolverWithParticles<scalar, ippl::fdtd_bc::absorbing> field_state(layout, mesh, pcount);
    field_state.particles.Q = scalar(4.0 * coulomb_in_unit_charges) / pcount;
    field_state.particles.mass = scalar(1.0) / pcount; //Irrelefant
    auto pview = field_state.particles.R.getView();
    auto p1view = field_state.particles.R_nm1.getView();
    constexpr scalar vy = meter_in_unit_lengths / second_in_unit_times;
    scalar timestep = field_state.field_solver.dt;
    //constexpr scalar vy = meter_in_unit_lengths / second_in_unit_times;
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    //scalar dt = 0.5 ** std::min_element(hx.begin(), hx.end());
    
    Kokkos::parallel_for(pcount, KOKKOS_LAMBDA(size_t i){
        //bunch.gammaBeta[i].fill(scalar(0));
        auto state = random_pool.get_state();
        pview(i)[0] = state.normal(0.0, 0.01 * meter_in_unit_lengths);
        pview(i)[2] = state.normal(0.0, 0.01 * meter_in_unit_lengths);
        pview(i)[1] = origin[1] + 4.0 * meter_in_unit_lengths * scalar(i) / (pcount - 1);
        p1view(i) = pview(i);
        p1view(i)[1] -= vy * timestep;
        random_pool.free_state(state);
    });
    Kokkos::fence();
    field_state.J = scalar(0.0);
    field_state.scatterBunch();
    for(size_t i = 0;i < 8*n;i++){
        field_state.field_solver.solve();
    }
    field_state.field_solver.evaluate_EB();
    Kokkos::fence();
    auto lDom = field_state.B.getLayout().getLocalNDIndex();
    
    std::ofstream line("ampere_line.txt");
    
    typename ippl::FELSimulationState<scalar>::b_view_type::host_mirror_type view = Kokkos::create_mirror_view(field_state.B.getView());
    //ippl::Vector<ippl::Vector<scalar, 3>, 2> ebg = gather_helper(view, ippl::Vector<scalar, 3>{0,0,0}, origin, hx, lDom);
    for(unsigned i = 1;i < nr[2];i++){
        vector_type pos = {scalar(0), scalar(0), (scalar)origin[2]};
        pos[2] += hx[2] * scalar(i);
        ippl::Vector<scalar, 3> ebg = gather_helper(view, pos, origin, hx, lDom);
        //line << pos.norm() * unit_length_in_meters << " " << (view(n / 4, n / 4, i)[0].norm()) * unit_electric_fieldstrength_in_voltpermeters << "\n";
        line << pos.norm() * unit_length_in_meters << " " << ebg[0] * unit_magnetic_fluxdensity_in_tesla << "\n";
    }
    return 0.0f;
}
int main(int argc, char* argv[]) {
    using scalar = float;
    ippl::initialize(argc, argv);
    {
        
        //test_gauss_law<scalar>(64);
        //test_amperes_law<scalar>(64);
        //goto exit;
        config cfg = read_config("../config.json");
        const scalar frame_gamma = std::max(decltype(cfg)::scalar(1), cfg.bunch_gamma / std::sqrt(1.0 + cfg.undulator_K * cfg.undulator_K * config::scalar(0.5)));
        cfg.extents[2] *= frame_gamma;
        cfg.total_time /= frame_gamma;
        
        const scalar frame_beta = std::sqrt(1.0 - 1.0 / double(frame_gamma * frame_gamma));
        const scalar frame_gammabeta = frame_gamma * frame_beta;
        UniaxialLorentzframe<scalar, 2> frame_boost(frame_gammabeta);
        ippl::undulator_parameters<scalar> uparams(cfg);
        /*const scalar k_u  = scalar(2.0 * M_PI) / uparams.lambda;
        const scalar distance_to_entry  = std::max(0.0 * uparams.lambda, 2.0 * cfg.sigma_position[2] * frame_gamma * frame_gamma);
        auto undulator_field = KOKKOS_LAMBDA(const ippl::Vector<scalar, 3>& position_in_lab_frame){
            Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> ret;
            ret.first.fill(0);
            ret.second.fill(0);

            if(position_in_lab_frame[2] < distance_to_entry){
                scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
                assert(z_in_undulator < 0);
                scalar scal = exp(-((k_u * z_in_undulator) * (k_u * z_in_undulator) * 0.5));
                ret.second[0] = 0;
                ret.second[1] = uparams.B_magnitude * cosh(k_u * position_in_lab_frame[1]) * z_in_undulator * k_u * scal;
                ret.second[2] = uparams.B_magnitude * sinh(k_u * position_in_lab_frame[1]) * scal;
            }
            else if(position_in_lab_frame[2] > distance_to_entry && position_in_lab_frame[2] < distance_to_entry + uparams.length){
                scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
                assert(z_in_undulator >= 0);
                ret.second[0] = 0;
                ret.second[1] = uparams.B_magnitude * cosh(k_u * position_in_lab_frame[1]) * sin(k_u * z_in_undulator);
                ret.second[2] = uparams.B_magnitude * sinh(k_u * position_in_lab_frame[1]) * cos(k_u * z_in_undulator);
            }
            return ret;

        };*/
        BunchInitialize<scalar> mithra_config = generate_mithra_config(cfg, frame_boost);
        ippl::NDIndex<3> owned(cfg.resolution[0], cfg.resolution[1], cfg.resolution[2]);

        std::array<bool, 3> isParallel;
        isParallel.fill(false);
        isParallel[2] = true;
        
        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

        //[-1, 1] box
        ippl::Vector<scalar, 3> hx     = {scalar( cfg.extents[0] / cfg.resolution[0]), scalar(cfg.extents[1] / cfg.resolution[1]), scalar(cfg.extents[2] / cfg.resolution[2])};
        ippl::Vector<scalar, 3> origin = {scalar(-cfg.extents[0] * 0.5), scalar(-cfg.extents[1] * 0.5), scalar(-cfg.extents[2] * 0.5)};
        ippl::UniformCartesian<scalar, 3> mesh(owned, hx, origin);
        std::cout << "hx: " << hx << "\n";
        std::cout << "origin: " << origin << "\n";
        std::cout << "extents: " << cfg.extents << std::endl;
        if(sq(hx[2] / hx[0]) + sq(hx[2] / hx[1]) >= 1){
            std::cerr << "Dispersion relation not satisfiable\n";
            abort();
        }

        ippl::FELSimulationState<scalar> fdtd_state(layout, mesh, 0 /*no resize function exists wtf cfg.num_particles*/, cfg);
        
        if(ippl::Comm->rank() == 0){
            std::cout << "Init particles: " << std::endl;
            size_t actual_pc = initialize_bunch_mithra(fdtd_state.fieldsAndParticles.particles, mithra_config, frame_gamma);
            fdtd_state.fieldsAndParticles.particles.Q = cfg.charge / actual_pc;
            fdtd_state.fieldsAndParticles.particles.mass = cfg.mass / actual_pc;
        }
        else{
            fdtd_state.fieldsAndParticles.particles.create(0);
        }
        {
            auto rview = fdtd_state.fieldsAndParticles.particles.R.getView();
            auto rm1view = fdtd_state.fieldsAndParticles.particles.R_nm1.getView();
            ippl::Vector<scalar, 3> meanpos = fdtd_state.fieldsAndParticles.particles.R.sum() * (1.0 / fdtd_state.fieldsAndParticles.particles.getTotalNum());
    
            Kokkos::parallel_for(fdtd_state.fieldsAndParticles.particles.getLocalNum(), KOKKOS_LAMBDA(size_t i){
                rview(i) -= meanpos;
                rm1view(i) -= meanpos;
            });
        }
        fdtd_state.fieldsAndParticles.particles.setParticleBC(ippl::NO);
        //fdtd_state.scatterBunch();
        //std::cout << cfg.charge << "\n";
        
        size_t timesteps_required = std::ceil(cfg.total_time / fdtd_state.dt());
        uint64_t starttime =  nanoTime();
        std::ofstream rad;
        FILE* ffmpeg_file = nullptr;
        if(ippl::Comm->rank() == 0){
            rad = std::ofstream("radiation.txt");
            const char* ffmpegCmd = "ffmpeg -y -f image2pipe -vcodec bmp -r 30 -i - -vf scale=force_original_aspect_ratio=decrease:force_divisible_by=2,format=yuv420p -c:v libx264 -tune zerolatency -movflags +faststart ffmpeg_popen.mkv";
            if(cfg.output_rhythm != 0){
                ffmpeg_file = popen(ffmpegCmd, "w");
            }
        }


        for(size_t i = 0;i < timesteps_required;i++){

            //fdtd_state.J = scalar(0.0);
            //fdtd_state.playout.update(fdtd_state.particles);
            //fdtd_state.scatterBunch();
            std::cout << i << "\n";
            fdtd_state.step();
            //fdtd_state.fieldStep();
            //fdtd_state.updateBunch(i * fdtd_state.dt, frame_boost, undulator_field);
            auto ldom = layout.getLocalNDIndex();
            auto nrg = fdtd_state.nr_global;
            auto eview = fdtd_state.fieldsAndParticles.E.getView();
            auto bview = fdtd_state.fieldsAndParticles.B.getView();
            //auto ebv = fdtd_state.EB.getView();
            double radiation = 0.0;
            Kokkos::parallel_reduce(ippl::getRangePolicy(eview, 1), KOKKOS_LAMBDA(uint32_t i, uint32_t j, uint32_t k, double& ref){
                //uint32_t ig = i + ldom.first()[0];
                //uint32_t jg = j + ldom.first()[1];
                Kokkos::pair<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> buncheb{eview(i,j,k), bview(i,j,k)};
                ippl::Vector<scalar, 3> Elab = frame_boost.inverse_transform_EB(buncheb).first;
                ippl::Vector<scalar, 3> Blab = frame_boost.inverse_transform_EB(buncheb).second;
                uint32_t kg = k + ldom.first()[2];
                if(kg == nrg[2] - 3){
                    ref += Elab.cross(Blab)[2];
                }

            }, radiation);
            double radiation_in_watt_on_this_rank = radiation *
            double(unit_powerdensity_in_watt_per_square_meter * unit_length_in_meters * unit_length_in_meters) *
            fdtd_state.hr_m[0] * fdtd_state.hr_m[1];
            double radiation_in_watt_global = 0.0;
            MPI_Reduce(&radiation_in_watt_on_this_rank, &radiation_in_watt_global, 1, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());
            if(ippl::Comm->rank() == 0){
                ippl::Vector<scalar, 3> pos{0,0,(float)cfg.extents[2]};
                frame_boost.primedToUnprimed(pos, fdtd_state.dt() * i);
                rad << pos[2] * unit_length_in_meters << " " << radiation_in_watt_global << "\n";
            }
            //std::cout << "A: " << fdtd_state.FA_n.getVolumeIntegral() << "\n";
            //std::cout << "J: " << fdtd_state.J.getVolumeIntegral() << "\n";
            //int rank = ippl::Comm->rank();
            //int size = ippl::Comm->size();
            if((cfg.output_rhythm != 0) && (i % cfg.output_rhythm == 0)){
                int width = 1920;
                int height = 700;
                rm::Vector<float, 3> pos{float(cfg.extents[1] * 0.75), 0.0f, (float)cfg.extents[2] * 0.4f};
                rm::Vector<float, 3> look{0.0f,0.0f,(float)cfg.extents[2] * 0.25f};
                ippl::Image parts =  ippl::drawParticles(fdtd_state.fieldsAndParticles.particles.R, width, height, rm::camera(pos, look - pos), cfg.extents[1] / 200.0f, ippl::Vector<float, 4>{0,1,0,1,0.3f});
                ippl::Image img = ippl::drawFieldFog(fdtd_state.fieldsAndParticles.B, width, height, rm::camera(pos, look - pos), []KOKKOS_FUNCTION(const ippl::Vector<scalar, 3> E){
                    return ippl::normalized_colormap(turbo_cm, Kokkos::sqrt(E.dot(E)) * 0.002f);
                }, parts);
                img.save_to(ffmpeg_file);
            }    
        }
        uint64_t endtime = nanoTime();
        if(ippl::Comm->rank() == 0)
            std::cout << ippl::Comm->size() << " " << double(endtime - starttime) / 1e9 << std::endl;
    }
    //exit:
    ippl::finalize();
}