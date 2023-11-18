#include "Ippl.h"

#include <cstdlib>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <fstream>
#include "Field/BcTypes.h"
#include "Types/Vector.h"

#include "Particle/ParticleAttrib.h"
#include "Solver/FDTDSolver.h"
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    
    {
        constexpr unsigned int Dim = 3;
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
    }
    exit:
    (void)0;
}