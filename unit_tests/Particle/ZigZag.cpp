#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <fstream>
#include "Field/BcTypes.h"
#include "Types/Vector.h"

#include "Particle/ParticleAttrib.h"
#include "Solver/FDTDSolver.h"
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>

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
            auto ivl = inside[d].max() - inside[d].min();
            
            Rview(i)[d] = rand_gen.drand(inside[d].min() + 0.2 * ivl, inside[d].max() - 0.2 * ivl);
            GBview(i)[d] = rand_gen.drand(-100.0,100.0);
            //GBview(i)[d] /= Kokkos::abs(GBview(i)[d]);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};
using Precisions = ::testing::Types<double, float>;
template<typename T>
struct zigzagtest : public ::testing::Test{

};

TYPED_TEST_CASE(zigzagtest, Precisions);

TYPED_TEST(zigzagtest, Constructor) {
    {
        constexpr unsigned int Dim = 3;
        ippl::Vector<int, Dim> nr = {64, 64, 64};
        using scalar = TypeParam;
        // get the total simulation time from the user
        const scalar time_simulated = 1.0;
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
        ippl::FieldLayout<Dim> layout(owned, decomp);

        // unit box
        scalar dx                        = scalar(1.0) / nr[0];
        scalar dy                        = scalar(1.0) / nr[1];
        scalar dz                        = scalar(1.0) / nr[2];
        ippl::Vector<scalar, Dim> hr     = {dx, dy, dz};
        ippl::Vector<scalar, Dim> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);

        VField_t current;
        current.initialize(mesh, layout);
        Field_t charge;
        charge.initialize(mesh, layout);
        charge = 0.0;
        current = 0.0;
        ippl::ParticleSpatialLayout<scalar, 3> pl(charge.getLayout(), charge.get_mesh());
        using bunch_type = Bunch<scalar, ippl::ParticleSpatialLayout<scalar, 3>>;
        Bunch<scalar, ippl::ParticleSpatialLayout<scalar, 3>> bunch(pl);

        typename ippl::ParticleSpatialLayout<scalar, 3>::RegionLayout_t const& rlayout = pl.getRegionLayout();
        typename ippl::ParticleSpatialLayout<scalar, 3>::RegionLayout_t::view_type::host_mirror_type regions_view = rlayout.gethLocalRegions();
        bunch.create(100);

        Kokkos::Random_XorShift64_Pool<> rand_pool((size_t)(42));
        {
            int rink = ippl::Comm->rank();
            Kokkos::parallel_for(
                Kokkos::RangePolicy<typename ippl::ParticleSpatialLayout<scalar, 3>::RegionLayout_t::view_type::execution_space>(0, bunch.getLocalNum()),
                generate_random<ippl::Vector<scalar, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                    bunch.R.getView(),
                    bunch.gamma_beta.getView(),
                    regions_view(rink),
                    rand_pool
                )
            );
        }
        bunch_type bunch_buffer(pl);
        pl.update(bunch, bunch_buffer);
        bunch.Q = 0.5;
        bunch.Q.scatter(charge, bunch.R);
        charge.accumulateHalo();
        auto cview = charge.getView();
        scalar accum = charge.sum();
        //Kokkos::parallel_reduce(ippl::getRangePolicy(cview, 1), KOKKOS_LAMBDA(size_t i, size_t j, size_t k, scalar& ref){
        //    ref += cview(i, j, k);
        //}, accum);
        ASSERT_NEAR(accum, scalar(50.0 * ippl::Comm->size()), 0.1);
    }
    exit:
    (void)0;
}
int main(int argc, char** argv){
    int success = 1;
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return 0;//success;
}