#include <cstddef>
using std::size_t;
#include <Kokkos_Core.hpp>
#include "Ippl.h"
#include "Types/Vector.h"
#include "Field/Field.h"
#include "MaxwellSolvers/FDTD.h"
#include "Utility/Rendering.hpp"

#include <Kokkos_Random.hpp>
template<typename scalar1, typename... scalar>
    requires((std::is_floating_point_v<scalar1>))
KOKKOS_INLINE_FUNCTION float gauss(scalar1 mean, scalar1 stddev, scalar... x){
    uint32_t dim = sizeof...(scalar);
    ippl::Vector<scalar1, sizeof...(scalar)> vec{scalar1(x - mean)...};
    scalar1 vecsum(0);
    for(unsigned d = 0;d < dim;d++){
        vecsum += vec[d] * vec[d];
        
    }
    #ifndef __CUDA_ARCH__
    using std::exp;
    #endif
    return exp(-(vecsum) / (stddev * stddev)); 
}

int main(int argc, char* argv[]){
    ippl::initialize(argc, argv);
    {
        using scalar = float;
        //const unsigned dim = 3;
        //using vector_type = ippl::Vector<scalar, 3>;
        //using vector4_type = ippl::Vector<scalar, 4>;
        //using FourField = ippl::Field<vector4_type, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        //using ThreeField = ippl::Field<vector_type, dim, ippl::UniformCartesian<scalar, dim>, typename ippl::UniformCartesian<scalar, dim>::DefaultCentering>;
        constexpr size_t n = 50;
        ippl::Vector<uint32_t, 3> nr{n / 2, n / 2, 2 * n};
        ippl::NDIndex<3> owned(nr[0], nr[1], nr[2]);
                               
        ippl::Vector<scalar, 3> extents{1.0,
                                        1.0,
                                        1.0};

        std::array<bool, 3> isParallel;
        isParallel.fill(false);
        isParallel[2] = true;

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

        ippl::Vector<scalar, 3> hx;
        for(unsigned d = 0;d < 3;d++){
            hx[d] = extents[d] / (scalar)nr[d];
        }
        ippl::Vector<scalar, 3> origin = {0,0,0};
        ippl::UniformCartesian<scalar, 3> mesh(owned, hx, origin);
        
        ippl::NSFDSolverWithParticles<scalar, ippl::absorbing> solver(layout, mesh, 1 << 5);

        
        auto pview = solver.particles.R.getView();
        auto p1view = solver.particles.R_nm1.getView();
        auto gbview = solver.particles.gamma_beta.getView();

        Kokkos::Random_XorShift64_Pool<> random_pool(12345);
        Kokkos::parallel_for(solver.particles.getLocalNum(), KOKKOS_LAMBDA(size_t i){
            auto state = random_pool.get_state();
            pview(i)[0] = state.normal(origin[0] + extents[0] * 0.5, 0.04 * extents[0]);
            pview(i)[1] = state.normal(origin[1] + extents[1] * 0.5, 0.04 * extents[1]);
            pview(i)[2] = state.normal(origin[2] + extents[2] * 0.5, 0.04 * extents[2]);
            p1view(i) = pview(i);
            gbview(i) = 0;
            random_pool.free_state(state);
        });
        {
            double var = 0;
            Kokkos::parallel_reduce(solver.particles.getLocalNum(), KOKKOS_LAMBDA(size_t i, double& ref){
                ippl::Vector<scalar, 3> pd(pview(i) - (origin + extents * 0.5));
                ref += pd.dot(pd);
            }, var);
            std::cout << ippl::Vector<double, 3>(var * (1.0 / solver.particles.getLocalNum())) << "\n";
        }
        solver.playout.update(solver.particles);
        solver.particles.Q = electron_charge_in_unit_charges * 1000;
        solver.particles.mass = electron_mass_in_unit_masses;
        for(int i = 0;i < 1200;i++){
            solver.solve();
            if(true){
                using vec3 = rm::Vector<float, 3>;
                //ippl::Image img = ippl::drawFieldCrossSection(solver.E, 1000, 1000, ippl::axis::x, extents[0] / 2, KOKKOS_LAMBDA(const ippl::Vector<scalar, 3>& E){
                //    return ippl::normalized_colormap(turbo_cm, Kokkos::sqrt(E.dot(E)) / 5000.0f);
                //});
                rm::camera cam(vec3{0.5,0.5,-1}, vec3{0,0,1});
                ippl::Image img = ippl::drawParticles(solver.particles.R, 700, 700, cam, 0.01f, ippl::Vector<float, 4>{0,1,0,1});
                ippl::Image img2 = ippl::drawFieldFog(solver.E, 700, 700,cam, KOKKOS_LAMBDA(const ippl::Vector<scalar, 3>& E){
                    return ippl::normalized_colormap(turbo_cm, Kokkos::sqrt(E.dot(E)) / 5000.0f);
                }, img);
                img2.save_to("renderdata/" + std::format("{:05d}.png", i));
            }
        }
        {
            double var = 0;
            Kokkos::parallel_reduce(solver.particles.getLocalNum(), KOKKOS_LAMBDA(size_t i, double& ref){
                ippl::Vector<scalar, 3> pd(pview(i) - (origin + extents * 0.5));
                ref += pd.dot(pd);
            }, var);
            std::cout << ippl::Vector<double, 3>(var * (1.0 / solver.particles.getLocalNum())) << "\n";
        }
        
    }
    //exit:
    ippl::finalize();
}