// Landau Damping Test
//   Usage:
//     srun ./LandauDamping
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT and CG supported)
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./LandauDamping 128 128 128 10000 10 FFT 0.01 --overallocate 2.0 --info 10
//
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan,
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

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "Ippl.h"
#include "Utility/IpplTimings.h"
#include "Manager/PicManager.h"
#include "datatypes.h"
#include "ParticleContainer.hpp"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
 #include "LandauDampingManager.h"
 #include "Random/InverseTransformSampling.h"    
 

KOKKOS_FUNCTION
double CDF(double y, double alpha, double k) {
    return y + (alpha / k) * std::sin(k * y);
}

KOKKOS_FUNCTION
double PDF(double y, double alpha, double k) {
    return  (1.0 + alpha * Kokkos::cos(k * y));
}

KOKKOS_FUNCTION
double ESTIMATE(double u, double alpha) {
    return u/(1+alpha); // maybe E[x] is good enough as the first guess
}

KOKKOS_FUNCTION
double PDF3D(const Vector_t<double, Dim>& xvec, const double& alpha, const Vector_t<double, Dim>& kw,
           const unsigned Dim) {
    double pdf = 1.0;

    for (unsigned d = 0; d < Dim; ++d) {
        pdf *= (1.0 + alpha * Kokkos::cos(kw[d] * xvec[d]));
    }
    return pdf;
}

const char* TestName = "LandauDamping";

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("LandauDamping");
        Inform msg2all("LandauDamping", INFORM_ALL_NODES);
        
        LandauDampingManager manager;

        int arg = 1;
        // read input arguments
        for (unsigned d = 0; d < Dim; d++) {
            manager.nr[d] = std::atoi(argv[arg++]);
        }
        manager.totalP = std::atoll(argv[arg++]);
        manager.nt  = std::atoi(argv[arg++]);
        manager.solver = argv[arg++];
        manager.lbs = std::atof(argv[arg++]);
        
       manager.pre_run();
        // create mesh and layout objects for this problem domain
        const bool isAllPeriodic = true;
        Mesh_t<Dim> mesh(manager.domain, manager.hr, manager.origin);
        FieldLayout_t<Dim> FL(manager.domain, manager.decomp, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);


        if (manager.solver == "OPEN") {
            throw IpplException("LandauDamping",
                                "Open boundaries solver incompatible with this simulation!");
        }
        
        using ParticleContainer_t = ParticleContainer<T, Dim>;
        std::shared_ptr<ParticleContainer_t> pc = std::make_shared<ParticleContainer_t>(PL);
        
        using FieldContainer_t = FieldContainer<T, Dim>;
        std::shared_ptr<FieldContainer_t> fc = std::make_shared<FieldContainer_t>(manager.hr, manager.rmin, manager.rmax, manager.decomp);
        
        fc->initializeFields(mesh, FL);
        
        using FieldSolver_t= FieldSolver<T, Dim>;
        std::shared_ptr<FieldSolver_t> fs = std::make_shared<FieldSolver_t>(manager.solver, fc->rho_m, fc->E_m);
        
        using LoadBalancer_t= LoadBalancer<T, Dim>;
        std::shared_ptr<LoadBalancer_t> lb = std::make_shared<LoadBalancer_t>(manager.solver, manager.lbs, fc->rho_m, fc->E_m, FL, pc->R);

        fs->initSolver();
                
        manager.setParticleContainer(pc);
        manager.setFieldContainer(fc);
        manager.setFieldSolver(fs);
        manager.setLoadBalancer(lb);
        
 	bool isFirstRepartition;
        if ((manager.loadbalancethreshold_m != 1.0) && (ippl::Comm->size() > 1)) {
            msg << "Starting first repartition" << endl;
            isFirstRepartition             = true;
            const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
            const int nghost               = fc->rho_m.getNghost();
            auto rhoview                   = fc->rho_m.getView();

            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            ippl::parallel_for(
                "Assign initial rho based on PDF", fc->rho_m.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // local to global index conversion
                    Vector_t<double, Dim> xvec = (args + lDom.first() - nghost + 0.5) * manager.hr + manager.origin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(rhoview, args) = PDF3D(xvec, manager.alpha, manager.kw, Dim);
                });

            Kokkos::fence();

            lb->initializeORB(FL, mesh);
            lb->repartition(FL, mesh, isFirstRepartition);
        }
        
        
         // Sample particle positions:
         ippl::detail::RegionLayout<double, 3, Mesh_t<3>> rlayout(FL, mesh);
         using InvTransSampl_t = ippl::random::InverseTransformSampling<double, 3, Kokkos::DefaultExecutionSpace>;
         ippl::random::Distribution<double, 3> distR;
         for(int d=0; d<3; d++){
                 double k = manager.kw[d];
                 double alpha = manager.alpha;
		 distR.setCdfFunction(d, [alpha, k](double y) { return CDF(y, alpha, k);});
		 distR.setPdfFunction(d, [alpha, k](double y) { return PDF(y, alpha, k);});
		 distR.setEstimationFunction(d, [alpha](double u) { return ESTIMATE(u, alpha);});
         }
         InvTransSampl_t its(manager.rmin, manager.rmax, rlayout, distR, manager.totalP);
         unsigned int nloc = its.getLocalNum();
         pc->create(nloc);
         its.generate(distR, pc->R.getView(), 42 + 100 * ippl::Comm->rank());
         
         // Sample particle velocity:
         // Box-Muller method
         //Kokkos::parallel_for(
         //   nloc, ippl::random::generate_random_normal<double, Kokkos::DefaultExecutionSpace, 3>(
         //            pc->P.getView(), 0.0, 1.0, 42 + 100 * ippl::Comm->rank()));
         
         // standard method
         Kokkos::Random_XorShift64_Pool<> rand_pool64_((size_type)(42 + 100 * ippl::Comm->rank()));
         Kokkos::parallel_for(
            nloc, ippl::random::generate_random_normal_basic<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                     pc->P.getView(), rand_pool64_));
        
        Kokkos::fence();
        ippl::Comm->barrier();

        pc->q = manager.Q / manager.totalP;
        msg << "particles created and initial conditions assigned " << endl;
        
        fc->rho_m = 0.0;
        fs->runSolver();
        manager.par2grid();
        fs->runSolver();
        manager.grid2par();

        manager.time_m = 0.0;
        
        // begin main timestep loop
        isFirstRepartition = false;
        msg << "Starting iterations ..." << endl;
        for (manager.it = 0; manager.it < manager.nt; manager.it++) {
        
            // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick

            pc->P = pc->P - 0.5 * manager.dt * pc->E;

            // drift
            pc->R = pc->R + manager.dt * pc->P;

            // Since the particles have moved spatially update them to correct processors
            pc->update();

            // Domain Decomposition
            if (lb->balance(manager.totalP, manager.it + 1)) {
                msg << "Starting repartition" << endl;
                lb->repartition(FL, mesh, isFirstRepartition);
            }

            // scatter the charge onto the underlying grid
            manager.par2grid();
            
            // Field solve
            fs->runSolver();

            // gather E field
            manager.grid2par();

            // kick
            pc->P = pc->P - 0.5 * manager.dt * pc->E;


            manager.time_m += manager.dt;            
            manager.dumpLandau();

            msg << "Finished time step: " << manager.it + 1 << " time: " << manager.time_m << endl;
            
        }
        msg << "LandauDamping: End." << endl;
        
    }
    
    ippl::finalize();

    return 0;
}
