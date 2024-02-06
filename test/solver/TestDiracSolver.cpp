#include "Ippl.h"
#include <complex>
#include <fstream>
#include <tuple>
#include <source_location>
#include <Eigen/Dense>
template<typename scalar>
using complex = std::complex<scalar>;
template<typename scalar>
using spinor = ippl::Vector<complex<scalar>, 4>;
template<typename scalar>
using eigen_spinor = Eigen::Matrix<complex<scalar>, 4, 1>;
template<typename value_type, unsigned int dim>
using ippl_matrix = ippl::Vector<ippl::Vector<value_type, dim>, dim>;

template<typename value_type, unsigned int dim>
using eigen_vector = Eigen::Matrix<value_type, dim, 1>;

template<typename value_type, unsigned int dim>
using eigen_matrix = Eigen::Matrix<value_type, dim, dim>;

template<typename value_type, unsigned int dim>
eigen_matrix<value_type, dim> to_eigen(const ippl_matrix<value_type, dim>& mat){
    eigen_matrix<value_type, dim> ret;
    for(unsigned int i = 0;i < dim;i++)
        for(unsigned int j = 0;j < dim;j++)
            ret(i, j) = mat[i][j];

    return ret;
}
template<typename value_type, unsigned int dim>
eigen_vector<value_type, dim> to_eigen(const ippl::Vector<value_type, dim>& vec){
    eigen_vector<value_type, dim> ret;
    for(unsigned int i = 0;i < dim;i++)
        ret(i) = vec[i];

    return ret;
}
template<typename value_type, unsigned int dim>
ippl::Vector<value_type, dim> to_ippl(const eigen_vector<value_type, dim>& vec){
    ippl::Vector<value_type, dim> ret;
    for(unsigned int i = 0;i < dim;i++)
        ret[i] = vec(i);

    return ret;
}

template<typename value_type, unsigned int dim>
ippl_matrix<value_type, dim> to_ippl(const eigen_matrix<value_type, dim>& mat){
    ippl_matrix<value_type, dim> ret;
    for(unsigned int i = 0;i < dim;i++)
        for(unsigned int j = 0;j < dim;j++)
            ret[i][j] = mat(i, j);

    return ret;
}

template<typename scalar>
spinor<scalar> flux_function_for_dx(const spinor<scalar>& psi){
    spinor<scalar> ret{-psi[3], -psi[2], -psi[1], -psi[0]};
    return ret;
}
template<typename scalar>
spinor<scalar> flux_function_for_dy(const spinor<scalar>& psi){
    using namespace std::complex_literals;
    const typename spinor<scalar>::value_type I = complex<scalar>(scalar(0), scalar(1));
    spinor<scalar> ret{psi[3] * I, -psi[2] * I, psi[1] * I, -psi[0] * I};
    return ret;
}
template<typename scalar>
spinor<scalar> flux_function_for_dz(const spinor<scalar>& psi){
    spinor<scalar> ret{-psi[2], psi[3], -psi[0], psi[1]};
    return ret;
}

template<unsigned int axis, typename scalar>
spinor<scalar> flux_function(const spinor<scalar>& psi){
    if constexpr(axis == 0)return flux_function_for_dx(psi);
    if constexpr(axis == 1)return flux_function_for_dy(psi);
    if constexpr(axis == 2)return flux_function_for_dz(psi);
}
template<typename scalar>
ippl_matrix<complex<scalar>, 4> flux_J_for_dx(){
    ippl_matrix<complex<scalar>, 4> ret;
    for(int i = 0;i < 4;i++)
        ret[i] = spinor<scalar>{complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0)};
    ret[0][3] = -1;
    ret[1][2] = -1;
    ret[2][1] = -1;
    ret[3][0] = -1;
    return ret;
}
template<typename scalar>
ippl_matrix<complex<scalar>, 4> flux_J_for_dy(){
    using namespace std::complex_literals;
    const typename spinor<scalar>::value_type I = complex<scalar>(scalar(0), scalar(1));;
    ippl_matrix<complex<scalar>, 4> ret;
    for(int i = 0;i < 4;i++)
        ret[i] = spinor<scalar>{complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0)};
    ret[0][3] = I;
    ret[1][2] = -I;
    ret[2][1] = I;
    ret[3][0] = -I;
    return ret;
}
template<typename scalar>
ippl_matrix<complex<scalar>, 4> flux_J_for_dz(){
    ippl_matrix<complex<scalar>, 4> ret;
    for(int i = 0;i < 4;i++)
        ret[i] = spinor<scalar>{complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0),complex<scalar>(0.0)};
    ret[0][2] = -1;
    ret[1][3] = 1;
    ret[2][0] = -1;
    ret[3][1] = 1;
    return ret;
}
template<typename scalar>
spinor<scalar> mass_term(const spinor<scalar>& psi){
    using namespace std::complex_literals;
    const typename spinor<scalar>::value_type I = complex<scalar>(scalar(0), scalar(1));
    spinor<scalar> ret(psi);
    ret[0] *= -I;ret[1] *= -I;ret[2] *= I;ret[3] *= I;
    return ret;
}
template<typename value_type, unsigned int dim>
struct eigensystem{
    using ippl_matrix_type = ippl_matrix <value_type, dim>;
    using matrix_type      = eigen_matrix<value_type, dim>;
    using vector_type      = eigen_vector<value_type, dim>;
    vector_type eigenvalues;
    matrix_type eigenvectors;
    matrix_type eigenvectors_inverse;
    constexpr unsigned int Dim(){return dim;}
    eigensystem() = default;
    eigensystem(const matrix_type& emat){
        Eigen::ComplexEigenSolver<eigen_matrix<value_type, dim>> solver(emat);
        eigenvalues = solver.eigenvalues();
        eigenvectors = solver.eigenvectors();
        eigenvectors_inverse = eigenvectors.inverse();
    }
    eigensystem(const ippl_matrix_type& x) : eigensystem(to_eigen(x)){}
};
template<typename value_type, unsigned int dim>
eigensystem(const eigen_matrix<value_type, dim>& x) -> eigensystem<value_type, dim>;
template<typename value_type, unsigned int dim>
eigensystem(const ippl_matrix<value_type, dim>& x) -> eigensystem<value_type, dim>;

template<typename scalar, int dim>
spinor<scalar> lax_friedrichs_flux_d(const spinor<scalar>& l, const spinor<scalar>& r){
    //return l;
    eigensystem<complex<scalar>, 4U> sys;
    if constexpr(dim == 0)
        sys = eigensystem(flux_J_for_dx<scalar>());
    else if constexpr (dim == 1)
        sys = eigensystem(flux_J_for_dy<scalar>());
    else if constexpr (dim == 2)
        sys = eigensystem(flux_J_for_dz<scalar>());

    eigen_spinor<scalar> l_in_eigenbasis = sys.eigenvectors_inverse * to_eigen(l);
    eigen_spinor<scalar> r_in_eigenbasis = sys.eigenvectors_inverse * to_eigen(r);
    

    eigen_spinor<scalar> composition_in_eigenbasis;
    //composition_in_eigenbasis.fill(decltype(composition_in_eigenbasis)::Scalar(0.0));
    for(unsigned int i = 0;i < sys.Dim();i++){
        if(sys.eigenvalues(i).real() < 0.0){
            composition_in_eigenbasis(i) = l_in_eigenbasis(i);
        }
        else{
            composition_in_eigenbasis(i) = r_in_eigenbasis(i);
        }
    }
    eigen_spinor<scalar> composition = sys.eigenvectors * composition_in_eigenbasis;
    if constexpr(dim == 0)
    return flux_function_for_dx<scalar>(to_ippl<complex<scalar>, 4>(composition));
    else if constexpr (dim == 1)
    return flux_function_for_dy<scalar>(to_ippl<complex<scalar>, 4>(composition));
    else if constexpr (dim == 2)
    return flux_function_for_dz<scalar>(to_ippl<complex<scalar>, 4>(composition));
}
KOKKOS_INLINE_FUNCTION auto square(auto x){
    return x * x;
}
template<typename T, unsigned N>
ippl::Vector<T, N> reverse(const ippl::Vector<T, N>& arg){
    ippl::Vector<T, N> ret;
    for(unsigned i = 0;i < N;i++){
        ret[i] = arg[N - 1 - i];
    }
    return ret;
}
template<typename T, unsigned N>
ippl::Vector<T, N> cwise_prod(const ippl::Vector<T, N>& arg1, const ippl::Vector<T, N>& arg2){
    ippl::Vector<T, N> ret;
    for(unsigned i = 0;i < N;i++){
        ret[i] = arg1[i] * arg2[i];
    }
    return ret;
}
template<typename T, unsigned N>
ippl::Vector<T, N> cwise_prod(const ippl::Vector<T, N>& arg1, const T& arg2){
    ippl::Vector<T, N> ret;
    for(unsigned i = 0;i < N;i++){
        ret[i] = arg1[i] * arg2;
    }
    return ret;
}
template<typename T, unsigned N>
T sum(const ippl::Vector<T, N>& arg){
    T init = 0;
    for(unsigned i = 0;i < N;i++){
        init += arg[i];
    }
    return init;
}
//CAREFUL: hardcoded
template<typename T, unsigned N>
struct actual_array_goddamnit{
    T data[N];
};
template<typename T>
constexpr KOKKOS_INLINE_FUNCTION actual_array_goddamnit<T, 3> lweights(){return {1.0 / 3.0, -7.0 / 6.0, 11.0/ 6.0};};
template<typename T>
constexpr KOKKOS_INLINE_FUNCTION actual_array_goddamnit<T, 3> cweights(){return {-1.0/ 6.0,  5.0 / 6.0, 1.0 / 3.0};};
template<typename T>
constexpr KOKKOS_INLINE_FUNCTION actual_array_goddamnit<T, 3> rweights(){return {1.0 / 3.0,  5.0 / 6.0, -1.0/ 6.0};};

template<typename T>
ippl::Vector<T, 3> stencil3(const ippl::Vector<T, 5>& u){
    ippl::Vector<T, 3> ret;
    ret[0] = ( 1.0 / 3.0) * u[0] - 7.0 / 6.0 * u[1] + 11.0 / 6.0 * u[2];
    ret[1] = (-1.0 / 6.0) * u[1] + 5.0 / 6.0 * u[2] + 1.0  / 3.0 * u[3];
    ret[2] = ( 1.0 / 3.0) * u[2] + 5.0 / 6.0 * u[3] - 1.0  / 6.0 * u[4];
    return ret;
}
template<typename T>
ippl::Vector<T, 3> smoothness_measures(const ippl::Vector<T, 5>& u){
    ippl::Vector<T, 3> ret;
    ret(0) = 13.0 / 12.0 * square(u[0] - 2 * u[1] + u[2]) + 0.25 * square(u[0] - 4 * u[1] + 3 * u[2]);
    ret(1) = 13.0 / 12.0 * square(u[1] - 2 * u[2] + u[3]) + 0.25 * square(u[1] - u[3]);
    ret(2) = 13.0 / 12.0 * square(u[2] - 2 * u[3] + u[4]) + 0.25 * square(3 * u[2] - 4 * u[3] + u[4]);
    return ret;
}
template<typename T>
ippl::Vector<T, 3> scronch_weights(const ippl::Vector<T, 3>& smoothness){
    constexpr T epsilon = 1e-7;

    ippl::Vector<T, 3> unscronched{0.1, 0.6, 0.3};
    ippl::Vector<T, 3> smoothness_multiplier;
    for(unsigned i = 0;i < 3;i++){
        smoothness_multiplier[i] = 1.0 / square(smoothness[i] + epsilon);
    }
    ippl::Vector<T, 3> w_notnorm = cwise_prod(unscronched, smoothness_multiplier);
    T inorm = 1.0 / sum(w_notnorm);

    return cwise_prod(w_notnorm, inorm);
}
template<typename T>
KOKKOS_INLINE_FUNCTION T R_L(ippl::Vector<T, 5> values){
    ippl::Vector<T, 3> smm = smoothness_measures(values);
    ippl::Vector<T, 3> st = stencil3(values);
    ippl::Vector<T, 3> weights = scronch_weights(smm);
    return sum(cwise_prod(st, weights));
}

template<typename T>
KOKKOS_INLINE_FUNCTION T R_R(ippl::Vector<T, 5> values){
    return R_L(reverse(values));
}
template<unsigned offsetaxis, typename view_type>
KOKKOS_INLINE_FUNCTION typename view_type::value_type access_with_offset(const view_type& v, size_t i, size_t j, size_t k, int offset){
    using Kokkos::abs;
    if constexpr(offsetaxis == 0){
        return v(i + offset, j, k);
    }
    if constexpr(offsetaxis == 1){
        return v(i, j + offset, k);
    }
    if constexpr(offsetaxis == 2){
        return v(i, j, k + offset);
    }
}
constexpr unsigned int Dim = 3;
template<typename scalar>
using mesh_t      = ippl::UniformCartesian<scalar, Dim>;
template<typename scalar>
using centering_t = typename mesh_t<scalar>::DefaultCentering;
template<typename scalar>
using spinor_field_template = ippl::Field<spinor<scalar>, Dim, mesh_t<scalar>, centering_t<scalar>>;

template<typename scalar>
using scalar_field_template = ippl::Field<scalar, Dim, mesh_t<scalar>, centering_t<scalar>>;

template<typename scalar>
scalar_field_template<scalar> to_scalar_field(const spinor_field_template<scalar>& spinor_field){
    scalar_field_template<scalar> scalar_field;
    mesh_t<scalar> mesh = spinor_field.get_mesh();
    ippl::FieldLayout<Dim> layout = spinor_field.getLayout();
    scalar_field.initialize(mesh, layout, 3);
    const Kokkos::View<const spinor<scalar>***> spinor_field_view = spinor_field.getView();
          Kokkos::View<             scalar ***> scalar_field_view = scalar_field.getView();
    Kokkos::parallel_for(ippl::getRangePolicy(spinor_field.getView(), 3),
            KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                #ifndef __CUDA_ARCH__
                using std::conj;
                #endif
                scalar_field_view(i, j, k) = 
                (spinor_field_view(i, j, k)[0] * conj(spinor_field_view(i, j, k)[0])).real() +
                (spinor_field_view(i, j, k)[1] * conj(spinor_field_view(i, j, k)[1])).real() +
                (spinor_field_view(i, j, k)[2] * conj(spinor_field_view(i, j, k)[2])).real() +
                (spinor_field_view(i, j, k)[3] * conj(spinor_field_view(i, j, k)[3])).real() ;
            }
    );
    return scalar_field;
}
template<typename scalar>
void dumpVTK(const scalar_field_template<scalar>& rho,
     /*Extents */int    nx, int    ny, int    nz,      int iteration,
     /*Spacings*/double dx, double dy, double dz) {
    using field_type = scalar_field_template<scalar>;
    typename field_type::view_type::host_mirror_type host_view = rho.getHostMirror();

    std::stringstream fname;
    fname << "data/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    Kokkos::deep_copy(host_view, rho.getView());

    std::ofstream vtkout(fname.str().c_str(), std::ios::trunc);
    vtkout.precision(6);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    // start with header
    vtkout << "# vtk DataFile Version 2.0\n";
    vtkout << "TestDirac\n";
    vtkout << "ASCII\n";
    vtkout << "DATASET STRUCTURED_POINTS\n";
    vtkout << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " " << nz + 1 << '\n';
    vtkout << "ORIGIN " << -dx << " " << -dy << " " << -dz << '\n';
    vtkout << "SPACING " << dx << " " << dy << " " << dz << '\n';
    vtkout << "CELL_DATA " << (nx + 2) * (ny + 2) * (nz + 2) << '\n';

    vtkout << "SCALARS Rho float\n";
    vtkout << "LOOKUP_TABLE default\n";
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                vtkout << host_view(x, y, z) << '\n';
            }
        }
    }
    vtkout << std::endl;
}
template<size_t i, typename... Ts>
auto get(Ts&&... args){
    return std::get<i>(std::tie(std::forward<Ts>(args)...));
}
template<unsigned int axis, typename scalar>
KOKKOS_INLINE_FUNCTION Kokkos::pair<spinor<scalar>, spinor<scalar>> discretize(const Kokkos::View<spinor<scalar>***>& view, size_t i, size_t j, size_t k){



    Kokkos::pair<spinor<scalar>, spinor<scalar>> ret;
    if constexpr(true){
        ret.first  = flux_function<axis>(access_with_offset<axis>(view, i, j, k, 0)) + access_with_offset<axis>(view, i, j, k, 0);
        ret.first  *= 0.5;
        ret.second = flux_function<axis>(access_with_offset<axis>(view, i, j, k, 0)) - access_with_offset<axis>(view, i, j, k, 0);
        ret.second *= 0.5;
        return ret;
    }
    
    
    
    for(unsigned comp = 0;comp < 4;comp++){
        ippl::Vector<scalar, 5> f_plus_real_reconstruction_input;
        ippl::Vector<scalar, 5> f_plus_imag_reconstruction_input;
        ippl::Vector<scalar, 5> f_minus_real_reconstruction_input;
        ippl::Vector<scalar, 5> f_minus_imag_reconstruction_input;
        for(int off = -2;off <= 2;off++){
            spinor<scalar> f_plus_u  = flux_function<axis>(access_with_offset<axis>(view, i, j, k, off)) - access_with_offset<axis>(view, i, j, k, off);
            spinor<scalar> f_minus_u = flux_function<axis>(access_with_offset<axis>(view, i, j, k, off)) + access_with_offset<axis>(view, i, j, k, off);
            f_plus_u  *= 0.5;
            f_minus_u *= 0.5;
            f_plus_real_reconstruction_input[off + 2]  = f_plus_u [comp].real();
            f_plus_imag_reconstruction_input[off + 2]  = f_plus_u [comp].imag();
            f_minus_real_reconstruction_input[off + 2] = f_minus_u[comp].real();
            f_minus_imag_reconstruction_input[off + 2] = f_minus_u[comp].imag();
        }
        scalar f_plus_realpart_at_i_plus_one_half   = R_L(f_plus_real_reconstruction_input);
        scalar f_minus_realpart_at_i_minus_one_half = R_R(f_minus_real_reconstruction_input);

        scalar f_plus_imagpart_at_i_plus_one_half   = R_L(f_plus_imag_reconstruction_input);
        scalar f_minus_imagpart_at_i_minus_one_half = R_R(f_minus_imag_reconstruction_input);

        ret.first[comp].real(f_minus_realpart_at_i_minus_one_half);
        ret.first[comp].imag(f_minus_imagpart_at_i_minus_one_half);

        ret.second[comp].real(f_plus_realpart_at_i_plus_one_half);
        ret.second[comp].imag(f_plus_imagpart_at_i_plus_one_half);
    }
    return ret;
}
template<typename scalar>
using sppair = Kokkos::pair<spinor<scalar>, spinor<scalar>>;
template<unsigned int axis, typename scalar>
KOKKOS_INLINE_FUNCTION spinor<scalar> Lterm(const Kokkos::View<spinor<scalar>***>& view, size_t i, size_t j, size_t k){
    spinor<scalar> ret;
    sppair<scalar> pears[3];
    for(int off = -1;off <= 1;off++){
        pears[off + 1] = discretize<axis>(view, i + off * int(axis == 0), j + off * int(axis == 1), k + off * int(axis == 2));
    }
    for(unsigned comp = 0; comp < 4;comp++){
        ret[comp] = (pears[2].first[comp] + pears[1].second[comp]) - (pears[0].second[comp] + pears[1].first[comp]);
    }
    return ret;
}


int main(int argc, char* argv[]) {
    using namespace std::complex_literals;

    ippl::initialize(argc, argv);{
        using scalar = double;
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);
        
        // get the gridsize from the user
        
        //using scalar_field = ippl::Field<scalar, Dim, mesh_t, centering_t>;
        using spinor_field = spinor_field_template<scalar>;
        constexpr scalar electron_mass = 0.5110;

        constexpr size_t extents = 16; 
        ippl::Vector<size_t, Dim> nr = {extents, extents * 4, extents};
        
        scalar dx                        = scalar(1.0) / nr[0];
        scalar dy                        = scalar(1.0) / nr[1];
        scalar dz                        = scalar(1.0) / nr[2];
        scalar dt = std::min<scalar>({dx, dy, dz}) * 1.0;
        ippl::Vector<scalar, Dim> hr     = {dx, dy, dz};
        ippl::Vector<scalar, Dim> origin = {0.0, 0.0, 0.0};
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        
        ippl::NDIndex<Dim> owned;
        for (unsigned i = 0; i < Dim; i++) {
            owned[i] = ippl::Index(nr[i]);
        }
        
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel);
        mesh_t<scalar> mesh(owned, hr, origin);

        spinor_field field;
        spinor_field field_n_plus_one;
        field.initialize(mesh, layout, 3);
        field_n_plus_one.initialize(mesh, layout, 3);
        using bc_type = typename decltype(field)::BConds_t;
        bc_type spinor_bcs;
        auto bcsetter_single = [&spinor_bcs, hr]<size_t Idx>(const std::index_sequence<Idx>&) {
            spinor_bcs[Idx] = std::make_shared<ippl::PeriodicFace<decltype(field)>>(Idx);
            return 0;
        };
        auto bcsetter = [bcsetter_single]<size_t... Idx>(const std::index_sequence<Idx...>&) {
            int x = (bcsetter_single(std::index_sequence<Idx>{}) ^ ...);
            (void)x;
        };
        bcsetter(std::make_index_sequence<Dim * 2>{});
        field.setFieldBC(spinor_bcs);
        
        field_n_plus_one.setFieldBC(spinor_bcs);
        field = spinor<scalar>{0,0,0,0};
        field_n_plus_one = spinor<scalar>{0,0,0,0};
        Kokkos::View<spinor<scalar>***> field_view = field.getView();
        spinor_field u1_field;
        u1_field.initialize(mesh, layout, 3);
        spinor_field u2_field;
        u2_field.initialize(mesh, layout, 3);
        Kokkos::View<spinor<scalar>***> u_1 = u1_field.getView();
        Kokkos::View<spinor<scalar>***> u_2 = u2_field.getView();


        Kokkos::View<spinor<scalar>***> field_n_plus_one_view = field_n_plus_one.getView();
        //field_view(extents / 2, extents / 2, extents / 2) = spinor<scalar>{1, 0, 0, 1};
        unsigned int steps = 0.2 / dt;

        {
            auto esx = eigensystem(flux_J_for_dx<scalar>());
            auto esy = eigensystem(flux_J_for_dy<scalar>());
            auto esz = eigensystem(flux_J_for_dz<scalar>());
            std::cout.precision(3);
            std::cout << "Esx EVS:\n" << esx.eigenvectors << "\n";
            std::cout << "Esy EVS:\n" << esy.eigenvectors << "\n";
            std::cout << "Esz EVS:\n" << esz.eigenvectors << "\n";


            Kokkos::parallel_for(ippl::getRangePolicy(field.getView(), 3),
            KOKKOS_LAMBDA(size_t i, size_t j, size_t k){

                ippl::Vector<double, 3> p{double(i - 3) / (nr[0] - 1),
                                          double(j - 3) / (nr[1] - 1),
                                          double(k - 3) / (nr[2] - 1)};
                //std::cout << std::format("{}", scal) << "\n";
                using Kokkos::exp;
                double sq = 0;
                for(int c = 0;c < 3;c++){
                    sq += (c == 1) * (p[c] - 0.5) * (p[c] - 0.5);
                }
                field_view(i, j, k) = std::complex(0.0, 0.0);
                field_view(i, j, k)[0] = exp(sq * -200.0);
                field_view(i, j, k)[1] = exp(sq * -200.0);
            });
        }
        std::cout << steps << " steps" << std::endl;
        bool tvdscheme = false;
        for(unsigned step = 0;step < steps;step++){
            field.getFieldBC().apply(field);
            if(tvdscheme){
                Kokkos::parallel_for(ippl::getRangePolicy(field.getView(), 3),
                KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                    spinor<scalar> flux_i_minus_one_half, flux_i_plus_one_half;
                    spinor<scalar> value_ijk = field_view(i, j, k);

                    spinor<scalar> rate_of_change = (Lterm<0>(field_view, i, j, k)) / hr[0] +
                                                    (Lterm<1>(field_view, i, j, k)) / hr[1] +
                                                    (Lterm<2>(field_view, i, j, k)) / hr[2];


                    u_1(i, j, k) = field_view(i, j, k) + rate_of_change * dt;
                });
                Kokkos::fence();
                u1_field.getFieldBC().apply(u1_field);
                Kokkos::parallel_for(ippl::getRangePolicy(field.getView(), 3),
                KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                    spinor<scalar> flux_i_minus_one_half, flux_i_plus_one_half;
                    spinor<scalar> value_ijk = field_view(i, j, k);

                    spinor<scalar> rate_of_change = (Lterm<0>(u_1, i, j, k)) / hr[0] +
                                                    (Lterm<1>(u_1, i, j, k)) / hr[1] +
                                                    (Lterm<2>(u_1, i, j, k)) / hr[2];

                    for(int c = 0;c < 4;c++)
                        u_2(i, j, k)[c] = 0.75 * field_view(i, j, k)[c] + 0.25 * u_1(i,j,k)[c] + 0.25 * dt * rate_of_change[c];
                });
                Kokkos::fence();
                u2_field.getFieldBC().apply(u2_field);
                Kokkos::parallel_for(ippl::getRangePolicy(field.getView(), 3),
                KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                    spinor<scalar> flux_i_minus_one_half, flux_i_plus_one_half;
                    spinor<scalar> value_ijk = field_view(i, j, k);

                    spinor<scalar> rate_of_change = (Lterm<0>(u_2, i, j, k)) / hr[0] +
                                                    (Lterm<1>(u_2, i, j, k)) / hr[1] +
                                                    (Lterm<2>(u_2, i, j, k)) / hr[2];


                    for(int c = 0;c < 4;c++)
                        field_n_plus_one_view(i, j, k)[c] = (1.0 / 3.0) * field_view(i, j, k)[c] + (2.0 / 3.0) * u_2(i,j,k)[c] + (2.0 / 3.0) * dt * rate_of_change[c];
                });
            }
            else{
                field.getFieldBC().apply(field);
                Kokkos::fence();
                Kokkos::parallel_for(ippl::getRangePolicy(field.getView(), 3),
                KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                    spinor<scalar> flux_i_minus_one_half, flux_i_plus_one_half;
                    spinor<scalar> value_ijk = field_view(i, j, k);

                    spinor<scalar> rate_of_change = Lterm<0>(field_view, i, j, k) / hr[0];
                    //std::cout << std::format("{}\n", sum(rate_of_change).real());

                    for(int c = 0;c < 4;c++)
                        field_n_plus_one_view(i, j, k)[c] = field_view(i, j, k)[c] + dt * rate_of_change[c];
                });
                Kokkos::fence();
                Kokkos::deep_copy(field_view, field_n_plus_one_view);
                field.getFieldBC().apply(field);
                Kokkos::fence();
                Kokkos::parallel_for(ippl::getRangePolicy(field.getView(), 3),
                KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                    spinor<scalar> flux_i_minus_one_half, flux_i_plus_one_half;
                    spinor<scalar> value_ijk = field_view(i, j, k);

                    spinor<scalar> rate_of_change = Lterm<1>(field_view, i, j, k) / hr[1];


                    for(int c = 0;c < 4;c++)
                        field_n_plus_one_view(i, j, k)[c] = field_view(i, j, k)[c] + dt * rate_of_change[c];
                });
                Kokkos::fence();
                Kokkos::deep_copy(field_view, field_n_plus_one_view);
                field.getFieldBC().apply(field);
                Kokkos::fence();
                Kokkos::parallel_for(ippl::getRangePolicy(field.getView(), 3),
                KOKKOS_LAMBDA(size_t i, size_t j, size_t k){
                    spinor<scalar> flux_i_minus_one_half, flux_i_plus_one_half;
                    spinor<scalar> value_ijk = field_view(i, j, k);

                    spinor<scalar> rate_of_change = Lterm<2>(field_view, i, j, k) / hr[2];
                    std::cout << std::format("{}\n", sum(rate_of_change).real());

                    for(int c = 0;c < 4;c++)
                        field_n_plus_one_view(i, j, k)[c] = field_view(i, j, k)[c] + dt * rate_of_change[c];
                });
                Kokkos::fence();
                Kokkos::deep_copy(field_view, field_n_plus_one_view);
                Kokkos::fence();
            }
            Kokkos::fence();
            scalar_field_template<scalar> scalar_field = to_scalar_field(field);
            //std::cout << sum(field_view(extents / 2, extents / 2, extents / 2)) << "\n";
            //dumpVTK(scalar_field, extents, extents, extents, step, hr[0], hr[1], hr[2]);
            Kokkos::deep_copy(field_view, field_n_plus_one_view);
        }
        std::ofstream of("cline.txt");
        for(int i = 3;i < 3 + nr[1];i++){
            of << sum(field_view(nr[0] / 2, i, nr[2] / 2)).real() + sum(field_view(nr[0] / 2, i, nr[2] / 2)).imag() << "\n";
        }

    }
    ippl::finalize();
}
