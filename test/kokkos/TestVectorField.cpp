#include <Kokkos_Core.hpp>

#include <initializer_list>
#include <iostream>

template <typename E>
class VectorExpr {
public:
    KOKKOS_INLINE_FUNCTION
    double operator[](size_t i) const {
        return static_cast<E const&>(*this)[i];
    }
};


template <typename T, unsigned D>
class Vector : public VectorExpr<Vector<T, D>> {

public:
    KOKKOS_FUNCTION
    Vector() : Vector(T(0)) {}

    KOKKOS_FUNCTION
    Vector(T val) {
        for (unsigned i = 0; i < D; ++i) {
            data_m[i] = val;
        }
    }

    KOKKOS_FUNCTION
    Vector(const std::initializer_list<T>& l) {
        int i = 0;
        for(auto a : l) {
            data_m[i] = a;
            ++i;
        }
    }

    KOKKOS_FUNCTION
    T operator[](const int i) const {
        return data_m[i];
    }

    KOKKOS_FUNCTION
    T& operator[](const int i) {
        return data_m[i];
    }

    KOKKOS_FUNCTION
    ~Vector() {}


    template <typename E>
    KOKKOS_FUNCTION
    Vector& operator=(VectorExpr<E> const& expr) {
        for (unsigned i = 0; i < D; ++i) {
            data_m[i] = expr[i];
        }
        return *this;
    }

private:
  T data_m[D];
};


template <typename E1, typename E2>
class VecSum : public VectorExpr<VecSum<E1, E2> > {
    E1 const _u;
    E2 const _v;

public:
    KOKKOS_FUNCTION
    VecSum(E1 const& u, E2 const& v) : _u(u), _v(v) { }

    KOKKOS_INLINE_FUNCTION
    double operator[](size_t i) const { return _u[i] + _v[i]; }
};



template <typename E1, typename E2>
KOKKOS_FUNCTION
VecSum<E1, E2>
operator+(VectorExpr<E1> const& u, VectorExpr<E2> const& v) {
   return VecSum<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
}


template<class T1, class T2> struct meta_cross {};

template<class T1, class T2>
struct meta_cross< Vector<T1, 3> , Vector<T2, 3> >
{
    KOKKOS_INLINE_FUNCTION
    static Vector<double, 3>
    apply(const Vector<T1, 3>& a, const Vector<T2, 3>& b) {
        Vector<double, 3> c;
        c[0] = a[1]*b[2] - a[2]*b[1];
        c[1] = a[2]*b[0] - a[0]*b[2];
        c[2] = a[0]*b[1] - a[1]*b[0];
        return c;
    }
    };


template < class T1, class T2, unsigned D >
KOKKOS_INLINE_FUNCTION Vector<T1,D>
cross(const Vector<T1,D> &lhs, const Vector<T2,D> &rhs)
{
    return meta_cross< Vector<T1,D> , Vector<T2,D> > :: apply(lhs,rhs);
}



int main(int argc, char *argv[]) {

    Kokkos::initialize(argc,argv);
    {
        constexpr int length = 10;

        typedef Vector<double, 3> vector_type;

        typedef Kokkos::View<vector_type*> vector_field_type;

        vector_field_type vfield("vfield", length);

        Kokkos::parallel_for("assign", length, KOKKOS_LAMBDA(const int i) {
            vfield(i) = {1.0, 2.0, 3.0};
        });


        vector_field_type wfield("wfield", length);

        Kokkos::parallel_for("assign", length, KOKKOS_LAMBDA(const int i) {
            wfield(i) = {4.0, -5.0, 6.0};
        });


        Kokkos::parallel_for("assign", length, KOKKOS_LAMBDA(const int i) {
            vfield(i) = cross(vfield(i), wfield(i)) + vfield(i);
        });

        Kokkos::fence();

        vector_field_type::HostMirror host_view = Kokkos::create_mirror_view(vfield);
        Kokkos::deep_copy(host_view, vfield);


        for (int i = 0; i < length; ++i) {
            std::cout << host_view(i)[0] << " " << host_view(i)[1] << " " << host_view(i)[2] << std::endl;
        }
    }
    Kokkos::finalize();

    return 0;
}
