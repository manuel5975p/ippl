#ifndef FRAST_MATH
#define FRAST_MATH
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <cmath>
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#endif
namespace rm{
template<typename T1, typename T2>
struct bigger_impl{
    using type = std::remove_all_extents_t<decltype(std::remove_all_extents_t<T1>{} + std::remove_all_extents_t<T2>{})>;
};
template<typename T1, typename T2>
using bigger = typename bigger_impl<T1, T2>::type;

template<typename T, unsigned int N>
struct Vector{
    using scalar = T;
    T data[N];

    #define OP4(X) KOKKOS_INLINE_FUNCTION constexpr Vector<T, N> operator X(const Vector<T, N>& o)const noexcept{\
        Vector<T, N> ret;\
        for(unsigned i = 0;i < N;i++){\
            ret[i] = (*this)[i] X o[i];\
        }\
        return ret;}
    #define OPA4(X) KOKKOS_INLINE_FUNCTION constexpr Vector<T, N>& operator X(const Vector<T, N>& o)noexcept{\
        for(unsigned i = 0;i < N;i++){\
            (*this)[i] X o[i];\
        }\
        return *this;}
    OP4(+)
    OP4(-)
    OP4(*)
    OP4(/)
    OPA4(+=)
    OPA4(-=)
    OPA4(*=)
    OPA4(/=)
    KOKKOS_INLINE_FUNCTION constexpr T operator[](size_t i)const noexcept{ return data[i]; }
    KOKKOS_INLINE_FUNCTION constexpr T& operator[](size_t i)noexcept{ return data[i]; }
    Vector<T, N> operator*(T x)const noexcept{
        Vector<T, N> ret;
        for(unsigned i = 0;i < N;i++){
            ret[i] = (*this)[i] * x;
        }
        return ret;
    }
    void setZero()noexcept{
        for(unsigned i = 0;i < N;i++){
            (*this)[i] = 0;
        }
    }
    template<typename O>
    bigger<T, O> dot(const Vector<O, N>& o)const noexcept{
        bigger<T, O> ret(0);
        for(unsigned i = 0;i < N;i++){
            ret += (*this)[i] * o[i];
        }
        return ret;
    }
    template<typename str>
    friend str& operator<<(str& s, const Vector<T, N>& x){
        s << "(";
        for(unsigned i = 0;i < N;i++){
            s << x[i];
            if(i < N - 1)s << " ";
        }
        return s << ")";
    }
    template<unsigned S>
    KOKKOS_INLINE_FUNCTION Vector<T, S> head()const noexcept{
        Vector<T, S> ret;
        for(unsigned i = 0;i < S;i++)ret[i] = (*this)[i];
        return ret;
    }
    template<unsigned S>
    KOKKOS_INLINE_FUNCTION Vector<T, S> tail()const noexcept{
        Vector<T, S> ret;
        for(unsigned i = 0;i < S;i++)ret[i] = (*this)[N - S + i];
        return ret;
    }
    KOKKOS_INLINE_FUNCTION constexpr T norm() const noexcept {
        using std::sqrt;
        return sqrt(squaredNorm());
    }

    KOKKOS_INLINE_FUNCTION constexpr T squaredNorm() const noexcept {
        T sum = 0;
        for (unsigned i = 0; i < N; ++i) {
            sum += data[i] * data[i];
        }
        return sum;
    }

    KOKKOS_INLINE_FUNCTION Vector<T, N>& normalize() noexcept {
        T in = T(1) / norm();
        *this = *this * in;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION Vector<T, N> normalized() const noexcept {
        T n = norm();
        return (*this) * (T(1.0) / n);
    }
    KOKKOS_INLINE_FUNCTION Vector<T, N> cross(const Vector<T, N>& o) const noexcept {
        static_assert(N == 3, "cross requires N == 3");
        return Vector<T, 3>{
            data[1] * o[2] - data[2] * o[1],
            data[2] * o[0] - data[0] * o[2],
            data[0] * o[1] - data[1] * o[0]
        };
    }

    template <typename U = T>
    requires(N == 3)
    KOKKOS_INLINE_FUNCTION Vector<U, 4> one_extend() const noexcept {
        Vector<U, 4> result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data[i] = data[i];
        }
        result.data[3] = 1;
        return result;
    }
    template <typename U = T>
    requires(N == 2)
    KOKKOS_INLINE_FUNCTION Vector<U, 3> zero_extend() const noexcept {
        Vector<U, 3> result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data[i] = data[i];
        }
        result[2] = 0;
        return result;
    }

    KOKKOS_INLINE_FUNCTION auto zerooneextend(){
        return this->zero_extend().one_extend();
    }
    template<typename O>
    KOKKOS_INLINE_FUNCTION 
    Vector<O, N> cast()const noexcept{
        Vector<O, N> ret;
        for (std::size_t i = 0; i < N; ++i) {
            ret.data[i] = O(data[i]);
        }
        return ret;
    }
    template <typename U = T>
    requires(N >= 1)
    KOKKOS_INLINE_FUNCTION T x() const noexcept {
        return data[0];
    }
    template <typename U = T>
    requires(N >= 2)
    KOKKOS_INLINE_FUNCTION T y() const noexcept {
        return data[1];
    }
    template <typename U = T>
    requires(N >= 3)
    KOKKOS_INLINE_FUNCTION T z() const noexcept {
        return data[2];
    }
    template <typename U = T>
    requires(N >= 4)
    KOKKOS_INLINE_FUNCTION T w() const noexcept {
        return data[3];
    }
};
template<typename T, unsigned M, unsigned N>
struct Matrix{
    Vector<T, M> data[N]; //Column-major
    template<typename str>
    friend str& operator<<(str& s, const Matrix<T, M, N>& x){
        for(unsigned i = 0;i < M;i++){
            for(unsigned j = 0;j < N;j++){
                s << x.data[j][i] << " ";
            }
            if(i < M - 1)s << "\n";
        }
        return s;
    }
    template<typename O, unsigned otherN>
    Matrix<bigger<T, O>, M, otherN> operator*(const Matrix<O, N, otherN>& mat)const noexcept{
        Matrix<bigger<T, O>, M, otherN> result;

        for (unsigned i = 0; i < M; ++i) {
            for (unsigned j = 0; j < otherN; ++j) {
                result.data[j][i] = 0;
                for (unsigned k = 0; k < N; ++k) {
                    result.data[j][i] += data[k][i] * mat.data[j][k];
                }
            }
        }
    
        return result;
    }
    T operator()(size_t i, size_t j)const noexcept{
        return data[j][i];
    }
    T& operator()(size_t i, size_t j) noexcept{
        return data[j][i];
    }
    void setZero(){
        for(size_t i = 0;i < N;i++){
            data[i].setZero();
        }
    }
    void setIdentity(){
        for(size_t i = 0;i < N;i++){
            data[i].setZero();
            data[i][i] = 1;
        }
    }
    template<typename O>
    Vector<bigger<T, O>, M> operator*(const Vector<O, N>& vec)const noexcept{
        Vector<bigger<T, O>, M> result;
        for (unsigned i = 0; i < M; ++i)result[i] = 0;

        for (unsigned i = 0; i < N; ++i) {
            result += data[i] * vec[i];
        }
    
        return result;
    }
    template<typename O>
    KOKKOS_INLINE_FUNCTION
    Matrix<O, M, N> cast()const noexcept{
        Matrix<O, M, N> ret;
        for (std::size_t i = 0; i < N; ++i) {
            ret.data[i] = data[i].template cast<O>();
        }
        return ret;
    }
};
template<typename T>
Matrix<T, 4, 4> lookAt(Vector<T, 3> const& eye, Vector<T, 3> const& center, Vector<T, 3> const& up){
	const Vector<T, 3> f((center - eye).normalized());
	const Vector<T, 3> s((f.cross(up).normalized()));
	const Vector<T, 3> u(s.cross(f));
	Matrix<T, 4, 4> Result;
    Result.setIdentity();
	Result(0, 0) = s.x();
	Result(0, 1) = s.y();
	Result(0, 2) = s.z();
	Result(1, 0) = u.x();
	Result(1, 1) = u.y();
	Result(1, 2) = u.z();
	Result(2, 0) =-f.x();
	Result(2, 1) =-f.y();
	Result(2, 2) =-f.z();
	Result(0, 3) = -s.dot(eye);
	Result(1, 3) = -u.dot(eye);
	Result(2, 3) =  f.dot(eye);
	return Result;
}
template<typename T>
Matrix<T, 4, 4> perspectiveRH_NO(T fovy, T aspect, T zNear, T zFar){
    using std::abs;
	assert(abs(aspect - std::numeric_limits<T>::epsilon()) > T(0));
	T const tanHalfFovy = tan(fovy / T(2));
	Matrix<T, 4, 4> Result;
    Result.setZero();
	Result(0, 0) = T(1) / (aspect * tanHalfFovy);
	Result(1, 1) = T(1) / (tanHalfFovy);
	Result(2, 2) = - (zFar + zNear) / (zFar - zNear);
	Result(3, 2) = - T(1);
	Result(2, 3) = - (T(2) * zFar * zNear) / (zFar - zNear);
	return Result;
}
template<typename T>
Matrix<T, 4, 4> ortho(T left, T right, T bottom, T top, T zNear, T zFar){
	Matrix<T, 4, 4> result;
    result.setIdentity();
	result(0, 0) = T(2) / (right - left);
	result(1, 1) = T(2) / (top - bottom);
	result(2, 2) = -T(2) / (zFar - zNear);
	result(0, 3) = -(right + left) / (right - left);
	result(1, 3) = -(top + bottom) / (top - bottom);
	result(2, 3) = -(zFar + zNear) / (zFar - zNear);
	return result;
}
struct camera{
    using vec3 = Vector<float, 3>;
    using mat4 = Matrix<float, 4, 4>;
    vec3 pos;
    float pitch, yaw;
    camera(vec3 p, float pt, float y) : pos(p), pitch(pt), yaw(y){

    }
    camera(vec3 p, vec3 look) : pos(p){
        look = look.normalized();
        pitch = std::asin(look.y());
        yaw = std::atan2(look.z(), look.x());
    }
    vec3 look_dir()const noexcept{
        vec3 fwd{std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};
        return fwd;
    }
    vec3 left()const noexcept{
        vec3 fwd{std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};
        vec3 up{0,1,0};
        return fwd.cross(up);
    }

    mat4 view_matrix()const noexcept{
        vec3 up       {0,1,0};
        vec3 fwd      {std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};

        //[[maybe_unused]] vec3 realup = {fwd.cross(fwd.cross(up))};
        //[[maybe_unused]] vec3 right =  fwd.cross(realup);
        mat4 ret = lookAt(pos, pos + fwd, up);
        return ret;
    }
    mat4 perspective_matrix(float width, float height)const noexcept{
        return perspectiveRH_NO(1.0f, width / height, 0.1f, 200.0f);
    }
    mat4 matrix(float width, float height)const noexcept{
        return perspective_matrix(width, height) * view_matrix();
    }
};
}
#endif