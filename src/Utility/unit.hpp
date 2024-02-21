#ifndef UNITS_HPP
#define UNITS_HPP
#include <Kokkos_Macros.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <concepts>
namespace funits {
    template<int exp>
    KOKKOS_INLINE_FUNCTION constexpr auto ipow(auto base) -> decltype(base){
        if constexpr(exp < 0)return decltype(base)(1.0) / ipow<-exp>(base);
        if constexpr(exp == 0) return 1.0;
        decltype(base) ret(1.0);
        for(int i = 0;i < exp;i++){
            ret *= base;
        }
        return ret;
    }
    struct SI_base{
        template<unsigned dimension>
        KOKKOS_INLINE_FUNCTION constexpr static double ratio(){
            return 1.0;
        }
    };
    struct planck_base{
        template<unsigned dimension>
        KOKKOS_INLINE_FUNCTION constexpr static double ratio(){
            if constexpr(dimension == 0){
                return 2.176434e-8;
            }
            if constexpr(dimension == 1){
                return 1.616255e-35;
            }
            if constexpr(dimension == 2){
                return 5.391247e-44;
            }
            if constexpr(dimension == 3){
                return 1.416784e32;
            }
            if constexpr(dimension == 4){
                return 3.4789e25;
            }
        }
    };

    struct cgs_base{
        template<unsigned dimension>
        KOKKOS_INLINE_FUNCTION constexpr static double ratio(){
            if constexpr(dimension == 0){
                return double(1e-3);
            }
            if constexpr(dimension == 1){
                return double(1e-2);
            }
            if constexpr(dimension == 2){
                return 1.0;
            }
            if constexpr(dimension == 3){
                return 1.0;
            }
            if constexpr(dimension == 4){
                return 1.0;
            }
        }
    };
    constexpr wchar_t exponents[] = L"⁰¹²³⁴⁵⁶⁷⁸⁹";
    template <int _mt, int _lt, int _tt, int _Tt, int _it, typename _scalart = double, typename _unit_base = SI_base>
    struct unit {
        constexpr static int m = _mt;
        constexpr static int l = _lt;
        constexpr static int t = _tt;
        constexpr static int T = _Tt;
        constexpr static int i = _it;
        using scalar           = _scalart;
        using unit_base           = _unit_base;
        using this_type        = unit<m, l, t, T, i, scalar, unit_base>;
        scalar value;
        KOKKOS_INLINE_FUNCTION constexpr unit()
            : value(1) {}
        KOKKOS_INLINE_FUNCTION constexpr unit(const scalar& v)
            : value(v) {}
        auto operator=(scalar x) = delete;
        template <int _m, int _l, int _t, int _T, int _i, typename _scalar>
        KOKKOS_INLINE_FUNCTION auto& operator=(const unit<_m, _l, _t, _T, _i, _scalar>& o) {
            static_assert(m == _m, "Non-matching dimensions");
            static_assert(l == _l, "Non-matching dimensions");
            static_assert(t == _t, "Non-matching dimensions");
            static_assert(T == _T, "Non-matching dimensions");
            static_assert(i == _i, "Non-matching dimensions");
            value = o.value;
            return *this;
        }
        template <int _m, int _l, int _t, int _T, int _i, typename _scalar>
        KOKKOS_INLINE_FUNCTION auto operator*(const unit<_m, _l, _t, _T, _i, _scalar>& o) const {
            return unit<m + _m, l + _l, t + _t, T + _T, i + _T, scalar>(value * o.value);
        }
        template <int _m, int _l, int _t, int _T, int _i, typename _scalar>
        KOKKOS_INLINE_FUNCTION auto operator/(const unit<_m, _l, _t, _T, _i, _scalar>& o) const {
            return unit<m - _m, l - _l, t - _t, T - _T, i - _T, scalar>(value / o.value);
        }
        template <int _m, int _l, int _t, int _T, int _i, typename _scalar>
        KOKKOS_INLINE_FUNCTION auto operator+(const unit<_m, _l, _t, _T, _i, _scalar>& o) const {
            static_assert(m == _m, "Non-matching dimensions");
            static_assert(l == _l, "Non-matching dimensions");
            static_assert(t == _t, "Non-matching dimensions");
            static_assert(T == _T, "Non-matching dimensions");
            static_assert(i == _i, "Non-matching dimensions");
            return unit<m, l, t, T, i, scalar>(value + o.value);
        }
        template <int _m, int _l, int _t, int _T, int _i, typename _scalar>
        KOKKOS_INLINE_FUNCTION auto operator-(const unit<_m, _l, _t, _T, _i, _scalar>& o) const {
            static_assert(m == _m, "Non-matching dimensions");
            static_assert(l == _l, "Non-matching dimensions");
            static_assert(t == _t, "Non-matching dimensions");
            static_assert(T == _T, "Non-matching dimensions");
            static_assert(i == _i, "Non-matching dimensions");
            return unit<m, l, t, T, i, scalar>(value - o.value);
        }
        template<typename o_scalar>
            requires(std::is_fundamental_v<o_scalar>)
        KOKKOS_INLINE_FUNCTION this_type operator*  (const o_scalar& x) const { return this_type(value * x); }
        template<typename o_scalar>
            requires(std::is_fundamental_v<o_scalar>)
        KOKKOS_INLINE_FUNCTION this_type operator/  (const o_scalar& x) const { return this_type(value / x); }
        template<typename o_scalar>
            requires(std::is_fundamental_v<o_scalar>)
        KOKKOS_INLINE_FUNCTION this_type& operator*=(const o_scalar& x) const {
            value *= x;
            return *this;
        }
        template<typename o_scalar>
            requires(std::is_fundamental_v<o_scalar>)
        KOKKOS_INLINE_FUNCTION this_type& operator/=(const o_scalar& x) const {
            value /= x;
            return *this;
        }
        KOKKOS_INLINE_FUNCTION scalar count() const noexcept { 
            //std::cout << "CONVERSION HAPPèENED" << std::endl;
            return value; 
        }
        KOKKOS_INLINE_FUNCTION operator scalar() const requires(m == 0 && l == 0 && t == 0 && T == 0 && i == 0) { 
            //std::cout << "CONVERSION HAPPèENED" << std::endl;
            return value; 
        }
        template<typename other_base>
        KOKKOS_INLINE_FUNCTION unit<m,l,t,T,i,scalar, other_base> convert_to()const noexcept{
            scalar val(value);
            val *= ipow<-m>(other_base::template ratio<0>() / unit_base::template ratio<0>());
            val *= ipow<-l>(other_base::template ratio<1>() / unit_base::template ratio<1>());
            val *= ipow<-t>(other_base::template ratio<2>() / unit_base::template ratio<2>());
            val *= ipow<-T>(other_base::template ratio<3>() / unit_base::template ratio<3>());
            val *= ipow<-i>(other_base::template ratio<4>() / unit_base::template ratio<4>());
            return unit<m, l, t, T, i, scalar, other_base>(val);
        }
        // friend std::ostream& operator<<(std::ostream& ost, const this_type& o);
    };
    template <typename unit1, typename unit2>
    struct unit_product {
        using type = unit<unit1::m + unit2::m, unit1::l + unit2::l, unit1::t + unit2::t,
                          unit1::T + unit2::T, unit1::i + unit2::i, typename unit1::scalar, typename unit1::unit_base>;
    };
    template <typename unit1, typename unit2>
    struct unit_quotient {
        using type = unit<unit1::m - unit2::m, unit1::l - unit2::l, unit1::t - unit2::t,
                          unit1::T - unit2::T, unit1::i - unit2::i, typename unit1::scalar, typename unit1::unit_base>;
    };
    template <typename unit1>
    struct inverse_unit {
        using type =
            unit<-unit1::m, -unit1::l, -unit1::t, -unit1::T, -unit1::i, typename unit1::scalar, typename unit1::unit_base>;
    };
    template <typename unit1>
    struct squared_unit {
        using type = unit<2 * unit1::m, 2 * unit1::l, 2 * unit1::t, 2 * unit1::T, 2 * unit1::i,
                          typename unit1::scalar, typename unit1::unit_base>;
    };
    template <typename unit1>
    struct cubed_unit {
        using type = unit<3 * unit1::m, 3 * unit1::l, 3 * unit1::t, 3 * unit1::T, 3 * unit1::i,
                          typename unit1::scalar, typename unit1::unit_base>;
    };
    template <typename T1, typename T2>
    using unit_product_t = typename unit_product<T1, T2>::type;
    template <typename T1, typename T2>
    using unit_quotient_t = typename unit_quotient<T1, T2>::type;
    template <typename T>
    using inverse_unit_t = typename inverse_unit<T>::type;
    template <typename T>
    using squared_unit_t = typename squared_unit<T>::type;
    template <typename T>
    using cubed_unit_t = typename cubed_unit<T>::type;

    template<typename T = double, typename base = SI_base>
    using mass = unit<1, 0, 0, 0, 0, T, base>;
    template<typename T = double, typename base = SI_base>
    using length = unit<0, 1, 0, 0, 0, T, base>;
    template<typename T = double, typename base = SI_base>
    using time = unit<0, 0, 1, 0, 0, T, base>;
    template<typename T = double, typename base = SI_base>
    using velocity = unit_quotient_t<length<T, base>, time<T, base>>;
    template<typename T = double, typename base = SI_base>
    using temperature = unit<0, 0, 0, 1, 0, T, base>;
    template<typename T = double, typename base = SI_base>
    using current = unit<0, 0, 0, 0, 1, T, base>;
    template<typename T = double, typename base = SI_base>
    using charge = unit<0, 0, 1, 0, 1, T, base>;
    template<typename T = double, typename base = SI_base>
    using force = unit_product_t<mass<T, base>, unit_product_t<length<T, base>, inverse_unit_t<squared_unit_t<time<T, base>>>>>;
    template<typename T = double, typename base = SI_base>
    using acceleration = unit_quotient_t<force<T, base>, mass<T, base>>;
    template<typename T = double, typename base = SI_base>
    using pressure = unit_product_t<force<T, base>, inverse_unit_t<squared_unit_t<length<T, base>>>>;
    template<typename T = double, typename base = SI_base>
    using density = unit_quotient_t<mass<T, base>, cubed_unit_t<length<T, base>>>;

    namespace detail{
        template<typename T = double, typename base = SI_base>
        struct speed_of_light_type{
            using SI_unit = velocity<T, SI_base>;
            using unit = velocity<T, base>;
            KOKKOS_INLINE_FUNCTION constexpr static unit value(){
                SI_unit c_in_si = SI_unit(299792458);
                return c_in_si.template convert_to<base>();
            } 
        };

        template<typename T = double, typename base = SI_base>
        struct electron_charge_type{
            using SI_unit = charge<T, SI_base>;
            using unit = charge<T, base>;
            KOKKOS_INLINE_FUNCTION constexpr static unit value(){
                SI_unit c_in_si = SI_unit(-1.602176634e-19);
                return c_in_si.template convert_to<base>();
            } 
        };

        template<typename T = double, typename base = SI_base>
        struct epsilon0_type{
            using SI_unit = unit<-1, -3, 4, 0, 2, double, SI_base>;
            using unit = unit<-1, -3, 4, 0, 2, double, base>;
            KOKKOS_INLINE_FUNCTION constexpr static unit value(){
                SI_unit c_in_si = SI_unit(8.8543292936275999131525162995513463247709717789273358140421e-12);
                return c_in_si.template convert_to<base>();
            } 
        };

        
    }
    template<typename base = SI_base, typename T = double>
    constexpr auto speed_of_light(){
        return detail::speed_of_light_type<T, base>::value();
    }
    template<typename base = SI_base, typename T = double>
    constexpr auto electron_charge(){
        return detail::electron_charge_type<T, base>::value();
    }
    template<typename base = SI_base, typename T = double>
    constexpr auto epsilon0(){
        return detail::epsilon0_type<T, base>::value();
    }
    template <int m, int l, int t, int T, int i, typename scalar, typename base>
    std::ostream& operator<<(std::ostream& ost, const unit<m, l, t, T, i, scalar, base>& o) {
        using ut = unit<m, l, t, T, i, scalar, base>;
        if(std::is_same_v<ut, pressure<scalar, SI_base>>) {
            if (o.value > 1e8) {
                ost << o.value / 1e9 << " GPa";
            } else if (o.value >= 1e4) {
                ost << o.value / 1e6 << " MPa";
            } else {
                ost << o.value << " Pa";
            }

            return ost;
        }
        //std::cout << "CGS PRESCHER: " << pressure<scalar, cgs_base>(1.0) << "\n\n";
        if(std::is_same_v<ut, pressure<scalar, cgs_base>>) {
            if (o.value > 1e8) {
                ost << o.value / 1e9 << " GBarye";
            } else if (o.value >= 1e4) {
                ost << o.value / 1e6 << " MBarye";
            } else {
                ost << o.value << " Barye";
            }

            return ost;
        }

        ost << o.value;
        bool printstar = false;
        if (m != 0) {
            if constexpr(std::is_same_v<base, SI_base>)
                ost << "kg";
            if constexpr(std::is_same_v<base, cgs_base>)
                ost << "g";
            if constexpr(std::is_same_v<base, planck_base>)
                ost << "planck-mass";
            if (m != 1) {
                ost << "^" << m;
            }
            printstar = true;
        }

        if (l != 0) {
            if (printstar)
                ost << "*";
            if constexpr(std::is_same_v<base, SI_base>)
                ost << "m";
            if constexpr(std::is_same_v<base, cgs_base>)
                ost << "cm";
            if constexpr(std::is_same_v<base, planck_base>)
                ost << "planck-length";
            if (l != 1) {
                ost << "^" << l;
            }
            printstar = true;
        }

        if (t != 0) {
            if (printstar)
                ost << "*";
            if constexpr(std::is_same_v<base, SI_base>)
                ost << "s";
            if constexpr(std::is_same_v<base, cgs_base>)
                ost << "s";
            if constexpr(std::is_same_v<base, planck_base>)
                ost << "planck-time";
            if (t != 1) {
                ost << "^" << t;
            }
            printstar = true;
        }

        if (T != 0) {
            if (printstar)
                ost << "*";
            if constexpr(std::is_same_v<base, SI_base>)
                ost << "K";
            if constexpr(std::is_same_v<base, cgs_base>)
                ost << "K";
            if constexpr(std::is_same_v<base, planck_base>)
                ost << "planck-temp";
            if (T != 1) {
                ost << "^" << T;
            }
            printstar = true;
        }

        if (i != 0) {
            if (printstar)
                ost << "*";
            if constexpr(std::is_same_v<base, SI_base>)
                ost << "A";
            if constexpr(std::is_same_v<base, cgs_base>)
                ost << "A";
            if constexpr(std::is_same_v<base, planck_base>)
                ost << "planck-current";
            if (i != 1) {
                ost << "^" << i;
            }
        }
        return ost;
    }

    template <typename T>
    KOKKOS_INLINE_FUNCTION auto millimeter(const T& x) {
        return length<T, SI_base>(x / 1000.0);
    }
    KOKKOS_INLINE_FUNCTION auto operator"" _m(const char* x) {
        double v = std::stod(x);
        return length<>(v);
    }
    KOKKOS_INLINE_FUNCTION auto operator"" _mm(const char* x) {
        double v = std::stod(x);
        return length<>(v / 1000.0);
    }
    KOKKOS_INLINE_FUNCTION auto operator"" _s(const char* x) {
        double v = std::stod(x);
        return time<double, SI_base>(v);
    }
    KOKKOS_INLINE_FUNCTION auto operator"" _ms(const char* x) {
        double v = std::stod(x);
        return time<double, SI_base>(v / 1000.0);
    }
    KOKKOS_INLINE_FUNCTION auto operator"" _kg(const char* x) {
        double v = std::stod(x);
        return mass<double, SI_base>(v);
    }
    KOKKOS_INLINE_FUNCTION auto operator"" _g(const char* x) {
        double v = std::stod(x);
        return mass<double, SI_base>(v / 1000.0);
    }
    KOKKOS_INLINE_FUNCTION auto operator"" _A(const char* x) {
        double v = std::stod(x);
        return current<double, SI_base>(v);
    }
    KOKKOS_INLINE_FUNCTION auto operator"" _C(const char* x) {
        double v = std::stod(x);
        return current<double, SI_base>(v) * 1_s;
    }
    KOKKOS_INLINE_FUNCTION auto operator"" _Pa(const char* x) {
        double v = std::stod(x);
        return pressure<double, SI_base>(v);
    }
    KOKKOS_INLINE_FUNCTION auto operator"" _MPa(const char* x) {
        double v = std::stod(x);
        return pressure<double, SI_base>(1e6 * v);
    }
    template <int _m, int _l, int _t, int _T, int _i, typename _scalar>
    std::string to_string(const unit<_m, _l, _t, _T, _i, _scalar>& o) {
        std::stringstream sstr;
        sstr << o;
        return sstr.str();
    }
}  // namespace funits
#endif
