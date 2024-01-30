//
// Class Variant
//   Variant class used for Variant fields and particle attributes like the coordinate.
//
#ifndef IPPL_Variant_H
#define IPPL_Variant_H


#include <cassert>
#include <cstring>
#include <iostream>
namespace ippl {
    /*!
     * @file Variant.h
     */
    using std::size_t;
    template<typename... Ts>
    struct maxSizeImpl;
    template<typename T, typename... Rs>
    struct maxSizeImpl<T, Rs...>{
        constexpr static size_t value = sizeof(T) > maxSizeImpl<Rs...>::value ? sizeof(T) : maxSizeImpl<Rs...>::value;
    };
    template<>
    struct maxSizeImpl<>{
        constexpr static size_t value = 0;
    };
    template<typename... Ts>
    constexpr size_t maxSize = maxSizeImpl<Ts...>::value;

    template<size_t Idx, typename... Ts>
    struct matchingTypeImpl;
    template<size_t Idx, typename T, typename C, typename... Cs>
    struct matchingTypeImpl<Idx, T, C, Cs...>{
        constexpr static size_t value = std::is_same_v<T, C> ? Idx : matchingTypeImpl<Idx + 1, Cs...>::value;
    };
    template<size_t Idx, typename T, typename C>
    struct matchingTypeImpl<Idx, T, C>{
        constexpr static size_t value = Idx;
        //static_assert(std::is_same_v<T, C>, "None matches");
    };
    template<size_t Idx, typename T>
    struct matchingTypeImpl<Idx, T>{
        constexpr static size_t value = Idx;
        //static_assert(std::is_same_v<T, C>, "None matches");
    };
    template<typename C, typename... Cs>
    size_t matchingType = matchingTypeImpl<0, C, Cs...>::value;

    template<size_t Idx, typename T, typename... Ts>
    struct extractType;

    template<typename T, typename... Ts>
    struct extractType<0, T, Ts...> {
        using type = T;
    };

    template<size_t Idx, typename T, typename... Ts>
    struct extractType {
        static_assert(Idx < 1 + sizeof...(Ts), "Index out of bounds");
        using type = typename extractType<Idx - 1, Ts...>::type;
    };
    /*!
     * @class Variant
     * @tparam Ts... intrinsic Variant data type
     */
    template <typename... Ts>
    class alignas(32) Variant {
    public:
        Variant()
        requires(std::is_default_constructible_v<Ts> && ...) : index_m(0) {
            new (buffer_m) typename extractType<0, Ts...>::type();
        }
        Variant()
        requires(!(std::is_default_constructible_v<Ts> && ...)) : index_m(invalidated), buffer_m{0}{}
        Variant(const Variant<Ts...>& x) : Variant(){
            index_m = x.index_m;
        }
        Variant(Variant<Ts...>&& x) : index_m(invalidated){
            constructFromVariant(std::forward<Variant<Ts...>>(x));
            x.index_m = invalidated;
        }

        template<typename T>
        Variant(T&& x){
            tryConstruct<0>(std::forward<T>(x));
        }
        template<typename T>
        Variant& operator=(T&& x){
            destruct();
            tryConstruct<0>(std::forward<T>(x));
            return *this;
        }
        template<size_t TIdx, typename T>
        void tryConstruct(T&& x){
            static_assert(TIdx < sizeof...(Ts), "Invalid variant constructor argument");
            if constexpr(std::is_same_v<T, typename extractType<TIdx, Ts...>::type>){
                std::cout << "Constructing " << TIdx << "\n";
                index_m = TIdx;
                new (buffer_m) T(std::forward<T>(x));
            }
            else{
                tryConstruct<TIdx + 1>(std::forward<T>(x));
            }
        }
        template<size_t Idx>
            //requires(Idx < sizeof...(Ts))
        const auto& get()const noexcept{
            assert(Idx == index_m);
            return *reinterpret_cast<const typename extractType<Idx, Ts...>::type*>(buffer_m);

        }
        
        ~Variant(){
            destruct();
        }

    public:
        /**
         * @brief Destructs the given variant
         * 
         */
        void destruct(){
            //If already invalidated, don't destruct again
            if(index_m == invalidated){
                return;
            }
            

            auto dtor_caller = []<size_t Idx>(size_t indexm, char* buffer){
                if(indexm == Idx){
                    using destructedType = typename extractType<Idx, Ts...>::type;
                    (*reinterpret_cast<typename extractType<Idx, Ts...>::type*>(buffer)).~destructedType();
                }
                return 0;
            };
            auto dtor_runner = []<size_t... Idx>(auto& dtc, size_t indexm, char* buffer, const std::index_sequence<Idx...>& seq){
                (void)seq;
                int x = (dtc.template operator()<Idx>(indexm, buffer) ^ ...);
                return x;
            };
            int x = dtor_runner(dtor_caller, index_m, buffer_m, std::make_index_sequence<sizeof...(Ts)>{});
            (void)x;
            index_m = invalidated;
        }
        void constructFromVariant(Variant<Ts...>&& var){
            //If already invalidated, don't destruct again
            if(var.index_m == invalidated){
                return;
            }
            

            auto ctor_caller = []<size_t Idx>(size_t indexm, char* buffer, char* obuffer){
                if(indexm == Idx){
                    using constructedType = typename extractType<Idx, Ts...>::type;
                    new (buffer) constructedType(std::move(*reinterpret_cast<constructedType*>(obuffer)));
                }
                return 0;
            };
            auto ctor_runner = []<size_t... Idx>(auto& dtc, size_t indexm, char* buffer, char* obuffer, const std::index_sequence<Idx...>& seq){
                (void)seq;
                int x = (dtc.template operator()<Idx>(indexm, buffer, obuffer) ^ ...);
                return x;
            };
            int x = ctor_runner(ctor_caller, var.index_m, buffer_m, var.buffer_m, std::make_index_sequence<sizeof...(Ts)>{});
            (void)x;
            index_m = invalidated;
        }
        constexpr static size_t msize = maxSize<Ts...>;
        constexpr static size_t invalidated = size_t(-1);
        
        size_t index_m;
        char buffer_m[msize];
    };
}  // namespace ippl

//#include "Variant.hpp"

#endif
