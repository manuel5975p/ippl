#ifndef IPPL_BOUNDARY_DISPATH_H
#define IPPL_BOUNDARY_DISPATH_H
#include "Field/Field.h"
#include <type_traits>
#include <cstdint>
#include <concepts>
//template<typename T>
//concept callable_with_three_size_t = requires(T t){
//
//};
enum boundary_occlusion : uint32_t{
    x_min = 1, y_min = 2 , z_min = 4,
    x_max = 8, y_max = 16, z_max = 32,
};
std::string to_string(boundary_occlusion value) {
    std::string result;
    if (value & x_min)
        result += "x_min ";
    if (value & y_min)
        result += "y_min ";
    if (value & z_min)
        result += "z_min ";
    if (value & x_max)
        result += "x_max ";
    if (value & y_max)
        result += "y_max ";
    if (value & z_max)
        result += "z_max ";
    
    // Remove trailing space if there is one
    if (!result.empty() && result.back() == ' ')
        result.pop_back();
    return result;
}

/**
 * @brief Dispatches a lambda over all cells that are less than boundary_thickness away from the boundary
 * This includes ghost cells, so if boundary_thickness == 1 then only ghost cells are visited. This function is view based.
 * 
 * @param f THe field, whose view is used to extract extents
 * @param boundary_thickness Thickness of the boundary in cells. If 1, visits only ghost cells.
 * @param boundary_function Callable that will be invoked with indices lying on the boundary. Can take a fourth argument of type boundary_occlusion. 
 * @param internal_function Callable on the interior
 */

//template<typename field, typename boundary_callable, typename interior_callable>
//void lambda_dispatch(field f, const size_t boundary_thickness, boundary_callable&& boundary_function, interior_callable&& internal_function, size_t offset_from_view_boundary = 0){
//    ippl::Vector<size_t, 3> nr_m_minus_boundary_thickness;
//    for (unsigned int i = 0; i < 3; ++i)
//        nr_m_minus_boundary_thickness[i] = f.getView().extent(i) - boundary_thickness;
//    Kokkos::parallel_for(
//    "Dispatch with case distinction between boundary and interior", ippl::getRangePolicy(f.getView(), offset_from_view_boundary),
//    KOKKOS_LAMBDA(size_t i, size_t j, size_t k) {
//        const size_t i_noghost = i /*+ ldom[0].first()*/ - offset_from_view_boundary;
//        const size_t j_noghost = j /*+ ldom[1].first()*/ - offset_from_view_boundary;
//        const size_t k_noghost = k /*+ ldom[2].first()*/ - offset_from_view_boundary;
//        if(
//          i_noghost < boundary_thickness || i_noghost >= nr_m_minus_boundary_thickness[0]||
//          j_noghost < boundary_thickness || j_noghost >= nr_m_minus_boundary_thickness[1]||
//          k_noghost < boundary_thickness || k_noghost >= nr_m_minus_boundary_thickness[2]){
//            if constexpr(std::is_invocable_v<boundary_callable, size_t, size_t, size_t>)
//                boundary_function(i, j, k);
//            if constexpr(std::is_invocable_v<boundary_callable, size_t, size_t, size_t, boundary_occlusion>){
//                uint32_t code = uint32_t(i_noghost < boundary_thickness)/*<< 0*/|
//                                uint32_t(j_noghost < boundary_thickness)  << 1  |
//                                uint32_t(k_noghost < boundary_thickness)  << 2  |
//                                uint32_t(i_noghost >= nr_m_minus_boundary_thickness[0]) << 3 |
//                                uint32_t(j_noghost >= nr_m_minus_boundary_thickness[1]) << 4 |
//                                uint32_t(k_noghost >= nr_m_minus_boundary_thickness[2]) << 5;
//                boundary_occlusion occl = (boundary_occlusion)code;
//                boundary_function(i, j, k, occl);
//            }
//        }
//        else{
//            internal_function(i, j, k);
//        }
//    });
//}

#endif