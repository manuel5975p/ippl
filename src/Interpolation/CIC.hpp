//
// Class CIC
//   First order/cloud-in-cell grid interpolation. Currently implemented as
//   global functions, but in order to support higher or lower order interpolation,
//   these should be moved into structs.
//
// Copyright (c) 2023, Paul Scherrer Institut, Villigen PSI, Switzerland
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

namespace ippl {
    namespace detail {
        template <unsigned long Point, unsigned long Index, typename T, unsigned Dim>
        KOKKOS_INLINE_FUNCTION constexpr T interpolationWeight(const Vector<T, Dim>& wlo,
                                                               const Vector<T, Dim>& whi) {
            if constexpr (Point & (1 << Index)) {
                return wlo[Index];
            } else {
                return whi[Index];
            }
            // device code cannot throw exceptions, but we need a
            // dummy return to silence the warning
            return 0;
        }

        template <unsigned long Point, unsigned long Index, typename IndexType, unsigned Dim>
        KOKKOS_INLINE_FUNCTION constexpr IndexType interpolationIndex(
            const Vector<IndexType, Dim>& args) {
            if constexpr (Point & (1 << Index)) {
                return args[Index] - 1;
            } else {
                return args[Index];
            }
            // device code cannot throw exceptions, but we need a
            // dummy return to silence the warning
            return 0;
        }

        template <unsigned long ScatterPoint, unsigned long... Index, typename View, typename T,
                  unsigned Dim, typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr int scatterToPoint(
            const std::index_sequence<Index...>&, const View& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args, const T& val) {
            Kokkos::atomic_add(&view(interpolationIndex<ScatterPoint, Index>(args)...),
                               val * (interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...));
            return 0;
        }

        template <unsigned long... ScatterPoint, typename View, typename T, unsigned Dim,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr void scatterToField(
            const std::index_sequence<ScatterPoint...>&, const View& view,
            const Vector<T, Dim>& wlo, const Vector<T, Dim>& whi,
            const Vector<IndexType, Dim>& args, T val) {
            // The number of indices is Dim
            [[maybe_unused]] auto _ = (scatterToPoint<ScatterPoint>(std::make_index_sequence<Dim>{},
                                                                    view, wlo, whi, args, val)
                                       ^ ...);
        }

        template <unsigned long GatherPoint, unsigned long... Index, typename View, typename T,
                  unsigned Dim, typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr T gatherFromPoint(const std::index_sequence<Index...>&,
                                                           const View& view,
                                                           const Vector<T, Dim>& wlo,
                                                           const Vector<T, Dim>& whi,
                                                           const Vector<IndexType, Dim>& args) {
            return (interpolationWeight<GatherPoint, Index>(wlo, whi) * ...)
                   * view(interpolationIndex<GatherPoint, Index>(args)...);
        }

        template <unsigned long... GatherPoint, typename View, typename T, unsigned Dim,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr T gatherFromField(
            const std::index_sequence<GatherPoint...>&, const View& view, const Vector<T, Dim>& wlo,
            const Vector<T, Dim>& whi, const Vector<IndexType, Dim>& args) {
            // The number of indices is Dim
            return (
                gatherFromPoint<GatherPoint>(std::make_index_sequence<Dim>{}, view, wlo, whi, args)
                + ...);
        }
        template <unsigned long Point, unsigned long Index, typename IndexType, unsigned Dim>
        KOKKOS_INLINE_FUNCTION constexpr IndexType zigzag_interpolationIndex(
            const Vector<IndexType, Dim>& args) {
            if constexpr (Point & (1 << Index)) {
                return args[Index] + 1;
            } else {
                return args[Index];
            }
            // device code cannot throw exceptions, but we need a
            // dummy return to silence the warning
            return 0;
        }
        /**
         * @brief Calculate a weight for zigzag scattering.
         *
         *
         * @tparam Point A special identifier that affects the weight calculation.
         * @tparam Index Another identifier used to determine the weight.
         * @tparam T The type of values used for calculations, like numbers.
         * @tparam Dim The number of dimensions of the particle
         * @param wlo The position information for the lower bound of data points.
         * @param whi The position information for the upper bound of data points.
         * @return The calculated weight, indicating how far a data point should move.
         *
         */
        template <unsigned long Point, unsigned long Index, typename T, unsigned Dim>
        KOKKOS_INLINE_FUNCTION constexpr T zigzag_interpolationWeight(const Vector<T, Dim>& wlo,
                                                                      const Vector<T, Dim>& whi) {
            if constexpr (Point & (1 << Index)) {
                return wlo[Index];
            } else {
                return whi[Index];
            }
            // device code cannot throw exceptions, but we need a
            // dummy return to silence the warning
        }

        template <unsigned long ScatterPoint, unsigned long... Index, typename T, unsigned Dim,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr int zigzag_scatterToPoint(
            const std::index_sequence<Index...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, 3>, Dim>::view_type& view,
            const Vector<T, Dim>& wlo, const Vector<T, Dim>& whi,
            const Vector<IndexType, Dim>& args, const Vector<T, Dim>& val, T scale) {
            //std::cout << args[0] << " " << args[1] << " " << args[2] << std::endl;
            //assert(((zigzag_interpolationIndex<ScatterPoint, Index>(args) < view.extent(0)) && ...));
            if(!((zigzag_interpolationIndex<ScatterPoint, Index>(args) < view.extent(0)) && ...))
                return 0;
            typename ippl::detail::ViewType<ippl::Vector<T, Dim>,
                                            Dim>::view_type::value_type::value_type* destptr =
                &(view(zigzag_interpolationIndex<ScatterPoint, Index>(args)...)[0]);
            Kokkos::atomic_add(
                destptr,
                scale * val[0] * (zigzag_interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...));
            destptr = &(view(zigzag_interpolationIndex<ScatterPoint, Index>(args)...)[1]);
            Kokkos::atomic_add(
                destptr,
                scale * val[1] * (zigzag_interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...));
            destptr = &(view(zigzag_interpolationIndex<ScatterPoint, Index>(args)...)[2]);
            Kokkos::atomic_add(
                destptr,
                scale * val[2] * (zigzag_interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...));

            return 0;
        }

        /**
         * This function performs a zigzag scatter operation from a field to a view.
         * 
         * @tparam ScatterPoint The scatter points to be used.
         * @tparam T The data type of the field and view.
         * @tparam Dim The dimensionality of the field and view.
         * @tparam IndexType The index type used for indexing.
         * @param view The view to scatter into.
         * @param from The starting position of the scatter operation.
         * @param to The ending position of the scatter operation.
         * @param hr The grid spacing.
         * @param scale The scaling factor to apply during the scatter operation. This is usually set to the inverse of the timestep.
         */
        template <unsigned long... ScatterPoint, typename T, unsigned Dim, typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr void zigzag_scatterToField(
            const std::index_sequence<ScatterPoint...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, 3>, Dim>::view_type& view,
            Vector<T, Dim> from, Vector<T, Dim> to,
            const Vector<T, Dim>& hr, T scale, const NDIndex<Dim> lDom, int nghost) {
            
            // Define utility functions
            using Kokkos::floor;
            using Kokkos::max;
            using Kokkos::min;
            constexpr auto fractional_part = KOKKOS_LAMBDA(double x) {
                return x - floor(x); // Return the fractional part of x value between 0 and 1
            };
            Vector<T, Dim> from_in_grid_coordinates;
            Vector<T, Dim> to_in_grid_coordinates;
            // Calculate the indices for the scatter operation
            ippl::Vector<IndexType, Dim> fromi, toi;
            for (unsigned int i = 0; i < Dim; i++) {
                from[i] += hr[i] * 0.5;
                to[i] += hr[i] * 0.5;
                from_in_grid_coordinates[i] = from[i] / hr[i];
                to_in_grid_coordinates  [i] = to[i] / hr[i];
                fromi[i] = Kokkos::floor(from_in_grid_coordinates[i]);
                toi[i]   = Kokkos::floor(to_in_grid_coordinates[i]);
            }
            ippl::Vector<IndexType, Dim> fromi_local = fromi - lDom.first();
            ippl::Vector<IndexType, Dim> toi_local = toi - lDom.first();

            // Calculate the relay point for each dimension
            ippl::Vector<T, Dim> relay;
            for (unsigned int i = 0; i < Dim; i++) {
                relay[i] =
                min(
                    min(fromi[i], toi[i]) * hr[i] + hr[i],
                    max(
                        max(fromi[i], toi[i]) * hr[i],
                        0.5 * (to[i] + from[i])
                    )
                );
            }
            ippl::Vector<T, Dim> relay_local = relay - lDom.first();

            // Calculate jcfrom and jcto
            ippl::Vector<T, Dim> jcfrom, jcto;
            jcfrom = relay;
            jcfrom -= from;
            jcto = to;
            jcto -= relay;
            //std::cout << ippl::Comm->rank() << jcfrom << "   \t" << jcto << "\n";
            //std::cout << lDom.first() << "  ldf\n";
            // Calculate wlo and whi
            Vector<T, Dim> wlo, whi;
            for (unsigned i = 0; i < Dim; i++) {
                wlo[i] = 1.0 - fractional_part((from[i] + relay[i]) * 0.5 / hr[i]);
                whi[i] = fractional_part((from[i] + relay[i]) * 0.5 / hr[i]);
            }

            // Perform the scatter operation for each scatter point
            for(unsigned i = 0;i < Dim;i++){
                assert(fromi_local[i] < view.extent(i));
                assert(toi_local[i] < view.extent(i));
            }
            //return;
            [[maybe_unused]] auto _ =
                (zigzag_scatterToPoint<ScatterPoint>(std::make_index_sequence<Dim>{}, view, wlo,
                                                     whi, fromi_local, jcfrom, scale)
                 ^ ...);
            for (unsigned i = 0; i < Dim; i++) {
                wlo[i] = 1.0 - fractional_part((to[i] + relay[i]) * 0.5 / hr[i]);
                whi[i] = fractional_part((to[i] + relay[i]) * 0.5 / hr[i]);
            }
            [[maybe_unused]] auto __ =
                (zigzag_scatterToPoint<ScatterPoint>(std::make_index_sequence<Dim>{}, view, wlo,
                                                     whi, toi_local, jcto, scale)
                 ^ ...);
        }
    }  // namespace detail
}  // namespace ippl
