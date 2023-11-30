//
// Class CIC
//   First order/cloud-in-cell grid interpolation. Currently implemented as
//   global functions, but in order to support higher or lower order interpolation,
//   these should be moved into structs.
//

namespace ippl {
    namespace detail {
        template <unsigned long Point, unsigned long Index, typename Weights>
        KOKKOS_INLINE_FUNCTION constexpr typename Weights::value_type interpolationWeight(
            const Weights& wlo, const Weights& whi) {
            if constexpr (Point & (1 << Index)) {
                return wlo[Index];
            } else {
                return whi[Index];
            }
            // device code cannot throw exceptions, but we need a
            // dummy return to silence the warning
            return 0;
        }

        template <unsigned long Point, unsigned long Index, typename Indices>
        KOKKOS_INLINE_FUNCTION constexpr typename Indices::value_type interpolationIndex(
            const Indices& args) {
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
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr void scatterToPoint(
            const std::index_sequence<Index...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args, const T& val) {
            bool isinbound = true;
            
            ippl::Vector<IndexType, View::rank> index3{interpolationIndex<ScatterPoint, Index>(args)...};
            for(unsigned int d = 0; d < View::rank;d++){
                isinbound &= (index3[d] < view.extent(d));
                isinbound &= (index3[d] >= 0);
            }
            if(!isinbound){
                //if(ippl::Comm->rank() == 0){
                //    std::cout << "scatter cancelled!!\n";
                //}
                return;
            }
            Kokkos::atomic_add(&view(interpolationIndex<ScatterPoint, Index>(args)...),
                               val * (interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...));
        }

        template <unsigned long... ScatterPoint, typename View, typename T, typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr void scatterToField(
            const std::index_sequence<ScatterPoint...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args, T val) {
            // The number of indices is equal to the view rank
            (scatterToPoint<ScatterPoint>(std::make_index_sequence<View::rank>{}, view, wlo, whi,
                                          args, val),
             ...);
        }

        template <unsigned long GatherPoint, unsigned long... Index, typename View, typename T,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr typename View::value_type gatherFromPoint(
            const std::index_sequence<Index...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args) {
                //LOG("    GD: " << (interpolationWeight<GatherPoint, Index>(wlo, whi) * ...) << " * " << view(interpolationIndex<GatherPoint, Index>(args)...) << "\n");
            return (interpolationWeight<GatherPoint, Index>(wlo, whi) * ...)
                   * view(interpolationIndex<GatherPoint, Index>(args)...);
        }

        template <unsigned long... GatherPoint, typename View, typename T, typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr typename View::value_type gatherFromField(
            const std::index_sequence<GatherPoint...>&, const View& view,
            const Vector<T, View::rank>& wlo, const Vector<T, View::rank>& whi,
            const Vector<IndexType, View::rank>& args) {
            // The number of indices is equal to the view rank
            return (gatherFromPoint<GatherPoint>(std::make_index_sequence<View::rank>{}, view, wlo,
                                                 whi, args)
                    + ...);
        }
        template <unsigned long Point, unsigned long Index, typename IndexType, unsigned Dim>
        KOKKOS_INLINE_FUNCTION constexpr IndexType zigzag_interpolationIndex(
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
                                            /* Index is in [0, Dim[  */
        template <unsigned long ScatterPoint, unsigned long... Index, typename T, unsigned Dim,
                  typename IndexType>
        KOKKOS_INLINE_FUNCTION constexpr int zigzag_scatterToPoint(
            const std::index_sequence<Index...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, 3>, Dim>::view_type& view,
            const Vector<T, Dim>& wlo, const Vector<T, Dim>& whi,
            const Vector<IndexType, Dim>& args, const Vector<T, Dim>& val, T scale) {
            //std::cout << args[0] << " " << args[1] << " " << args[2] << std::endl;
            //assert(((zigzag_interpolationIndex<ScatterPoint, Index>(args) < view.extent(0)) && ...));
            bool isinbound = true;
            ippl::Vector<T, Dim> depot = scale * val * (zigzag_interpolationWeight<ScatterPoint, Index>(wlo, whi) * ...);
            ippl::Vector<IndexType, Dim> index3{zigzag_interpolationIndex<ScatterPoint, Index>(args)...};
            for(unsigned int d = 0; d < Dim;d++){
                isinbound &= (index3[d] < view.extent(d));
            }
            if(!isinbound){
                //if(ippl::Comm->rank() == 0){
                //    std::cout << "scatter cancelled!!\n";
                //}
                return 0;
            }
            //LOG("First_depot: " << depot);
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
        template<typename T>
        KOKKOS_INLINE_FUNCTION T fractional_part(T x){
            using Kokkos::floor;
            //using ::floor;
            return x - floor(x);
        }
        template <unsigned long... ScatterPoint, typename T, unsigned Dim, typename IndexType>
        KOKKOS_INLINE_FUNCTION void zigzag_scatterToField(
            const std::index_sequence<ScatterPoint...>&,
            const typename ippl::detail::ViewType<ippl::Vector<T, 3>, Dim>::view_type& view,
            Vector<T, Dim> from, Vector<T, Dim> to,
            const Vector<T, Dim> hr, T scale, const NDIndex<Dim> lDom, int nghost) {
            
            // Define utility functions

            
            using Kokkos::max;
            using Kokkos::min;
            using Kokkos::floor;
            //using ::max;
            //using ::min;
            //using ::floor;
            Vector<T, Dim> from_in_grid_coordinates;
            Vector<T, Dim> to_in_grid_coordinates;
            // Calculate the indices for the scatter operation
            ippl::Vector<IndexType, Dim> fromi, toi;
            for (unsigned int i = 0; i < Dim; i++) {
                from[i] += hr[i] * T(0.5);
                to[i] += hr[i] * T(0.5);
                from_in_grid_coordinates[i] = from[i] / hr[i];
                to_in_grid_coordinates  [i] = to[i] / hr[i];
                fromi[i] = floor(from_in_grid_coordinates[i]) + nghost;
                toi[i]   = floor(to_in_grid_coordinates[i])   + nghost;
            }
            //LOG("fromi and toi: " << fromi << toi);
            ippl::Vector<IndexType, Dim> fromi_local = fromi - lDom.first();
            ippl::Vector<IndexType, Dim> toi_local = toi - lDom.first();
            /*if(ippl::Comm->rank() == 0){
                ippl::Vector<IndexType, Dim> extentvec{view.extent(0), view.extent(1), view.extent(2)};
                LOG(extentvec);
                LOG(fromi_local);
                LOG(toi_local << "\n\n");
            }*/
            // Calculate the relay point for each dimension
            ippl::Vector<T, Dim> relay;
            for (unsigned int i = 0; i < Dim; i++) {
                relay[i] =
                min(
                    min(fromi[i] - nghost, toi[i] - nghost) * hr[i] + hr[i],
                    max(
                        max(fromi[i] - nghost, toi[i] - nghost) * hr[i],
                        T(0.5) * (to[i] + from[i])
                    )
                );
            }
            //ippl::Vector<T, Dim> relay_local = relay - lDom.first();
            
            // Calculate jcfrom and jcto
            ippl::Vector<T, Dim> jcfrom, jcto;
            jcfrom = relay;
            jcfrom -= from;
            //LOG("JCFROM: " << jcfrom);
            jcto = to;
            jcto -= relay;
            for(int i = 0;i < 3;i++){
                if(std::signbit(jcfrom[i]) != std::signbit(jcto[i])){
                    std::cerr << "Violation\n" << from << relay << to << "\n";
                    //abort();
                }
            }
            //LOG("JCTO: " << jcfrom);
            //LOG("From to Relay: " << from << to << relay);
            //LOG("Scale: " << scale);
            //std::cout << ippl::Comm->rank() << jcfrom << "   \t" << jcto << "\n";
            //std::cout << lDom.first() << "  ldf\n";
            // Calculate wlo and whi
            Vector<T, Dim> wlo, whi;
            for (unsigned i = 0; i < Dim; i++) {
                wlo[i] = T(1.0) - fractional_part((from[i] + relay[i]) * T(0.5) / hr[i]);
                whi[i] = fractional_part((from[i] + relay[i]) * T(0.5) / hr[i]);
            }
            //LOG("Current weights: " << whi);
            // Perform the scatter operation for each scatter point
            auto _ =
                (zigzag_scatterToPoint<ScatterPoint>(std::make_index_sequence<Dim>{}, view, wlo,
                                                     whi, fromi_local, jcfrom, scale)
                 ^ ...);
            
            for (unsigned i = 0; i < Dim; i++) {
                wlo[i] = T(1.0) - fractional_part((to[i] + relay[i]) * T(0.5) / hr[i]);
                whi[i] = fractional_part((to[i] + relay[i]) * T(0.5) / hr[i]);
            }

            //LOG("Current weights: " << whi);
            
            auto __ =
                (zigzag_scatterToPoint<ScatterPoint>(std::make_index_sequence<Dim>{}, view, wlo,
                                                     whi, toi_local, jcto, scale)
                 ^ ...);

            (void)_;
            (void)__; // [[maybe_unused]] causes issues on certain compilers
        }
    }  // namespace detail
}  // namespace ippl
