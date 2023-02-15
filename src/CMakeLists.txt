add_executable(assign)
target_sources(assign PRIVATE assign.cxx)
target_link_libraries(assign PRIVATE common benchmark::benchmark std::mdspan)

if (SYCL_BENCH_ENABLE_GTENSOR)
  add_executable(assign_gt)
  target_sources(assign_gt PRIVATE assign_gt.cxx)
  target_link_libraries(assign_gt PRIVATE
                        common benchmark::benchmark gtensor::gtensor)
endif()
