add_executable(assign)
target_sources(assign PRIVATE assign.cxx)
target_link_libraries(assign PRIVATE benchmark::benchmark std::mdspan)
