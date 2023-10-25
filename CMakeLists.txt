cmake_minimum_required(VERSION 3.3)

project(hnswlib VERSION "1.0.0" LANGUAGES CXX)

add_library(hnswlib INTERFACE)
target_include_directories ( hnswlib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hnswlib>$<INSTALL_INTERFACE:include> )

set ( EXPORT_CMAKE_DIR "lib/cmake/hnswlib" )

install(
    FILES
        ${PROJECT_SOURCE_DIR}/hnswlib/bruteforce.h
		${PROJECT_SOURCE_DIR}/hnswlib/hnswalg.h
		${PROJECT_SOURCE_DIR}/hnswlib/hnswlib.h
		${PROJECT_SOURCE_DIR}/hnswlib/space_ip.h
		${PROJECT_SOURCE_DIR}/hnswlib/space_l2.h
		${PROJECT_SOURCE_DIR}/hnswlib/visited_list_pool.h
    DESTINATION include
)

install ( TARGETS hnswlib EXPORT hnswlibexport ARCHIVE DESTINATION "lib/$<CONFIG>" )
#install ( FILES "$<TARGET_FILE_DIR:hnswlib>/hnswlib.pdb" EXPORT hnswlibexport DESTINATION "lib/$<CONFIG>" OPTIONAL )
install ( EXPORT hnswlibexport FILE "hnswlibTargets.cmake" DESTINATION "${EXPORT_CMAKE_DIR}" NAMESPACE "hnswlib::" )

include ( CMakePackageConfigHelpers )

set ( pkgconfin "${CMAKE_CURRENT_BINARY_DIR}/hnswlibConfig.cmake.in" )
file ( WRITE "${pkgconfin}" "@PACKAGE_INIT@

if(NOT TARGET hnswlib::hnswlib)
    include(\"\${CMAKE_CURRENT_LIST_DIR}/hnswlibTargets.cmake\")
    get_target_property ( configs hnswlib::hnswlib IMPORTED_CONFIGURATIONS )
        foreach (config RELWITHDEBINFO)
		if (NOT \${config} IN_LIST configs)
			set_property ( TARGET hnswlib::hnswlib PROPERTY MAP_IMPORTED_CONFIG_\${config} Release )
		endif ()
	endforeach ()
endif()" )

configure_package_config_file ( "${pkgconfin}" "${CMAKE_CURRENT_BINARY_DIR}/hnswlibConfig.cmake" INSTALL_DESTINATION "${EXPORT_CMAKE_DIR}" )
install ( FILES "${CMAKE_CURRENT_BINARY_DIR}/hnswlibConfig.cmake" DESTINATION "${EXPORT_CMAKE_DIR}" )

write_basic_package_version_file ( "hnswlibConfigVersion.cmake" COMPATIBILITY ExactVersion )
install ( FILES "${CMAKE_CURRENT_BINARY_DIR}/hnswlibConfigVersion.cmake" DESTINATION "${EXPORT_CMAKE_DIR}" )
