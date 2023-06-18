target_sources(mpboot PRIVATE
        modelgtr.cpp
        modelbin.cpp
        modeldna.cpp
        modelfactory.cpp
        modelnonrev.cpp
        modelprotein.cpp
        modelset.cpp
        modelsubst.cpp
        partitionmodel.cpp
        rategamma.cpp
        rategammainvar.cpp
        rateheterogeneity.cpp
        rateinvar.cpp
        ratemeyerdiscrete.cpp
        ratemeyerhaeseler.cpp
        ratekategory.cpp
        ratefree.cpp
        ratefreeinvar.cpp
        modelcodon.cpp
        modelmorphology.cpp
        )
#target_include_directories(model PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
#set(Eigen3_DIR "${PROJECT_SOURCE_DIR}/extern/eigen/cmake")
#find_package(Eigen3 3.2 REQUIRED NO_MODULE)
#target_link_libraries(model PRIVATE Eigen3::Eigen)
