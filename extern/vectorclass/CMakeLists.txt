cmake_minimum_required(VERSION 3.13)

project(vectorclass VERSION 2)

add_library(vectorclass vectorclass/instrset_detect.cpp)
target_compile_features(vectorclass PUBLIC cxx_std_17)
target_include_directories(vectorclass PRIVATE vectorclass)
target_include_directories(vectorclass INTERFACE .)
