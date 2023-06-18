cmake_minimum_required(VERSION 3.13)

project(vectorclass VERSION 1)

add_library(vectorclass vectorclass/instrset_detect.cpp)
target_include_directories(vectorclass PRIVATE vectorclass)
target_include_directories(vectorclass INTERFACE .)