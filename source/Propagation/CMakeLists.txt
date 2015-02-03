include(${ITK_USE_FILE})

find_package(STATISMO REQUIRED)
set (STATISMO_INCLUDE_DIR ${STATISMO_DIR}/include)
set(MATLAB_ROOT /usr/pack/matlab-8.3r2014a-fg/ )
#message(${CMAKE_MODULE_PATH})
find_package(MATLAB REQUIRED)

INCLUDE_DIRECTORIES(
    /usr/include/libxml2
    ../External/RF
    ../External/c-ugmix
    ../Common
    /home/gasst/work/src/statismoREADONLY/build/include/statismo_ITK
    /home/gasst/work/src/statismoREADONLY/build/include/Representers/ITK/
    ${STATISMO_INCLUDE_DIR} ${STATISMO_INCLUDE_DIR}/statismo_ITK  ${STATISMO_INCLUDE_DIR}/3rdParty  ${STATISMO_INCLUDE_DIR}  
    ${STATISMO_INCLUDE_DIR}/Representers/ITK
 )




