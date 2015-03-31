PROJECT(ConsitencyBasedRegistrationRectification)
INCLUDE_REGULAR_EXPRESSION("^.*$")


#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/Matlab/")
#set(MATLAB_ROOT "/usr/pack/matlab-8.3r2014a-fg/" CACHE  FILEPATH "Matlab root directory" FORCE)  
set(MATLAB_ROOT "/usr/pack/matlab-7.13r2011b-sd/" CACHE  FILEPATH "Matlab root directory" FORCE)  
find_package(MATLAB REQUIRED)

if (NOT EXISTS ${MATLAB_ROOT})
set(MATLAB_ROOT "NOTFOUND" CACHE  FILEPATH "Matlab root directory" FORCE)  
endif()

if ( ${MATLAB_ROOT} MATCHES "NOTFOUND" )
  message(SEND_ERROR "MATLAB not found, please enter MATLAB_ROOT and try again")  
 
 
endif()

add_library(eng SHARED IMPORTED) 
set_property(TARGET eng PROPERTY IMPORTED_LOCATION  ${MATLAB_ENG_LIBRARY})
add_library(mx SHARED IMPORTED) 
set_property(TARGET mx PROPERTY IMPORTED_LOCATION ${MATLAB_MX_LIBRARY})

option( USE_MIND "Use MIND local similarity functions" OFF )
if( ${USE_MIND} MATCHES "ON" )
  add_definitions(-DWITH_MIND)
  set(DIR_MIND "../External/MIND" CACHE  FILEPATH "Directory for MIND")
  include_directories( ${DIR_MIND} ) 
endif()

#Aquirc like stuff
message( "${MATLAB_ROOT} ${MATLAB_ENG_LIBRARY}  ${MATLAB_MX_LIBRARY} ${MATLAB_INCLUDE_DIR} ")
include_directories(${MATLAB_INCLUDE_DIR})

ADD_EXECUTABLE(AquircGlobalNorm2D AquircGlobalNorm2D.cxx )
TARGET_LINK_LIBRARIES(AquircGlobalNorm2D    ${ITK_LIBRARIES} mx eng  )
ADD_EXECUTABLE(AquircLocalErrors2D AquircLocalError2D.cxx )
TARGET_LINK_LIBRARIES(AquircLocalErrors2D   ${ITK_LIBRARIES} mx eng  )

ADD_EXECUTABLE(CBRR2D CBRR2D.cxx )
TARGET_LINK_LIBRARIES(CBRR2D Utils       eng mx ${ITK_LIBRARIES} )
ADD_EXECUTABLE(CBRR3D CBRR3D.cxx )
TARGET_LINK_LIBRARIES(CBRR3D Utils      ${ITK_LIBRARIES} mx eng )
