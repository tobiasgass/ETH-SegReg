PROJECT(ConsitencyBasedRegistrationRectification)
INCLUDE_REGULAR_EXPRESSION("^.*$")


#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/Matlab/")
#set(MATLAB_ROOT "/usr/pack/matlab-8.3r2014a-fg/bin/glnxa64" CACHE  FILEPATH "Matlab root directory" FORCE)  
#find_package(MATLAB REQUIRED)

if ( ${MATLAB_ENG_LIBRARY} MATCHES "NOTFOUND" )
  message(SEND_ERROR "MATLAB not found, please enter MATLAB_ROOT and try again")  
  #set(MATLAB_ROOT "NOTFOUND" CACHE  FILEPATH "Matlab root directory" FORCE) 
 
endif()

option( USE_MIND "Use MIND local similarity functions" OFF )
if( ${USE_MIND} MATCHES "ON" )
  add_definitions(-DWITH_MIND)
  set(DIR_MIND "../External/MIND" CACHE  FILEPATH "Directory for MIND")
  include_directories( ${DIR_MIND} ) 
endif()

#Aquirc like stuff
#message( ${MATLAB_ENG_LIBRARY}  ${MATLAB_MX_LIBRARY})
include_directories(${MATLAB_INCLUDE_DIR})

ADD_EXECUTABLE(AquircGlobalNorm2D AquircGlobalNorm2D.cxx )
TARGET_LINK_LIBRARIES(AquircGlobalNorm2D    ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )
ADD_EXECUTABLE(AquircLocalErrors2D AquircLocalError2D.cxx )
TARGET_LINK_LIBRARIES(AquircLocalErrors2D   ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )

ADD_EXECUTABLE(CBRR2D CBRR2D.cxx )
TARGET_LINK_LIBRARIES(CBRR2D Utils      ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )
ADD_EXECUTABLE(CBRR3D CBRR3D.cxx )
TARGET_LINK_LIBRARIES(CBRR3D Utils      ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )
