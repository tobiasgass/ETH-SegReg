PROJECT(ConsitencyBasedRegistrationRectification)
INCLUDE_REGULAR_EXPRESSION("^.*$")


include_directories (${SRS-MRF_SOURCE_DIR}/Common) 



option( USE_MIND "Use MIND local similarity functions" ON )
if( ${USE_MIND} MATCHES "ON" )
  add_definitions(-DWITH_MIND)
  set(DIR_MIND "${SRS-MRF_SOURCE_DIR}/External/MIND" CACHE  FILEPATH "Directory for MIND")
  #  option( MIND_DIR "MIND directory" ${SRS-MRF_SOURCE_DIR}/External/MIND )
  include_directories( ${DIR_MIND} )
endif()

#Aquirc like stuff
#message( ${MATLAB_ENG_LIBRARY}  ${MATLAB_MX_LIBRARY})
include_directories(${MATLAB_INCLUDE_DIR})

ADD_EXECUTABLE(AquircGlobalNorm2D AquircGlobalNorm2D.cxx )
TARGET_LINK_LIBRARIES(AquircGlobalNorm2D    ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )
ADD_EXECUTABLE(AquircLocalErrors2D AquircLocalError2D.cxx )
TARGET_LINK_LIBRARIES(AquircLocalErrors2D   ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )

ADD_EXECUTABLE(CLERC2D CLERC2D.cxx )
TARGET_LINK_LIBRARIES(CLERC2D Utils      ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )
ADD_EXECUTABLE(CLERC3D CLERC3D.cxx )
TARGET_LINK_LIBRARIES(CLERC3D Utils      ${ITK_LIBRARIES} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY}  )