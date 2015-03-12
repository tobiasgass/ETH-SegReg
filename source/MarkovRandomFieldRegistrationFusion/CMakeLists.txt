PROJECT(MRegFuse)
INCLUDE_REGULAR_EXPRESSION("^.*$")

option( USE_STATISMO "Use STATISMO" OFF )

include_directories( ${DIR_TRWS} )


ADD_EXECUTABLE(RegProp2D RegistrationPropagation-2D.cxx )
TARGET_LINK_LIBRARIES(RegProp2D   Utils   TRWS_LIBRARIES  ${ITK_LIBRARIES} )
ADD_EXECUTABLE(RegProp3D RegistrationPropagation-3D.cxx )
TARGET_LINK_LIBRARIES(RegProp3D   Utils    TRWS_LIBRARIES   ${ITK_LIBRARIES}   )

ADD_EXECUTABLE(RegFuse2D RegistrationFusion2D.cxx )
TARGET_LINK_LIBRARIES(RegFuse2D   Utils    TRWS_LIBRARIES   ${ITK_LIBRARIES} )
ADD_EXECUTABLE(RegFuse3D RegistrationFusion3D.cxx )
TARGET_LINK_LIBRARIES(RegFuse3D   Utils     TRWS_LIBRARIES  ${ITK_LIBRARIES}  )



if (${USE_STATISMO} MATCHES "ON")
  find_package(STATISMO REQUIRED)
  add_definitions(-DWITH_STATISMO)
  #reg-propagation
  INCLUDE_DIRECTORIES(
    ${STATISMO_INCLUDE_DIR} ${STATISMO_INCLUDE_DIR}/statismo_ITK  ${STATISMO_INCLUDE_DIR}/3rdParty  ${STATISMO_INCLUDE_DIR}  
    ${STATISMO_INCLUDE_DIR}/Representers/ITK
    )
endif()

if (0)
#Legacy and removed stuff
ADD_EXECUTABLE(RandomFusionReg2D RandomFusionReg2D.cxx )
TARGET_LINK_LIBRARIES(RandomFusionReg2D   Utils      ${ITK_LIBRARIES}  )
endif()