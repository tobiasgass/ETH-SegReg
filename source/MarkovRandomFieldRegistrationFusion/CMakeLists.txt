PROJECT(ImageRegistration)
INCLUDE_REGULAR_EXPRESSION("^.*$")


#reg-propagation
INCLUDE_DIRECTORIES(
  ../Common
    /home/gasst/work/src/statismoREADONLY/build/include/statismo_ITK
    /home/gasst/work/src/statismoREADONLY/build/include/Representers/ITK/
    ${STATISMO_INCLUDE_DIR} ${STATISMO_INCLUDE_DIR}/statismo_ITK  ${STATISMO_INCLUDE_DIR}/3rdParty  ${STATISMO_INCLUDE_DIR}  
    ${STATISMO_INCLUDE_DIR}/Representers/ITK
 )

ADD_EXECUTABLE(RegProp2D RegistrationPropagation-2D.cxx )
TARGET_LINK_LIBRARIES(RegProp2D   Utils  SRSMRFModels    ${ITK_LIBRARIES} ${STATISMO_LIBRARIES} )
ADD_EXECUTABLE(RegProp3D RegistrationPropagation-3D.cxx )
TARGET_LINK_LIBRARIES(RegProp3D   Utils  SRSMRFModels    ${ITK_LIBRARIES}  ${STATISMO_LIBRARIES} )

ADD_EXECUTABLE(RegFuse2D RegistrationFusion2D.cxx )
TARGET_LINK_LIBRARIES(RegFuse2D   Utils  SRSMRFModels    ${ITK_LIBRARIES} ${STATISMO_LIBRARIES} )
ADD_EXECUTABLE(RegFuse3D RegistrationFusion3D.cxx )
TARGET_LINK_LIBRARIES(RegFuse3D   Utils  SRSMRFModels    ${ITK_LIBRARIES} ${STATISMO_LIBRARIES} )

ADD_EXECUTABLE(RandomFusionReg2D RandomFusionReg2D.cxx )
TARGET_LINK_LIBRARIES(RandomFusionReg2D   Utils  SRSMRFModels    ${ITK_LIBRARIES} ${STATISMO_LIBRARIES} )


