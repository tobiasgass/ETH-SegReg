
# BOOST
FIND_PACKAGE(Boost 1.34 COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(
  ${Boost_INCLUDE_DIR}
)

#set(DIR_XML  "/usr/include/libxml2/")
set(DIR_LIBXML "/usr/include/libxml2/" CACHE  FILEPATH "Parent directory of directory libxml ")
mark_as_advanced(DIR_LIBXML)


IF(NOT IS_DIRECTORY "${DIR_LIBXML}/libxml" )
  message( SEND_ERROR "libxml not found at ${DIR_LIBXML}/libxml, check directory" )
  set(DIR_LIBXML "DIR NOT FOUND"   FILEPATH "Parent directory of directory libxml " FORCE)
  mark_as_advanced(CLEAR DIR_LIBXML)
endif()

INCLUDE_DIRECTORIES(
    "${DIR_LIBXML}/"
 )


# SOURCE DECLARATION
SET(ICG_RF_SRC
  pairnode.cpp
  pairtree.cpp
  pairforest.cpp
  forest.cpp
  tree.cpp
  hyperparameters.h
  data.cpp
  node.cpp
  utilities.cpp
  nodegini.cpp
  nodeinfogain.cpp
  nodehyperplane.cpp
  tree.h
  utilities.h
)

ADD_LIBRARY(RandomForest
  pairnode.cpp
  pairtree.cpp
  pairforest.cpp
  forest.cpp
  tree.cpp
  hyperparameters.h
  data.cpp
  node.cpp
  utilities.cpp
  nodegini.cpp
  nodeinfogain.cpp
  nodehyperplane.cpp
  tree.h
  utilities.h
  randomnaivebayes.cpp
  naivebayes.cpp
  naivebayeshyperplane.cpp
  naivebayesfeature.cpp
)
#TARGET_LINK_LIBRARIES (RandomForest xml2)


# ADD_EXECUTABLE(RandomForest
#     RandomForest.cpp
#     ${ICG_RF_SRC}
# )
# 
IF(WIN32)
#TARGET_LINK_LIBRARIES(RandomForest
#  libconfig++ iconv libxml2
#)
TARGET_LINK_LIBRARIES(RandomForest
  libconfig++ iconv libxml2
)
ELSE(WIN32)
TARGET_LINK_LIBRARIES(RandomForest
   xml2 config++ gomp)
ENDIF(WIN32)

# Add unit tests
IF( RUN_TEST )
    ADD_SUBDIRECTORY( UnitTests )
ENDIF( RUN_TEST )

