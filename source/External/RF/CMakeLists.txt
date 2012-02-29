
# BOOST
FIND_PACKAGE(Boost 1.34 COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(
  ${Boost_INCLUDE_DIR}
)


INCLUDE_DIRECTORIES(
    /usr/include/libxml2
 )

LINK_DIRECTORIES(
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

ADD_LIBRARY(RF
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
TARGET_LINK_LIBRARIES (RF)


ADD_EXECUTABLE(RandomForest
    RandomForest.cpp
    ${ICG_RF_SRC}
)

IF(WIN32)
#TARGET_LINK_LIBRARIES(RandomForest
#  libconfig++ iconv libxml2
#)
TARGET_LINK_LIBRARIES(RandomForestLib
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

