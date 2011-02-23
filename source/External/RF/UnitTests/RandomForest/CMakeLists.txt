# CMakeLists.txt
# Unit Tests RandomForest

if( WIN32 )
    #ADD_DEFINITIONS( -DFWK_WIN32 )
endif( WIN32 )

INCLUDE( .../CMakeMacroAddCxxTest.cmake )

if( WIN32 )
    INCLUDE_DIRECTORIES( ../../
 	)

#   LINK_DIRECTORIES( $ENV{OPENCV_DIR}/lib
#                         ${LaViTo_SOURCE_DIR}/EXTERN/libxml2/lib 
#                         ${LaViTo_SOURCE_DIR}/TOOLBOX/ImageIO/libjpeg )
else( WIN32 )
    INCLUDE_DIRECTORIES( /usr/local/include
			 ../)
endif( WIN32 )

#ADD_DEPENDENCIES( testRandomForest lvtDetector )

# SOURCE DECLARATION
SET(ICG_RF_SRC
  forest.cpp
  forest.h
  tree.cpp
  tree.h
  hyperparameters.h
  data.h
  data.cpp
  node.cpp
  node.h
  sharedptr.h
  utilities.h
  utilities.cpp
)

ADD_CXXTEST( testRandomForest testRandomForest.h $(ICG_RF_SRC) )

TARGET_LINK_LIBRARIES(testRandomForest
  config++
)
