# #find_package(ITK REQUIRED)
# include(${ITK_USE_FILE})
FIND_PACKAGE( Boost 1.40 COMPONENTS system filesystem program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
# 
ADD_LIBRARY(Utils
   Log.h
   Log.cpp
   ArgumentParser.h
   ArgumentParser.cpp
 )
TARGET_LINK_LIBRARIES(Utils ${Boost_LIBRARIES}  )
 
