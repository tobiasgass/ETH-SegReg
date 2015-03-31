# #find_package(ITK REQUIRED)
# include(${ITK_USE_FILE})
# 
ADD_LIBRARY(Utils
   Log.h
   Log.cpp
   ArgumentParser.h
   ArgumentParser.cpp
 )
TARGET_LINK_LIBRARIES(Utils ${Boost_LIBRARIES}  )
 
