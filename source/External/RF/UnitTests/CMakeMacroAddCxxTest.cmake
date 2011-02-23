# This macro simplifies the inclusion of tests written using the CxxTest testing framework.
# Python or Perl is required to generate the test source files on the developer machine.
# However, since the generated source files will be placed on the source directory where the macro
# is called, there is no need for the end user to have Python in order to build and run the tests.
# Basic structure for this macro is taken from http://www.cmake.org/Wiki/CMakeMacroAddCxxTest
#
# Copyright (c) 2007 ICG. All rights reserved.
# Institute for Computer Graphics and Vision
# Graz, University of Technology / Austria
# Author: Markus Storer (storer@icg.tugraz.at)

INCLUDE_DIRECTORIES( extern/CxxTest )

# Make sure testing is enabled (enable testing for current directory and below, so this command
# should also be executed in the Top CMakeLists file)
ENABLE_TESTING()

# Use Python interpreter
FIND_PACKAGE( PythonInterp )
# Use Perl interpreter
FIND_PACKAGE( Perl )

MACRO( ADD_CXXTEST NAME )

	IF(PERL_FOUND)
		SET( SCRIPT_EXECUTABLE ${PERL_EXECUTABLE} )
		#Path to the cxxtestgen.pl script
		SET(CXXTESTGEN extern/CxxTest/cxxtestgen.pl)
	ENDIF(PERL_FOUND)
	
	IF(PYTHONINTERP_FOUND)
		SET( SCRIPT_EXECUTABLE ${PYTHON_EXECUTABLE} )
		#Path to the cxxtestgen.py script
		SET(CXXTESTGEN extern/CxxTest/cxxtestgen.py)
	ENDIF(PYTHONINTERP_FOUND)
	
	IF( PERL_FOUND OR PYTHONINTERP_FOUND )
		ADD_CUSTOM_COMMAND(
	      OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.cpp
	      COMMAND
	        ${SCRIPT_EXECUTABLE} ${CXXTESTGEN}
	        "--runner=ParenPrinter"	 #  If you use CMake and Visual Studio, then the -- options (e.g. --runner=CLASS) don't work in Visual Studio; so you must take the - option (e.g. -runner CLASS) or you write "--runner=CLASS"
	        -o ${NAME}.cpp ${ARGN}
	      DEPENDS ${ARGN}
	      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	    )
		
		ADD_EXECUTABLE(${NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.cpp ${ARGN})
		ADD_TEST(${NAME} ${EXECUTABLE_OUTPUT_PATH}/${NAME}.exe ${CMAKE_CURRENT_SOURCE_DIR})
		
	ELSE( PERL_FOUND OR PYTHONINTERP_FOUND )
		MESSAGE("Error: CMake: ADD_CXXTEST: There must be Python or Perl installed to run CxxTest")
	ENDIF( PERL_FOUND OR PYTHONINTERP_FOUND )
 
ENDMACRO(ADD_CXXTEST)
