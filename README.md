Overview
-------

This package contains the source code which was developed during the PHD of Tobias Gass at the Computer Vision Lab at ETH Zurich, Switzerland, from 2010-2015.

Contained are four sub-projects that cover the work related to individual publications and an additional project containing general tools. If you use a part of the code for research purposes, please make sure to cite the relevant paper:

1. SimultaneousRegistrationSegmentation (SRS):  
  * *Tobias Gass,  Gabor Székely,  Orcun Goksel,  *"Simultaneous Segmentation and Multiresolution Nonrigid Atlas Registration"*,  IEEE Transactions on Image Processing, Vol. 23,  No. 7,  pp. 2931 - 2943, 2014   
  * Tobias Gass,  Gabor Székely,  Orcun Goksel,  *"Auxiliary Anatomical Labels for Joint Segmentation and Atlas Registration"*,  Proc. SPIE Medical Imaging, February 2014   

2. MarkovRandomFieldRegistrationFusion (MRegFuse):    
  * Tobias Gass,  Gabor Székely,  Orcun Goksel,  *"Registration Fusion using Markov Random Fields"*,  Workshop on Biomedical Image Registration (WBIR), July 2014 

3. SemiSupervisedSegmentationPropagation (SSSP):    
  * Tobias Gass,  Gabor Székely,  Orcun Goksel,  *"Semi-supervised Segmentation Using Multiple Segmentation Hypotheses from a Single Atlas"*,  MICCAI Workshop on Medical Computer Vision, October 2012 

4. ConsistencyBasedRegistrationRectification (CBBR):  
  * Tobias Gass,  Gabor Székely,  Orcun Goksel,  *"Consisteny-Based Registration Rectification"*, Journal of Medical Imaging, 2015, to appear  
  * Tobias Gass,  Gabor Székely,  Orcun Goksel,  *"Detection and Correction of Inconsistency-based Errors in Non-Rigid Registration"*,  Proc. SPIE Medical Imaging, February 2014   
  * Tobias Gass,  Gabor Székely,  Orcun Goksel,  *"Consistent Dense Correspondences from Pairwise Registrations"*,  Proc. Symposium on Statistical Shape Models & Applications, June 2014  
						     


INSTALLATION INSTRUCTIONS
-------------------------

The build uses cmake. It is advisable to create a build directory and run CMake from there. 

    > mkdir build
    > cd build
    > ccmake ../
    > make
    > make doc #optional

In the GUI of ccmake you can then chose which sub-projects to build, eg BUILD_SRS = ON. 

Dependencies of the framework vary with enabling/disabling individual
subprojects. For instance, in order to build SRS, only the
dependencies for SRS are necessary. Some general dependencies are shared for all sub-projects and are listed first:


General Dependencies
--------------------

Build was tested on Debian wheezy and OSX 10.10. In OSX, the
RandomForest library will not build easily since OpenMP is not
included in clang, but this is an optional feature.

* CMAKE 2.8
* ITK is required, versions tested and working are 4.5-4.8. Turn
  USE_REVIEW and ITKv3Compatibility on.
* BOOST is required, 1.49 tested and working. This is mainly used for command line parsing and some formatting, and could probably be removed if need be.



SRS Dependencies
----------------

SRS allows for a choice of discrete optimizers. Interfaces for TRW-S,
GCO and OpenGM are provided, but those need to be downloaded and
installed by the user under their respective license. Note that the
optimizers are optional and can be enabled/disabled during cmake
configuration. The respective dependencies and checks will be automatically added to the build.

* OpenGM can be obtained here: http://hci.iwr.uni-heidelberg.de/opengm2/
current wrapper utilizes BOOST, so make sure to install opengm with their boost feature enabled
Note that openGM comes with a wide variety of discrete optimizers and wrappers, but may be less efficient than using the direct wrappers provided by SRS directly (TRW-S, GCO)

* TRW-S can be downloaded here: http://research.microsoft.com/en-us/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/
A patch will be applied automatically during the build process. This will disable all energyTypes except typeGeneral from the downloaded library because the extended functionality was not implemented for those.

* GCO can be downloaded here: http://vision.csd.uwo.ca/code/
The same directory can also be used as root for the binary graph-cut
(GC) optimizer.

A choice of classifiers is available which can be optionally
enabled/disabled. C-UGMIX is a Gaussian mixture model estimator, and RF a
random forest library. Both are included under their own licenses in
this package. Note that the RF library depends on both Boost and OpenMP.


MRegFuse Dependencies
---------------------

* Requires TRW-S, will work with the same patched version as SRS
* There is some code which requires STATISMO, but that will be made optional in the final build


SSSP Dependencies
-----------------

* Nothing special, just ITK should work fine.  
* Some experimental features/methods (not published) need or can optionally utilize graph cut optimization. This will require GCO as discussed under the SRS dependencies. If not present, this features/methods will be disabled.


CBRR Dependencies
-----------------

CBRR requires a working copy of Matlab. Tested with 2013 and 2014 versions. It also requires minFunc, which can be downloaded from here:

* http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html (2012)

Make sure to add the path to your MATLABPATH, as well as the directory 

     > source/ConsistencyBasedRegistrationRectification/Matlab/

which contains some convenience wrappers for minFunc. There is also experimental support for l1 minimization using matlab, which additionally requires 
http://www.cs.ubc.ca/~schmidtm/Software/L1General.html


License
-------

The main part of this project is licensed under the terms of the BSD
(below). Some included parts of external projects are subject to their
own license, which is included either within the source code itself or
supplied in a separate License file.


Copyright (c) 2010-2015, Tobias Gass (tobiasgass@gmail.com) unless
explicitly stated otherwise
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies, 
either expressed or implied, of the FreeBSD Project.

