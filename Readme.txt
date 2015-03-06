This package contains the source code which was developed during the PHD of Tobias Gass at the Computer Vision Lab at ETH Zurich, Switzerland, from 2010-2015.

Contained are four sub-projects that cover the work related to individual publications and an additional project containing general tools. If you use a part of the code for research purposes, please make sure to cite the relevant paper:

1) SimultaneousRegistrationSegmentation (SRS): Tobias Gass,  Gabor Székely,  Orcun Goksel,  "Simultaneous Segmentation and Multiresolution Nonrigid Atlas Registration",  IEEE Transactions on Image Processing, Vol. 23,  No. 7,  pp. 2931 - 2943, 2014 
   					       Tobias Gass,  Gabor Székely,  Orcun Goksel,  "Auxiliary Anatomical Labels for Joint Segmentation and Atlas Registration",  Proc. SPIE Medical Imaging, February 2014 
2) MarkovRandomFieldRegistrationFusion (MRegFuse) :  Tobias Gass,  Gabor Székely,  Orcun Goksel,  "Registration Fusion using Markov Random Fields",  Workshop on Biomedical Image Registration (WBIR), July 2014 
3) SemiSupervisedSegmentationPropagation (SSSP) :  Tobias Gass,  Gabor Székely,  Orcun Goksel,  "Semi-supervised Segmentation Using Multiple Segmentation Hypotheses from a Single Atlas",  MICCAI Workshop on Medical Computer Vision, October 2012 
4) ConsistencyBasedRegistrationRectification (CBBR): Tobias Gass,  Gabor Székely,  Orcun Goksel,  "Consisteny-Based Registration Rectification", Journal of Medical Imaging, 2015, to appear
   					     	     Tobias Gass,  Gabor Székely,  Orcun Goksel,  "Detection and Correction of Inconsistency-based Errors in Non-Rigid Registration",  Proc. SPIE Medical Imaging, February 2014 
   					     	     Tobias Gass,  Gabor Székely,  Orcun Goksel,  "Consistent Dense Correspondences from Pairwise Registrations",  Proc. Symposium on Statistical Shape Models & Applications, June 2014 
						     

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
INSTALLATION INSTRUCTIONS:

The build uses cmake. It is advisable to create a build directory and run CMake from there. 

mkdir build
cd build
ccmake ../source

In the GUI of ccmake you can then chose which sub-projects to build. Dependencies vary for the sub-projects:

--------------------
GENERAL DEPENDENCIES

-CMAKE 2.8
-ITK is required, versions tested and working are 4.5-4.8. Turn USE_REVIEW on.
-BOOST is required, 1.49 tested and working. This is mainly used for command line parsing and some formatting, and could probably be removed if need be.

--------------------
SRS DEPENDENCIES:
SRS allows for a choice of discrete optimizers. Interfaces for TRW-S, GCO and OpenGM are provided, but those need to be downloaded and installed by the user under their respective license.

OpenGM can be obtained here: http://hci.iwr.uni-heidelberg.de/opengm2/
-current wrapper utilizes BOOST, so make sure to install opengm with their boost feature enabled
Note that openGM comes with a wide variety of discrete optimizers and wrappers, but may be less efficient than using the direct wrappers provided by SRS directly (TRW-S, GCO)

TRW-S can be downloaded here: http://research.microsoft.com/en-us/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/
A patch can(will: todo) be applied automatically during the build process. This will disable all energyTypes except typeGeneral from the downloaded library because the extended functionality was not implemented for those.

GCO can be downloaded here: http://vision.csd.uwo.ca/code/
The same directory can also be used as root for the binary graph-cut (GC) optimizer.

---------------------
MRegFuse DEPENDENCIES

-Requires TRW-S, will work with the same patched version as SRS
-There is some code which requires STATISMO, but that will be made optional in the final build

---------------------
SSSP DEPENDENCIES

Nothing special, just ITK should work fine.

--------------------
CBRR DEPENDENCIES

CBRR requires a working copy of Matlab. Tested with 2013 and 2014 versions. It also requires minFunc, which can be downloaded from here:

http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html (2012)

Make sure to add the path to your MATLABPATH, as well as the directory 

source/ConsistencyBasedRegistrationRectification/Matlab/

which contains some convenience wrappers for minFunc. There is also experimental support for l1 minimization using matlab, which additionally requires 
http://www.cs.ubc.ca/~schmidtm/Software/L1General.html
