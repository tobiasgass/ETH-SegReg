#include <stdio.h>
#include <iostream>

#include "ArgumentParser.h"
#include "Log.h"
#include <vector>
#include <map>
#include "itkImageRegionIterator.h"
#include "TransformationUtils.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <sstream>
#include "ArgumentParser.h"
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include "itkConstNeighborhoodIterator.h"
#include "itkDisplacementFieldTransform.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include "itkFixedPointInverseDeformationFieldImageFilter.h"

//using namespace std;
typedef unsigned char PixelType;
static const unsigned int D=2 ;
typedef itk::Image<PixelType,D> ImageType;
typedef   ImageType::Pointer ImagePointerType;
typedef   ImageType::IndexType IndexType;
typedef   ImageType::PointType PointType;
typedef   ImageType::OffsetType OffsetType;
typedef   ImageType::SizeType SizeType;
typedef   ImageType::ConstPointer ImageConstPointerType;
typedef   ImageType::ConstPointer ConstImagePointerType;
typedef   ImageUtils<ImageType,double>::FloatImageType FloatImageType;
typedef   FloatImageType::Pointer FloatImagePointerType;


typedef   TransfUtils<ImageType>::DisplacementType DisplacementType;
typedef   TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
typedef   DeformationFieldType::Pointer DeformationFieldPointerType;
typedef   itk::ImageRegionIterator<ImageType> ImageIteratorType;
typedef   itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
typedef   itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
typedef   itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
typedef   ImageNeighborhoodIteratorType * ImageNeighborhoodIteratorPointerType;
typedef   ImageNeighborhoodIteratorType::RadiusType RadiusType;

typedef  itk::ImageRegionIteratorWithIndex<DeformationFieldType> DeformationFieldIterator;
typedef  itk::AddImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationAddFilterType;
typedef  DeformationAddFilterType::Pointer DeformationAddFilterPointer;
typedef  itk::SubtractImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationSubtractFilterType;

typedef  itk::FixedPointInverseDeformationFieldImageFilter<DeformationFieldType,DeformationFieldType> InverseDeformationFieldFilterType;
typedef  InverseDeformationFieldFilterType::Pointer InverseDeformationFieldFilterPointerType;
enum MetricType {NONE,MAD,NCC,MI,NMI,MSD};
enum WeightingType {UNIFORM,GLOBAL,LOCAL};
enum SolverType {LOCALDEFORMATIONANDERROR};

typedef map<string, map< string, string> > FileListCacheType;
typedef  map<string,ImagePointerType> ImageCacheType;
typedef  map<string, ImagePointerType>::iterator ImageListIteratorType;
typedef map< string, map <string, DeformationFieldPointerType> > DeformationCacheType;





int main(int argc, char ** argv){

  


    double m_sigma;
    RadiusType m_patchRadius;
    //feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string maskFileList="",groundTruthSegmentationFileList="",landmarkFileList="",deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir="",outputSuffix="",weightListFilename="",trueDefListFilename="",ROIFilename="";
    int verbose=0;
    int radius=3;
    int maxHops=1;
    string metricName="NCC";
    string weightingName="uniform";
    bool lateFusion=false;
     m_sigma=10;
    string solverName="localnorm";
    double wwd=0.0,wwt=1.0,wws=0.0,wwcirc=1.0,wwdelta=0.0,wwsum=0,wsdelta=0.0,m_exponent=1.0,wwInconsistencyError=0.0,wErrorStatistics=0.0;
    double scalingFactorForConsistentSegmentation = 1.0;
    bool oracle = false;
    string localSimMetric="lncc";
    bool evalLowResolutionDeformationss=false;
    bool roiShift=false;
    bool useConstraints=false;
    double annealing=1.0;
    as->parameter ("T", deformationFileList, " list of deformations", true);
    as->parameter ("true", trueDefListFilename, " list of TRUE deformations", false);
    as->parameter ("ROI", ROIFilename, "file containing a ROI on which to perform erstimation", false);
    as->parameter ("i", imageFileList, " list of  images", true);

    as->parameter ("verbose", verbose,"get verbose output",false);
    as->parse();
    
       
    //late fusion is only well defined for maximal 1 hop.
    //it requires to explicitly compute all n!/(n-nHops) deformation paths to each image and is therefore infeasible for nHops>1
    //also strange to implement
    if (lateFusion)
        maxHops=min(maxHops,1);

    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) m_patchRadius[i] = radius;

 
    logSetStage("IO");
    logSetVerbosity(verbose);

    
    //read images and image IDs
    ImageCacheType inputImages;
    std::vector<string> imageIDs;
    LOG<<"Reading input images."<<endl;
    inputImages = ImageUtils<ImageType>::readImageList( imageFileList, imageIDs );
    int nImages = inputImages.size();

    ImageCacheType * inputMasks=NULL;
    if (maskFileList!=""){
        inputMasks=new ImageCacheType;
        std::vector<string> buff;
        LOG<<"Reading input masks."<<endl;
        (*inputMasks) = ImageUtils<ImageType>::readImageList( maskFileList, buff );
    }
        
 
    //read  target deformations filenames
    DeformationCacheType defCache,trueDefCache;
    ifstream ifs(deformationFileList.c_str());
    while (!ifs.eof()){
        string sourceID,targetID,defFileName;
        ifs >> sourceID;
        if (sourceID!=""){
            ifs >> targetID;
            ifs >> defFileName;
            if (inputImages.find(sourceID)==inputImages.end() || inputImages.find(targetID)==inputImages.end() ){
                LOGV(3)<<sourceID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                //exit(0);
            }else{
                LOGV(3)<<"Reading deformation "<<defFileName<<" for deforming "<<sourceID<<" to "<<targetID<<endl;
                defCache[sourceID][targetID]=ImageUtils<DeformationFieldType>::readImage(defFileName);
            }
        }
    }
    
    //read  true deformation filenames
    if (trueDefListFilename!=""){
        ifstream ifs(trueDefListFilename.c_str());
        while (!ifs.eof()){
            string sourceID,targetID,defFileName;
            ifs >> sourceID;
            if (sourceID!=""){
                ifs >> targetID;
                ifs >> defFileName;
                if (inputImages.find(sourceID)==inputImages.end() || inputImages.find(targetID)==inputImages.end() ){
                    LOGV(3)<<sourceID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                    //exit(0);
                }else{
                    trueDefCache[sourceID][targetID]=ImageUtils<DeformationFieldType>::readImage(defFileName);
                }  
            }
        }
    }

    double ADE=TransfUtils<ImageType>::computeError(&defCache,&trueDefCache,&imageIDs);
    double C=TransfUtils<ImageType>::computeInconsistency(&defCache,&imageIDs,NULL);
    LOG<<VAR(ADE)<<" "<<VAR(C)<<endl;

    return 1;
}//main
