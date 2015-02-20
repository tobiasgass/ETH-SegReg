
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
#include "SolveAquircLocalDeformationAndError.h"
#include "SolveAquircLocalDeformationAndErrorIndependentDimensions.h"

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
typedef   ImageUtils<ImageType>::FloatImageType FloatImageType;
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



map<string,ImagePointerType> * readImageList(string filename,std::vector<string> & imageIDs){
    map<string,ImagePointerType> * result=new  map<string,ImagePointerType>;
    ifstream ifs(filename.c_str());
    if (!ifs){
        LOG<<"could not read "<<filename<<endl;
        exit(0);
    }
    while( ! ifs.eof() ) 
        {
            string imageID;
            ifs >> imageID;                
            if (imageID!=""){
                imageIDs.push_back(imageID);
                ImagePointerType img;
                string imageFileName ;
                ifs >> imageFileName;
                LOGV(3)<<"Reading image "<<imageFileName<< " with ID "<<imageID<<endl;

                img=ImageUtils<ImageType>::readImage(imageFileName);
                if (result->find(imageID)==result->end())
                    (*result)[imageID]=img;
                else{
                    LOG<<"duplicate image ID "<<imageID<<", aborting"<<endl;
                    exit(0);
                }
            }
        }
    return result;
}        
  

double computeError( map< string, map <string, DeformationFieldPointerType> >  & defs,  map< string, map <string, DeformationFieldPointerType> > & trueDefs){
    
    return 0.0;

}
  

int main(int argc, char ** argv){

  


    double m_sigma;
    RadiusType m_patchRadius;
    feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir="",outputSuffix="",weightListFilename="",trueDefListFilename="",ROIFilename="";
    int verbose=0;
    double pWeight=1.0;
    int radius=3;
    int maxHops=1;
    bool uniformUpdate=true;
    string metricName="NCC";
    string weightingName="uniform";
    bool lateFusion=false;
    bool dontCacheDeformations=false;
    bool graphCut=false;
    double smoothness=1.0;
    double lambda=0.0;
    double resamplingFactor=1.0;
    m_sigma=10;
    string solverName="localnorm";
    double wwd=1.0,wwt=1.0,wws=1.0,wwcirc=1.0,wwdelta=1.0,wwsum=100,wsdelta=0.0,m_exponent=1.0,wwInconsistencyError=1.0;
    bool linear=false;
    double shearing = 1.0;
    double m_sigmaD = 0.0;
    double circWeightScaling = 1.0;
    double scalingFactorForConsistentSegmentation = 1.0;
    as->parameter ("T", deformationFileList, " list of deformations", true);
    as->parameter ("true", trueDefListFilename, " list of TRUE deformations", false);
    as->parameter ("ROI", ROIFilename, "file containing a ROI on which to perform erstimation", false);

    as->parameter ("i", imageFileList, " list of  images", true);
    as->parameter ("solver", solverName,"solver used {globalnorm,localnorm,localerror,localcomposederror,localdeformationanderror}",false);
    as->parameter ("s", m_sigma," kernel width for lncc",false);
    as->parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
    as->parameter ("maxHops", maxHops,"maximum number of hops",false);

    as->parameter ("wwd", wwd,"weight for def1 in circle",false);
    as->parameter ("wwt", wwt,"weight for def1 in circle",false);
    as->parameter ("wws", wws,"weight for def1 in circle",false);
    as->parameter ("wsdelta", wsdelta,"weight for def1 in circle",false);
    as->parameter ("wwdelta", wwdelta,"weight for def1 in circle",false);
    as->parameter ("wwcirc", wwcirc,"weight for def1 in circle",false);
    as->parameter ("wwsum", wwsum,"weight for def1 in circle",false);
    as->parameter ("wwincerr",wwInconsistencyError ,"weight for def1 in circle",false);

    as->parameter ("exp",m_exponent ,"exponent for local similarity weights",false);
    as->parameter ("shearing",shearing ,"reduction coefficient for shearing potentials in spatial smoothing",false);
    as->parameter ("sigmaD", m_sigmaD,"scaling for residual distance based circle weight ",false);
    as->parameter ("circScale", circWeightScaling,"scaling of circ weight per iteration ",false);
    as->option ("linear", linear," use linear interpolation (instead of NN) when building equations for circles.");
    as->parameter ("segmentationConsistencyScaling",scalingFactorForConsistentSegmentation,"factor for increasing the weight on consistency for segmentated pixels",false);
    as->parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", false);

    //        as->option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
    //as->parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
    as->parameter ("resamplingFactor", resamplingFactor,"lower resolution by a factor",false);
    as->parameter ("verbose", verbose,"get verbose output",false);
    as->parse();
    
       
    //late fusion is only well defined for maximal 1 hop.
    //it requires to explicitly compute all n!/(n-nHops) deformation paths to each image and is therefore infeasible for nHops>1
    //also strange to implement
    if (lateFusion)
        maxHops==min(maxHops,1);

    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) m_patchRadius[i] = radius;

 
    logSetStage("IO");
    logSetVerbosity(verbose);
        
    MetricType metric;
  
    SolverType solverType=LOCALDEFORMATIONANDERROR;
    if (solverName=="localdeformationanderror"){
        solverType=LOCALDEFORMATIONANDERROR;
    } 

   
    map<string,ImagePointerType> *inputImages;
    typedef  map<string, ImagePointerType>::iterator ImageListIteratorType;
    std::vector<string> imageIDs;
    LOG<<"Reading input images."<<endl;
    inputImages = readImageList( imageFileList, imageIDs );
    int nImages = inputImages->size();
        
   

    if (dontCacheDeformations){
        LOG<<"Reading deformation file names."<<endl;
    }else{
        LOG<<"CACHING all deformations!"<<endl;
    }
    map< string, map <string, DeformationFieldPointerType> > deformationCache, errorAccumulators, trueDeformations,downSampledDeformationCache;
    map< string, map <string, string> > deformationFilenames;
    map<string, map<string, float> > globalWeights;
    DisplacementType zeroDisp;
    zeroDisp.Fill(0.0);

    ImagePointerType origReference=ImageUtils<ImageType>::duplicate( (*inputImages)[imageIDs[0]]);
    ImagePointerType ROI;
    if (ROIFilename!="") {
        ROI=ImageUtils<ImageType>::readImage(ROIFilename);
    }else{
        ROI=origReference;
    }
    if (resamplingFactor>1.0)
        ROI=FilterUtils<ImageType>::LinearResample(ROI,1.0/resamplingFactor,false);
    if (false){
        for (int t=0;t<imageIDs.size();++t){
            string targetID=imageIDs[t];
            //(*inputImages)[targetID]=FilterUtils<ImageType>::LinearResample((*inputImages)[targetID],ROI );
            (*inputImages)[targetID]=FilterUtils<ImageType>::LinearResample((*inputImages)[targetID],ROI,true );
        }
    }
    
    
    LOG<<"WARNING ! ! EVERY IMAGE IS RESAMPLED TO FIRST ONE or ROI!11"<<endl;
    {
        ifstream ifs(deformationFileList.c_str());
        while (!ifs.eof()){
            string sourceID,targetID,defFileName;
            ifs >> sourceID;
            if (sourceID!=""){
                ifs >> targetID;
                ifs >> defFileName;
                if (inputImages->find(sourceID)==inputImages->end() || inputImages->find(targetID)==inputImages->end() ){
                    LOGV(3)<<sourceID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                    //exit(0);
                }else{
                    if (!dontCacheDeformations){
                        LOGV(3)<<"Reading deformation "<<defFileName<<" for deforming "<<sourceID<<" to "<<targetID<<endl;
                        deformationCache[sourceID][targetID]=ImageUtils<DeformationFieldType>::readImage(defFileName);
                        //deformationCache[sourceID][targetID]=TransfUtils<ImageType>::gaussian(deformationCache[sourceID][targetID],resamplingFactor);
                        downSampledDeformationCache[sourceID][targetID]=TransfUtils<ImageType>::linearInterpolateDeformationField( deformationCache[sourceID][targetID], (ConstImagePointerType)ROI);

                        if (false){
                            ImagePointerType deformedSource = TransfUtils<ImageType>::warpImage( (*inputImages)[sourceID] , downSampledDeformationCache[sourceID][targetID] );

                            FloatImagePointerType lncc = Metrics<ImageType,FloatImageType>::LNCC(deformedSource, (*inputImages)[targetID], m_sigma, m_exponent);
                            FloatImagePointerType lssd = Metrics<ImageType,FloatImageType>::LSSDNorm(deformedSource, (*inputImages)[targetID], m_sigma, m_exponent);

                            ostringstream o1,o2;
                            o1<<"lncc-"<<sourceID<<"-"<<targetID<<".nii";
                            o2<<"lssd-"<<sourceID<<"-"<<targetID<<".nii";
                            
                            LOGI(6,ImageUtils<ImageType>::writeImage(o1.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255))));
                            LOGI(6,ImageUtils<ImageType>::writeImage(o2.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lssd,255))));
                            

                        }
                        if (outputDir !=""){
                            ostringstream trueDef2;
                            trueDef2<<outputDir<<"/downSampledDeformation-FROM-"<<sourceID<<"-TO-"<<targetID<<".mha";
                            LOGI(6,ImageUtils<DeformationFieldType>::writeImage(trueDef2.str().c_str(),downSampledDeformationCache[sourceID][targetID]));
                        }
                        LOGV(6)<<VAR(deformationCache[sourceID][targetID]->GetLargestPossibleRegion())<<endl;

                        globalWeights[sourceID][targetID]=1.0;
                    }else{
                        LOGV(3)<<"Reading filename "<<defFileName<<" for deforming "<<sourceID<<" to "<<targetID<<endl;
                        deformationFilenames[sourceID][targetID]=defFileName;
                        globalWeights[sourceID][targetID]=1.0;
                    }
                }
            }
        }
    }
    if (outputDir!=""){
        mkdir(outputDir.c_str(),0755);
    }
    if (trueDefListFilename!=""){
        double trueErrorNorm=0.0;
        ifstream ifs(trueDefListFilename.c_str());
        int c=0;
        while (!ifs.eof()){
            string intermediateID,targetID,defFileName;
            ifs >> intermediateID;
            if (intermediateID!=""){
                ifs >> targetID;
                ifs >> defFileName;
                if (inputImages->find(intermediateID)==inputImages->end() || inputImages->find(targetID)==inputImages->end() ){
                    LOGV(3)<<intermediateID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                    //exit(0);
                }else{
                    if (!dontCacheDeformations){
                        LOGV(3)<<"Reading TRUE deformation "<<defFileName<<" for deforming "<<intermediateID<<" to "<<targetID<<endl;
                        trueDeformations[intermediateID][targetID]=ImageUtils<DeformationFieldType>::readImage(defFileName);
                        trueDeformations[intermediateID][targetID]=TransfUtils<ImageType>::linearInterpolateDeformationField( trueDeformations[intermediateID][targetID], (ConstImagePointerType)ROI);
                        if (outputDir!=""){
                            DeformationFieldPointerType diff=TransfUtils<ImageType>::subtract(downSampledDeformationCache[intermediateID][targetID],trueDeformations[intermediateID][targetID]);
                            ostringstream trueDefNorm;
                            trueDefNorm<<outputDir<<"/trueLocalDeformationNorm-FROM-"<<intermediateID<<"-TO-"<<targetID<<".png";
                            LOGI(8,ImageUtils<ImageType>::writeImage(trueDefNorm.str().c_str(),FilterUtils<FloatImageType,ImageType>::truncateCast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(TransfUtils<ImageType>::computeLocalDeformationNorm(diff,1.0),50))));
                            ostringstream trueDef;
                            trueDef<<outputDir<<"/trueLocalDeformationERROR-FROM-"<<intermediateID<<"-TO-"<<targetID<<".mha";
                            LOGI(1,ImageUtils<DeformationFieldType>::writeImage(trueDef.str().c_str(),diff));
                          
                        }
                        trueErrorNorm+=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract(downSampledDeformationCache[intermediateID][targetID], trueDeformations[intermediateID][targetID]),1);
                        ++c;
                    }  
                    
                    else{
                        LOG<<"error, not caching true defs not implemented"<<endl;
                        exit(0);
                    }
                }
                
            }
            
        }
        trueErrorNorm=trueErrorNorm/c;
        LOGV(1)<<VAR(trueErrorNorm)<<endl;
    }
    
            
    map<string,ImagePointerType> *atlasSegmentations = NULL;
    if (atlasSegmentationFileList!=""){
        std::vector<string> buff;
        atlasSegmentations=readImageList(atlasSegmentationFileList,buff);
        
    }
   
    
    //AquircLocalDeformationAndErrorSolver<ImageType> * solver;
    AquircLocalDeformationAndErrorSolverIndependentDimensions<ImageType> * solver;

    switch(solverType){
    case LOCALDEFORMATIONANDERROR:
        //solver= new AquircLocalDeformationAndErrorSolver<ImageType>;
        solver= new AquircLocalDeformationAndErrorSolverIndependentDimensions<ImageType>;
        break;
    }
    
    solver->setWeightWd(wwd);
    solver->setWeightWs(wws);
    solver->setWeightWT(wwt);
    solver->setWeightWdelta(wwdelta);
    solver->setWeightWsDelta(wsdelta);
    solver->setWeightWcirc(wwcirc); 
    solver->setWeightSum(wwsum); 
    solver->setWeightInconsistencyError(wwInconsistencyError); 

    solver->setLinearInterpol(linear);
    solver->setSigma(m_sigma);
    solver->setSigmaD(m_sigmaD);
    solver->setLocalWeightExp(m_exponent);
    solver->setShearingReduction(shearing);
    solver->SetVariables(&imageIDs,&deformationCache,&trueDeformations,ROI,inputImages,&downSampledDeformationCache);


    if (atlasSegmentationFileList!=""){
        solver->setSegmentationList(atlasSegmentations);
        solver->setScalingFactorForConsistentSegmentation(scalingFactorForConsistentSegmentation);
    }

    for (int h=0;h<maxHops;++h){
        solver->setWeightWcirc(wwcirc*pow(circWeightScaling,h)); 
        //solver->setWeightWdelta(wwdelta*pow(2,h));
        //solver->setSigma(m_sigma/pow(2,h)); 
        solver->createSystem();
        solver->solve();
        solver->storeResult(outputDir);
        
        
#if 0
        map< string, map <string, DeformationFieldPointerType> > * estimatedDeforms=solver->getEstimatedDeformations();
        solver->computeError(estimatedDeforms);
        deformationCache=*estimatedDeforms;
        if (outputDir!=""){
            
            for (int t=0;t<imageIDs.size();++t){
                string targetID=imageIDs[t];
                for (int s=0;s<imageIDs.size();++s){
                    if (s!=t){
                        string sourceID=imageIDs[s];
                        DeformationFieldPointerType def=  deformationCache[sourceID][targetID];
                        if (resamplingFactor !=1.0){
                            def=TransfUtils<ImageType>::bSplineInterpolateDeformationField(def,origReference);
                        }
                        ostringstream oss;
                        oss<<outputDir<<"/deformation-"<<solverName<<"-from-"<<sourceID<<"-to-"<<targetID<<"-hop"<<h<<".mha";
                        ImageUtils<DeformationFieldType>::writeImage(oss.str().c_str(),def);
                        
                    }
                }
            }

        }
        delete estimatedDeforms;
#endif
    }

    

    
    
    
    return 1;
}//main
