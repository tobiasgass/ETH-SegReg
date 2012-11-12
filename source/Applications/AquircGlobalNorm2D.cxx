
#include <stdio.h>
#include <iostream>

#include "argstream.h"
#include "Log.h"
#include <vector>
#include <map>
#include "itkImageRegionIterator.h"
#include "TransformationUtils.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <sstream>
#include "argstream.h"
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
#include "SolveAquircGlobalDeformationNorm.h"
#include "SolveAquircGlobalDeformationNormCVariables.h"
#include "SolveAquircLocalDeformationNorms.h"
#include "SolveAquircLocalDeformations.h"
#include "SolveAquircLocalComposedDeformations.h"
#include "SolveAquircLocalError.h"

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

typedef  itk::AddImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationAddFilterType;
typedef  DeformationAddFilterType::Pointer DeformationAddFilterPointer;
typedef  itk::SubtractImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationSubtractFilterType;

typedef  itk::FixedPointInverseDeformationFieldImageFilter<DeformationFieldType,DeformationFieldType> InverseDeformationFieldFilterType;
typedef  InverseDeformationFieldFilterType::Pointer InverseDeformationFieldFilterPointerType;
enum MetricType {NONE,MAD,NCC,MI,NMI,MSD};
enum WeightingType {UNIFORM,GLOBAL,LOCAL};
enum SolverType {GLOBALNORM,LOCALNORM,LOCALERROR,LOCALCOMPOSEDERROR,LOCALCOMPOSEDNORM};



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
  

  
  

int main(int argc, char ** argv){

  


    double m_sigma;
    RadiusType m_patchRadius;
    feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    argstream * as=new argstream(argc,argv);
    string deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir=".",outputSuffix="",weightListFilename="",trueDefListFilename="";
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
    double alpha=0.5;
    double resamplingFactor=1.0;
    m_sigma=30;
    string solverName="localnorm";
    //(*as) >> parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", true);
    (*as) >> parameter ("T", deformationFileList, " list of deformations", true);
    (*as) >> parameter ("true", trueDefListFilename, " list of TRUE deformations", false);

    (*as) >> parameter ("i", imageFileList, " list of  images", true);
    (*as) >> parameter ("solver", solverName,"solver used {globalnorm,localnorm,localerror,localcomposederror}",false);
    (*as) >> parameter ("s", m_sigma,"sigma for exp(- metric/sigma)",false);
    //(*as) >> parameter ("radius", radius,"patch radius for local metrics",false);
    (*as) >> parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",true);
    //(*as) >> parameter ("radius", radius,"patch radius for NCC",false);
    (*as) >> parameter ("maxHops", maxHops,"maximum number of hops",false);
    (*as) >> parameter ("alpha", alpha,"update rate",false);
    (*as) >> option ("lateFusion", lateFusion,"fuse segmentations late. maxHops=1");
    (*as) >> option ("dontCacheDeformations", dontCacheDeformations,"read deformations only when needed to save memory. higher IO load!");
    //        (*as) >> option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
    //(*as) >> parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
    (*as) >> parameter ("resamplingFactor", resamplingFactor,"lower resolution by a factor",false);
    (*as) >> parameter ("verbose", verbose,"get verbose output",false);
    (*as) >> help();
    as->defaultErrorHandling();
       

    //late fusion is only well defined for maximal 1 hop.
    //it requires to explicitly compute all n!/(n-nHops) deformation paths to each image and is therefore infeasible for nHops>1
    //also strange to implement
    if (lateFusion)
        maxHops==min(maxHops,1);

    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) m_patchRadius[i] = radius;

    mkdir(outputDir.c_str(),0755);
    logSetStage("IO");
    logSetVerbosity(verbose);
        
    MetricType metric;
  
    SolverType solverType;
    if (solverName=="globalnorm" ){
        solverType=GLOBALNORM;}
    else if (solverName=="localnorm")
        solverType=LOCALNORM;
    else if (solverName=="localerror"){
        solverType=LOCALERROR;
    }  else if (solverName=="localcomposederror"){
        solverType=LOCALCOMPOSEDERROR;
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
    map< string, map <string, DeformationFieldPointerType> > deformationCache, errorAccumulators, trueDeformations;
    map< string, map <string, string> > deformationFilenames;
    map<string, map<string, float> > globalWeights;
    DisplacementType zeroDisp;
    zeroDisp.Fill(0.0);
    {
        ifstream ifs(deformationFileList.c_str());
        while (!ifs.eof()){
            string intermediateID,targetID,defFileName;
            ifs >> intermediateID;
            if (intermediateID!=""){
                ifs >> targetID;
                ifs >> defFileName;
                if (inputImages->find(intermediateID)==inputImages->end() || inputImages->find(targetID)==inputImages->end() ){
                    LOG<<intermediateID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                    //exit(0);
                }else{
                    if (!dontCacheDeformations){
                        LOGV(3)<<"Reading deformation "<<defFileName<<" for deforming "<<intermediateID<<" to "<<targetID<<endl;
                        deformationCache[intermediateID][targetID]=ImageUtils<DeformationFieldType>::readImage(defFileName);
                        errorAccumulators[intermediateID][targetID]=ImageUtils<DeformationFieldType>::createEmpty(deformationCache[intermediateID][targetID]);
                        errorAccumulators[intermediateID][targetID]->FillBuffer(zeroDisp);

                        globalWeights[intermediateID][targetID]=1.0;
                    }else{
                        LOGV(3)<<"Reading filename "<<defFileName<<" for deforming "<<intermediateID<<" to "<<targetID<<endl;
                        deformationFilenames[intermediateID][targetID]=defFileName;
                        globalWeights[intermediateID][targetID]=1.0;
                    }
                }
            }
        }
    }

    if (trueDefListFilename!=""){
        ifstream ifs(trueDefListFilename.c_str());
        while (!ifs.eof()){
            string intermediateID,targetID,defFileName;
            ifs >> intermediateID;
            if (intermediateID!=""){
                ifs >> targetID;
                ifs >> defFileName;
                if (inputImages->find(intermediateID)==inputImages->end() || inputImages->find(targetID)==inputImages->end() ){
                    LOG<<intermediateID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                    //exit(0);
                }else{
                    if (!dontCacheDeformations){
                        LOGV(3)<<"Reading TRUE deformation "<<defFileName<<" for deforming "<<intermediateID<<" to "<<targetID<<endl;
                        trueDeformations[intermediateID][targetID]=ImageUtils<DeformationFieldType>::readImage(defFileName);
                        DeformationFieldPointerType diff=TransfUtils<ImageType>::subtract(deformationCache[intermediateID][targetID],trueDeformations[intermediateID][targetID]);
                        ostringstream trueDefNorm;
                        trueDefNorm<<outputDir<<"/trueLocalDeformationNorm-FROM-"<<intermediateID<<"-TO-"<<targetID<<".png";
                        ImageUtils<ImageType>::writeImage(trueDefNorm.str().c_str(),FilterUtils<FloatImageType,ImageType>::truncateCast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(TransfUtils<ImageType>::computeLocalDeformationNorm(diff,1.0),50)));
                        ostringstream trueDef;
                        trueDef<<outputDir<<"/trueLocalDeformationERROR-FROM-"<<intermediateID<<"-TO-"<<targetID<<".mha";
                        ImageUtils<DeformationFieldType>::writeImage(trueDef.str().c_str(),diff);
                        
                          
                    }else{
                        LOG<<"error, not caching true defs not implemented"<<endl;
                        exit(0);
                    }
                }
               
            }
        }
            
    }

    if (resamplingFactor !=1.0){
        for (int t=0;t<imageIDs.size();++t){
            string targetID=imageIDs[t];
            (*inputImages)[targetID]=FilterUtils<ImageType>::LinearResample( (*inputImages)[targetID],1.0/resamplingFactor);
            for (int s=0;s<imageIDs.size();++s){
                if (s!=t){
                    string sourceID=imageIDs[s];
                    deformationCache[sourceID][targetID]=TransfUtils<ImageType>::linearInterpolateDeformationField( deformationCache[sourceID][targetID],(ConstImagePointerType) (*inputImages)[targetID]);
                    if (trueDefListFilename!=""){
                        trueDeformations[sourceID][targetID]=TransfUtils<ImageType>::linearInterpolateDeformationField( trueDeformations[sourceID][targetID],(ConstImagePointerType) (*inputImages)[targetID]);
                    }
                }
            }
        }
        


    }
     
    AquircGlobalDeformationNormSolverCVariables<ImageType> * solver;
    switch(solverType){
    case GLOBALNORM:
        solver=new AquircGlobalDeformationNormSolverCVariables<ImageType> ;
        break;
    case LOCALNORM:
        solver=new AquircLocalDeformationNormSolver<ImageType>;
        break;

    case LOCALERROR:
        solver= new AquircLocalDeformationSolver<ImageType>;
        break;

    case LOCALCOMPOSEDERROR:
        //solver = new AquircLocalComposedDeformationSolver<ImageType> ;
        solver = new AquircLocalErrorSolver<ImageType> ;
        break;

    }
    solver->SetVariables(&imageIDs,&deformationCache,&trueDeformations);
    solver->createSystem();
    solver->solve();
    solver->storeResult(outputDir);
    //std::vector<double> result=Solver.getResult();
    
    
    // return 1;
}//main
