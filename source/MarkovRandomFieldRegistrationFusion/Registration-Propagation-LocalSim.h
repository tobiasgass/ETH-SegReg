
#pragma once

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
#include "Metrics.h"
#include "itkGaussianImage.h"

using namespace std;

template <class ImageType>
class RegistrationPropagationLocalSim{
public:
    typedef typename ImageType::PixelType PixelType;
    static const unsigned int D=ImageType::ImageDimension;
    typedef typename  ImageType::Pointer ImagePointerType;
    typedef typename  ImageType::IndexType IndexType;
    typedef typename  ImageType::PointType PointType;
    typedef typename  ImageType::OffsetType OffsetType;
    typedef typename  ImageType::SizeType SizeType;
    typedef typename  ImageType::ConstPointer ImageConstPointerType;
    typedef typename  ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename  FloatImageType::Pointer FloatImagePointerType;

    typedef typename  TransfUtils<ImageType>::DisplacementType DisplacementType;
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename  itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef typename  itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
    typedef typename  itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
    typedef typename  itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
    typedef   ImageNeighborhoodIteratorType * ImageNeighborhoodIteratorPointerType;
    typedef typename  ImageNeighborhoodIteratorType::RadiusType RadiusType;

    typedef typename itk::AddImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationAddFilterType;

    enum MetricType {NONE,MAD,NCC,MI,NMI,MSD};
    enum WeightingType {UNIFORM,GLOBAL,LOCAL};
protected:
    double m_sigma;
    RadiusType m_patchRadius;
public:
    int run(int argc, char ** argv){
        feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
        ArgumentParser * as=new ArgumentParser(argc,argv);
        string trueDefListFilename="",deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir=".",outputSuffix="",weightListFilename="";
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
        m_sigma=30;
        //as->parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", true);
        as->parameter ("T", deformationFileList, " list of deformations", true);
        as->parameter ("i", imageFileList, " list of  images", true);
        as->parameter ("true", trueDefListFilename, " list of TRUE deformations", false);
        //as->parameter ("W", weightListFilename,"list of weights for deformations",false);
        as->parameter ("metric", metricName,"metric to be used for global or local weighting, valid: NONE,SAD,MSD,NCC,MI,NMI",false);
        as->parameter ("weighting", weightingName,"internal weighting scheme {uniform,local,global}. non-uniform will only work with metric != NONE",false);
        as->parameter ("s", m_sigma,"sigma for exp(- metric/sigma)",false);
        as->parameter ("radius", radius,"patch radius for local metrics",false);
        as->parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
        as->parameter ("maxHops", maxHops,"maximum number of hops",false);
        as->parameter ("alpha", alpha,"update rate",false);
        as->option ("dontCacheDeformations", dontCacheDeformations,"read deformations only when needed to save memory. higher IO load!");
        //        as->option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
        //as->parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
        as->parameter ("verbose", verbose,"get verbose output",false);
        as->help();
        as->parse();
       

        for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) m_patchRadius[i] = radius;

        mkdir(outputDir.c_str(),0755);
        logSetStage("IO");
        logSetVerbosity(verbose);
        
        MetricType metric;
        if (metricName=="NONE")
            metric=NONE;
        else if (metricName=="MSD")
            metric=MSD;
        else if (metricName=="MAD")
            metric=MAD;
        else if (metricName=="NCC")
            metric=NCC;
        else if (metricName=="NMI")
            metric=NMI;
        else if (metricName=="MI")
            metric=MI;
        else{
            LOG<<"don't understand "<<metricName<<", defaulting to NONE"<<endl;
            metric=NONE;
        }
        WeightingType weighting;
        if (weightingName=="uniform" || metric==NONE){
            weighting=UNIFORM;}
        else if (weightingName=="global")
            weighting=GLOBAL;
        else if (weightingName=="local"){
            weighting=LOCAL;
            if (metric==NMI || metric == MI ){
                LOG<<VAR(metric)<<" incompatibel with local weighing, aborting"<<endl;
                exit(0);
            }
        }
        else{
            LOG<<"don't understand "<<VAR(weightingName)<<", defaulting to uniform."<<endl;
            weighting=UNIFORM;
        }

        if (metric==MAD || metric==MSD){
            if (m_sigma ==0.0){
                weighting=UNIFORM;
                metric=NONE;
            }
        }

        map<string,ImagePointerType> inputImages;
        typedef typename map<string, ImagePointerType>::iterator ImageListIteratorType;
        LOG<<"Reading input images."<<endl;
        std::vector<string> imageIDs;
     
        inputImages = ImageUtils<ImageType>::readImageList( imageFileList , imageIDs);
        int nImages = inputImages.size();
        
        LOGV(2)<<VAR(metric)<<" "<<VAR(weighting)<<endl;
        LOGV(2)<<VAR(m_sigma)<<" "<<VAR(lateFusion)<<" "<<VAR(m_patchRadius)<<endl;

        if (dontCacheDeformations){
            LOG<<"Reading deformation file names."<<endl;
        }else{
            LOG<<"CACHING all deformations!"<<endl;
        }
        map< string, map <string, DeformationFieldPointerType> > deformationCache, trueDeformations;
        map< string, map <string, string> > deformationFilenames;
        map<string, map<string, float> > globalWeights;
        {
            ifstream ifs(deformationFileList.c_str());
            while (!ifs.eof()){
                string intermediateID,targetID,defFileName;
                ifs >> intermediateID;
                if (intermediateID!=""){
                    ifs >> targetID;
                    ifs >> defFileName;
                    if (inputImages.find(intermediateID)==inputImages.end() || inputImages.find(targetID)==inputImages.end() ){
                        LOG<<intermediateID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                        //exit(0);
                    }else{
                        if (!dontCacheDeformations){
                            LOGV(3)<<"Reading deformation "<<defFileName<<" for deforming "<<intermediateID<<" to "<<targetID<<endl;
                            deformationCache[intermediateID][targetID]=ImageUtils<DeformationFieldType>::readImage(defFileName);
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
                    if (inputImages.find(intermediateID)==inputImages.end() || inputImages.find(targetID)==inputImages.end() ){
                        LOG<<intermediateID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                        //exit(0);
                    }else{
                        if (!dontCacheDeformations){
                            LOGV(3)<<"Reading TRUE deformation "<<defFileName<<" for deforming "<<intermediateID<<" to "<<targetID<<endl;
                            trueDeformations[intermediateID][targetID]=ImageUtils<DeformationFieldType>::readImage(defFileName);
                          
                        }else{
                            LOG<<"error, not caching true defs not implemented"<<endl;
                            exit(0);
                        }
                    }
                }
            }
            
        }
        if (weightListFilename!=""){
            ifstream ifs(weightListFilename.c_str());
            while (!ifs.eof()){
                string intermediateID,targetID;
                ifs >> intermediateID;
                ifs >> targetID;
                if (inputImages.find(intermediateID)==inputImages.end() || inputImages.find(targetID)==inputImages.end() ){
                    LOG << intermediateID<<" or "<<targetID<<" not in image database while reading weights, skipping"<<endl;
                }else{
                    ifs >> globalWeights[intermediateID][targetID];
                }
            }
        }

        double error;
        double inconsistency;
        error=TransfUtils<ImageType>::computeError(&deformationCache,&trueDeformations,&imageIDs);
        //inconsistency = TransfUtils<ImageType>::computeInconsistency(&deformationCache,&imageIDs, &trueDeformations);
        int iter = 0;
        LOG<<VAR(iter)<<" "<<VAR(error)<<" "<<VAR(inconsistency)<<endl;
        
        
        logSetStage("Zero Hop");
        LOG<<"Computing"<<std::endl;
        for (iter=1;iter<maxHops+1;++iter){
            map< string, map <string, DeformationFieldPointerType> > TMPdeformationCache;
            int circles=0;
            double globalResidual=0.0;
            double trueResidual=0.0;

            for (ImageListIteratorType sourceImageIterator=inputImages.begin();sourceImageIterator!=inputImages.end();++sourceImageIterator){           
                //iterate over sources
                string sourceID= sourceImageIterator->first;
                for (ImageListIteratorType targetImageIterator=inputImages.begin();targetImageIterator!=inputImages.end();++targetImageIterator){                //iterate over targets
                    string targetID= targetImageIterator->first;
                    if (targetID !=sourceID){
                        GaussianEstimatorVectorImage<ImageType> estimator;
                        DeformationFieldPointerType deformationSourceTarget;
                        deformationSourceTarget = deformationCache[sourceID][targetID];
                        
                        if (radius>0){
                            ImagePointerType warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImageIterator->second,deformationSourceTarget);
                            FloatImagePointerType metric=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                            FilterUtils<FloatImageType>::lowerThresholding(metric,0.0001);
                            estimator.addImage(deformationSourceTarget,metric);
                        }else{
                            estimator.addImage(deformationSourceTarget);
                        }
                        for (ImageListIteratorType intermediateImageIterator=inputImages.begin();intermediateImageIterator!=inputImages.end();++intermediateImageIterator){                //iterate over intermediates
                            string intermediateID= intermediateImageIterator->first;
                            if (targetID != intermediateID && sourceID!=intermediateID){
                                //get all deformations for full circle
                                DeformationFieldPointerType deformationSourceIntermed;
                                deformationSourceIntermed = deformationCache[sourceID][intermediateID];
                                DeformationFieldPointerType deformationIntermedTarget;
                                deformationIntermedTarget = deformationCache[intermediateID][targetID];
                                LOGV(3)<<"Adding "<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(intermediateID)<<endl;
                                DeformationFieldPointerType indirectDef = TransfUtils<ImageType>::composeDeformations(deformationIntermedTarget,deformationSourceIntermed);
                                if (radius>0){
                                    ImagePointerType warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImageIterator->second,indirectDef);
                                    FloatImagePointerType metric=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                    FilterUtils<FloatImageType>::lowerThresholding(metric,0.0001);
                                    estimator.addImage(indirectDef,metric);
                                }else{
                                    estimator.addImage(indirectDef);
                                }


                            }//if
                        }//intermediate image
                        
                        estimator.finalize();
                        DeformationFieldPointerType result=estimator.getMean();
                        if (outputDir!=""){
                            ostringstream oss;
                            oss<<outputDir<<"/avgDeformation-"<<sourceID<<"-TO-"<<targetID<<".mha";
                            ImageUtils<DeformationFieldType>::writeImage(oss.str(),result);
                        }
                            
                        TMPdeformationCache[sourceID][targetID]=estimator.getMean();
                        //create mask of valid deformation region

                        
                    }//if

                   
                }//target images
              
            }//source images
            deformationCache=TMPdeformationCache;
            error=TransfUtils<ImageType>::computeError(&deformationCache,&trueDeformations,&imageIDs);
            //inconsistency = TransfUtils<ImageType>::computeInconsistency(&deformationCache,&imageIDs,&trueDeformations);
            LOG<<VAR(iter)<<" "<<VAR(error)<<" "<<VAR(inconsistency)<<endl;

        }//hops
        LOG<<"done"<<endl;
        
        LOG<<"done"<<endl;
        LOG<<"Storing output. and checking convergence"<<endl;
        for (ImageListIteratorType targetImageIterator=inputImages.begin();targetImageIterator!=inputImages.end();++targetImageIterator){
            string id= targetImageIterator->first;
        }
        // return 1;
    }//run
protected:
    map<string,ImagePointerType> * readImageList(string filename){
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
  

   
};//class
