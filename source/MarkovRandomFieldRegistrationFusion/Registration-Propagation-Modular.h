
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


using namespace std;

template <class ImageType>
class RegistrationPropagationModular{
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
        //as->parameter ("metric", metricName,"metric to be used for global or local weighting, valid: NONE,SAD,MSD,NCC,MI,NMI",false);
        //as->parameter ("weighting", weightingName,"internal weighting scheme {uniform,local,global}. non-uniform will only work with metric != NONE",false);
        as->parameter ("s", m_sigma,"sigma for exp(- metric/sigma)",false);
        //as->parameter ("radius", radius,"patch radius for local metrics",false);
        as->parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
        //as->parameter ("radius", radius,"patch radius for NCC",false);
        as->parameter ("maxHops", maxHops,"maximum number of hops",false);
        as->parameter ("alpha", alpha,"update rate",false);
        as->option ("lateFusion", lateFusion,"fuse segmentations late. maxHops=1");
        as->option ("dontCacheDeformations", dontCacheDeformations,"read deformations only when needed to save memory. higher IO load!");
        //        as->option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
        //as->parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
        as->parameter ("verbose", verbose,"get verbose output",false);
        as->help();
        as->parse();
       

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

        map<string,ImagePointerType> *inputImages;
        typedef typename map<string, ImagePointerType>::iterator ImageListIteratorType;
        LOG<<"Reading input images."<<endl;
        inputImages = readImageList( imageFileList );
        int nImages = inputImages->size();
        
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
                    if (inputImages->find(intermediateID)==inputImages->end() || inputImages->find(targetID)==inputImages->end() ){
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
                    if (inputImages->find(intermediateID)==inputImages->end() || inputImages->find(targetID)==inputImages->end() ){
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
                if (inputImages->find(intermediateID)==inputImages->end() || inputImages->find(targetID)==inputImages->end() ){
                    LOG << intermediateID<<" or "<<targetID<<" not in image database while reading weights, skipping"<<endl;
                }else{
                    ifs >> globalWeights[intermediateID][targetID];
                }
            }
        }
#define AVERAGE
        //#define LOCALWEIGHTING
        logSetStage("Zero Hop");
        LOG<<"Computing"<<std::endl;
        for (int h=0;h<maxHops;++h){
            map< string, map <string, DeformationFieldPointerType> > TMPdeformationCache;
            int circles=0;
            double globalResidual=0.0;
            double trueResidual=0.0;

            for (ImageListIteratorType sourceImageIterator=inputImages->begin();sourceImageIterator!=inputImages->end();++sourceImageIterator){           
                //iterate over sources
                string sourceID= sourceImageIterator->first;
                for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){                //iterate over targets
                    string targetID= targetImageIterator->first;
                    DeformationFieldPointerType sourceTargetAccumulator;
                    FloatImagePointerType sourceTargetWeightAccumulator;
                    double bestResidual=10000000.0;
                    int count=0;
                    double weight=0.0;
                    double averageResidual=0.0;
                    if (targetID !=sourceID){
                        for (ImageListIteratorType intermediateImageIterator=inputImages->begin();intermediateImageIterator!=inputImages->end();++intermediateImageIterator){                //iterate over intermediates
                            string intermediateID= intermediateImageIterator->first;
                            if (targetID != intermediateID && sourceID!=intermediateID){
                                //get all deformations for full circle
                                DeformationFieldPointerType deformationSourceIntermed;
                                deformationSourceIntermed = deformationCache[sourceID][intermediateID];
                                DeformationFieldPointerType deformationIntermedTarget;
                                deformationIntermedTarget = deformationCache[intermediateID][targetID];
                                DeformationFieldPointerType deformationTargetSource;
                                deformationTargetSource = deformationCache[targetID][sourceID];

                                //create mask of valid deformation region
                                ImagePointerType mask=ImageType::New();
                                mask->SetRegions(deformationTargetSource->GetLargestPossibleRegion());
                                mask->SetOrigin(deformationTargetSource->GetOrigin());
                                mask->SetSpacing(deformationTargetSource->GetSpacing());
                                mask->SetDirection(deformationTargetSource->GetDirection());
                                mask->Allocate();
                                mask->FillBuffer(1);
                                mask=TransfUtils<ImageType>::warpImage(mask,deformationSourceIntermed,true);
                                mask=TransfUtils<ImageType>::warpImage(mask,deformationIntermedTarget,true);
                                mask=TransfUtils<ImageType>::warpImage(mask,deformationTargetSource,true);
                                ostringstream tmpFilename;
                                tmpFilename<<outputDir<<"/mask-from-"<<sourceID<<"-VIA-"<<intermediateID<<"-TO-"<<targetID<<"-hop"<<h<<".png";
                                //ImageUtils<ImageType>::writeImage(tmpFilename.str().c_str(),FilterUtils<ImageType>::normalize(mask));

                                //compute circle
                                DeformationFieldPointerType sourceTarget=TransfUtils<ImageType>::composeDeformations(deformationIntermedTarget,deformationSourceIntermed);
                                DeformationFieldPointerType circle=TransfUtils<ImageType>::composeDeformations(deformationTargetSource,sourceTarget);
#if 0
                                DeformationFieldPointerType intermedSource=TransfUtils<ImageType>::composeDeformations(deformationTargetSource,deformationIntermedTarget);
                                DeformationFieldPointerType circleTEST=TransfUtils<ImageType>::composeDeformations(intermedSource,deformationSourceIntermed);

                                DeformationFieldPointerType delta=TransfUtils<ImageType>::subtract(circle,circleTEST);
                                ostringstream tmpFilename231;
                                tmpFilename231<<outputDir<<"/compOrderDifference-from-"<<sourceID<<"-VIA-"<<intermediateID<<"-TO-"<<targetID<<"-hop"<<h<<".png";
                                ImageUtils<ImageType>::writeImage(tmpFilename231.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(TransfUtils<ImageType>::computeLocalDeformationNormWeights(delta,m_sigma),65535)));
#endif



#if 1
                                ImagePointerType deformedSourceCircle=TransfUtils<ImageType>::warpImage(sourceImageIterator->second,circle);
                                ostringstream tmpFilename2;
                                tmpFilename2<<outputDir<<"/deformed-from-"<<sourceID<<"-VIA-"<<intermediateID<<"-TO-"<<targetID<<"-hop"<<h<<".png";
                                ImageUtils<ImageType>::writeImage(tmpFilename2.str().c_str(),deformedSourceCircle);

                                
                                FloatImagePointerType localDeformationNormWeights=TransfUtils<ImageType>::computeLocalDeformationNormWeights(circle,m_sigma);
                                FloatImagePointerType localDeformationNormWeightsSourceTarget=TransfUtils<FloatImageType>::warpImage(localDeformationNormWeights, sourceTarget);
                                
                                ostringstream tmpSegmentationFilename;
                                //tmpSegmentationFilename<<outputDir<<"/error-from-"<<sourceID<<"-VIA-"<<intermediateID<<"-TO-"<<targetID<<"-hop"<<h<<".png";
                                //ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(localDeformationNormWeights,65535)));
#endif
                                //double residual=TransfUtils<ImageType>::computeDeformationNorm(circle,1.0);
                                double residual=TransfUtils<ImageType>::computeDeformationNormMask(circle,mask,1.0);
                                averageResidual+=residual;
                                double w=exp(-residual/m_sigma);
                                LOGV(3)<<"hop"<<h<<" "<<VAR(residual)<<" "<<VAR(w)<<endl;
                                weight+=w;



#ifdef AVERAGE
                                
#ifdef LOCALWEIGHTING
                                sourceTarget=TransfUtils<ImageType>::locallyScaleDeformation(sourceTarget,localDeformationNormWeightsSourceTarget);                     
#else
                                ImageUtils<DeformationFieldType>::multiplyImage(sourceTarget,w);
#endif                

                                if (count){
                                    typename DeformationAddFilterType::Pointer adder=DeformationAddFilterType::New();
                                    adder->InPlaceOff();
                                    adder->SetInput1(sourceTargetAccumulator);
                                    adder->SetInput2(sourceTarget);
                                    adder->Update();
                                    sourceTargetAccumulator=adder->GetOutput();
#ifdef LOCALWEIGHTING
                                    sourceTargetWeightAccumulator=FilterUtils<FloatImageType>::add(sourceTargetWeightAccumulator,localDeformationNormWeightsSourceTarget);
#endif
                                    
                                }else{
                                    sourceTargetAccumulator=sourceTarget;
#ifdef LOCALWEIGHTING

                                    sourceTargetWeightAccumulator=localDeformationNormWeightsSourceTarget;
#endif
                                }
#else
                                if (residual<bestResidual){

                                    sourceTargetAccumulator=sourceTarget;
                                    bestResidual=residual;
                                }
                                
#endif
                                count++;
                                circles++;


                                if (trueDefListFilename!=""){
                                    //DeformationFieldPointerType trueError12=TransfUtils<ImageType>::subtract(deformationCache[sourceID][intermediateID],trueDeformations[sourceID][intermediateID]);
                                    //DeformationFieldPointerType trueError23=TransfUtils<ImageType>::subtract(deformationCache[intermediateID][targetID],trueDeformations[intermediateID][targetID]);
                                    DeformationFieldPointerType trueError31=TransfUtils<ImageType>::subtract(deformationCache[targetID][sourceID],trueDeformations[targetID][sourceID]);
                                    //trueResidual+=TransfUtils<ImageType>::computeDeformationNorm(trueError12,1);
                                    //trueResidual+=TransfUtils<ImageType>::computeDeformationNorm(trueError23,1);
                                    trueResidual+=TransfUtils<ImageType>::computeDeformationNorm(trueError31,1);
                            }
                            }//if
                        }//intermediate image

#ifdef AVERAGE
#ifdef LOCALWEIGHTING
                        sourceTargetAccumulator=  TransfUtils<ImageType>::locallyInvertScaleDeformation(sourceTargetAccumulator,sourceTargetWeightAccumulator);
#else
                        ImageUtils<DeformationFieldType>::multiplyImage(sourceTargetAccumulator,alpha/weight);
#endif
#else
                        ImageUtils<DeformationFieldType>::multiplyImage(sourceTargetAccumulator,alpha);

#endif
                      

                        DeformationFieldPointerType tmp=ImageUtils<DeformationFieldType>::multiplyImageOutOfPlace(deformationCache[sourceID][targetID],1.0-alpha);
                        typename DeformationAddFilterType::Pointer adder=DeformationAddFilterType::New();
                        adder->InPlaceOff();
                        adder->SetInput1(tmp);
                        adder->SetInput2(sourceTargetAccumulator);
                        adder->Update();
                        TMPdeformationCache[sourceID][targetID]=adder->GetOutput();
                        globalResidual+=1.0*averageResidual;
                    }//if
                   
                }//target images
              
            }//source images
            globalResidual/=circles;
            trueResidual/=circles;
            LOG<<VAR(circles)<<" "<<VAR(globalResidual)<<" "<<VAR(trueResidual)<<endl;
            for (ImageListIteratorType sourceImageIterator=inputImages->begin();sourceImageIterator!=inputImages->end();++sourceImageIterator){           
                //iterate over sources
                string sourceID= sourceImageIterator->first;
                for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){                //iterate over targets
                    string targetID= targetImageIterator->first;
                    if (targetID !=sourceID){
                        
                        ostringstream tmpSegmentationFilename;
                        tmpSegmentationFilename<<outputDir<<"/registration-from-"<<sourceID<<"-TO-"<<targetID<<"-hop"<<h<<".mha";
                        ImageUtils<DeformationFieldType>::writeImage(tmpSegmentationFilename.str().c_str(),deformationCache[sourceID][targetID]);
                        deformationCache[sourceID][targetID]= TMPdeformationCache[sourceID][targetID];

                    }
                }
            }

        }//hops
        LOG<<"done"<<endl;


            LOG<<"done"<<endl;
            LOG<<"Storing output. and checking convergence"<<endl;
            for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){
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
  

    double localMAD(ImageNeighborhoodIteratorPointerType tIt, ImageNeighborhoodIteratorPointerType aIt,ImageNeighborhoodIteratorPointerType mIt){
        double result=0;
        int count=0;
        for (unsigned int i=0;i<tIt->Size();++i){
            if (mIt->GetPixel(i)){
                result+=fabs(tIt->GetPixel(i)-aIt->GetPixel(i));
                count++;
            }
        }
        if (!count)
            return 1.0;
        return exp(-result/count/m_sigma);
    }
    double localMSD(ImageNeighborhoodIteratorPointerType tIt, ImageNeighborhoodIteratorPointerType aIt,ImageNeighborhoodIteratorPointerType mIt){
        double result=0;
        int count=0;
        for (unsigned int i=0;i<tIt->Size();++i){
            if (mIt->GetPixel(i)){
                double tmp=(tIt->GetPixel(i)-aIt->GetPixel(i));
                result+=tmp*tmp;
                count++;
            }
        }
        if (!count)
            return 1.0;
        return  exp(-result/count/(m_sigma*m_sigma));
    }
    double localNCC(ImageNeighborhoodIteratorPointerType tIt, ImageNeighborhoodIteratorPointerType aIt,ImageNeighborhoodIteratorPointerType mIt){
        double result=0;
        int count=0;
        double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
        for (unsigned int i=0;i<tIt->Size();++i){
            if (mIt->GetPixel(i)){
                double f=tIt->GetPixel(i);
                double m= aIt->GetPixel(i);
                sff+=f*f;
                smm+=m*m;
                sfm+=f*m;
                sf+=f;
                sm+=m;
                count+=1;
            }
        }
        if (!count)
            return 0.5;
        else{
            double NCC=0;
            sff -= ( sf * sf / count );
            smm -= ( sm * sm / count );
            sfm -= ( sf * sm / count );
            if (smm*sff>0){
                NCC=1.0*sfm/sqrt(smm*sff);
            }
            result=(1.0+NCC)/2;
        }
        return result;
    }

    double globalMAD(ImagePointerType target, ImagePointerType moving, DeformationFieldPointerType deformation){
        std::pair<ImagePointerType,ImagePointerType> deformedMoving = TransfUtils<ImageType>::warpImageWithMask(moving,deformation);
        ImageIteratorType tIt(target,target->GetLargestPossibleRegion());
        ImageIteratorType mIt(deformedMoving.first,deformedMoving.first->GetLargestPossibleRegion());
        ImageIteratorType maskIt(deformedMoving.second,deformedMoving.second->GetLargestPossibleRegion());
        tIt.GoToBegin();mIt.GoToBegin();maskIt.GoToBegin();
        double result=0.0;int count=0;
        for (;!tIt.IsAtEnd();++tIt,++mIt,++maskIt){
            if (maskIt.Get()){
                result+=fabs(tIt.Get()-mIt.Get());
                count++;
            }
        }
        if (count)
            return exp(-result/count/m_sigma);
        else
            return 0.0;
    }
    
};//class
