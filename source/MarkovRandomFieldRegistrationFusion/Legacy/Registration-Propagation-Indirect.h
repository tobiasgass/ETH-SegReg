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
#include "itkGaussianImage.h"

using namespace std;

template <class ImageType>
class RegistrationPropagationIndirect{
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
        bool localSimilarityWeighting=false;
        bool graphCut=false;
        double smoothness=1.0;
        double alpha=0.5;
        m_sigma=30;
        double m_sigmaSmooth=0.0;
        bool gaussianReweight=false;
        int maxIter=1;
        //as->parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", true);
        as->parameter ("T", deformationFileList, " list of deformations", true);
        as->parameter ("i", imageFileList, " list of  images", true);
        as->parameter ("true", trueDefListFilename, " list of TRUE deformations", false);
        //as->parameter ("W", weightListFilename,"list of weights for deformations",false);
        //as->parameter ("metric", metricName,"metric to be used for global or local weighting, valid: NONE,SAD,MSD,NCC,MI,NMI",false);
        //as->parameter ("weighting", weightingName,"internal weighting scheme {uniform,local,global}. non-uniform will only work with metric != NONE",false);
        as->parameter ("s", m_sigma,"sigma for exp(- metric/sigma)",false);
        as->parameter ("smooth", m_sigmaSmooth,"kernel size for spatial smoothing of mean-shift weighting weighting",false);
        //as->parameter ("radius", radius,"patch radius for local metrics",false);
        as->parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
        //as->parameter ("radius", radius,"patch radius for NCC",false);
        as->parameter ("maxHops", maxHops,"maximum number of hops",false);
        as->parameter ("maxIter", maxIter,"maximum number of mean shift iterations",false);
        as->parameter ("alpha", alpha,"update rate",false);
        as->option ("lateFusion", lateFusion,"fuse segmentations late. maxHops=1");
        as->option ("dontCacheDeformations", dontCacheDeformations,"read deformations only when needed to save memory. higher IO load!");
        as->option ("gaussianReweight", gaussianReweight,"Use reweighted mean for reconstruction");
        as->option ("localSimWeight", localSimilarityWeighting,"Use local similarities (LNCC) as weights for mean/gaussian estimation.");
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
            double oldResidual=0.0;
            int count=0;

            for (ImageListIteratorType sourceImageIterator=inputImages->begin();sourceImageIterator!=inputImages->end();++sourceImageIterator){           
                //iterate over sources
                string sourceID= sourceImageIterator->first;
                for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){                //iterate over targets
                    string targetID= targetImageIterator->first;
                    if (targetID !=sourceID){

                        DeformationFieldPointerType deformationSourceTarget;
                        if (dontCacheDeformations){
                            deformationSourceTarget=ImageUtils<DeformationFieldType>::readImage( deformationFilenames[sourceID][targetID]);
                        }
                        else{
                            deformationSourceTarget = deformationCache[sourceID][targetID];
                        }
                        
                        //initialize accumulators
                        DeformationFieldPointerType avgIndirectDeformation;
                        double finalResidual;
                        bool converged = false ; 
                        LocalVectorMeanShift<ImageType> meansShiftEstimator;
                        meansShiftEstimator.setSigma(m_sigma);

                        for ( int iter = 0; iter < maxIter && !converged; ++iter) {
                            double residual=0.0;
                            int localCount=0;
                            ImagePointerType localCountsIndirect;
                            if (iter){
                                meansShiftEstimator.addImage(deformationSourceTarget);
                                                                        
                            }else{
                                if (gaussianReweight){
                                    //initialize mean shift with mean deform
                                    FastGaussianEstimatorVectorImage<ImageType> gaussEstimator;
                                    gaussEstimator.initialize(deformationSourceTarget);
                                    for (ImageListIteratorType intermediateImageIterator=inputImages->begin();intermediateImageIterator!=inputImages->end();++intermediateImageIterator){                //iterate over intermediates
                                        string intermediateID= intermediateImageIterator->first;
                                        if (targetID != intermediateID && sourceID!=intermediateID){
                                            LOGV(3)<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(intermediateID)<<endl;
                                            //get all deformations for full circle
                                            DeformationFieldPointerType deformationSourceIntermed;
                                            DeformationFieldPointerType deformationIntermedTarget;
                                            if (dontCacheDeformations){
                                                deformationSourceIntermed=ImageUtils<DeformationFieldType>::readImage(deformationFilenames[sourceID][intermediateID]);
                                                deformationIntermedTarget=ImageUtils<DeformationFieldType>::readImage(deformationFilenames[intermediateID][targetID]);
                                            }
                                            else{
                                                deformationSourceIntermed = deformationCache[sourceID][intermediateID];
                                                deformationIntermedTarget = deformationCache[intermediateID][targetID];
                                            }

                                            //create mask of valid deformation region
                                            ImagePointerType mask=ImageUtils<ImageType>::createEmpty(sourceImageIterator->second);
                                            mask->FillBuffer(1);
                                            
                                            //compute indirect path
                                            DeformationFieldPointerType sourceTargetIndirect=TransfUtils<ImageType>::composeDeformations(deformationIntermedTarget,deformationSourceIntermed);
                                            //add to accumulator
                                            gaussEstimator.addImage(sourceTargetIndirect);
                                        }//if
                                    }//intermediate image
                                    gaussEstimator.finalize();
                                    //meansShiftEstimator.initialize(gaussEstimator.getMean(),gaussEstimator.getVariance());
                                    //meansShiftEstimator.initialize(gaussEstimator.getMean());
                                    meansShiftEstimator.initialize(deformationSourceTarget,gaussEstimator.getVariance());

                                }else{
                                    //initialize with direct deform
                                    meansShiftEstimator.initialize(deformationSourceTarget);
                                }
                            }
                         
                            for (ImageListIteratorType intermediateImageIterator=inputImages->begin();intermediateImageIterator!=inputImages->end();++intermediateImageIterator){                //iterate over intermediates
                                string intermediateID= intermediateImageIterator->first;
                                if (targetID != intermediateID && sourceID!=intermediateID){
                                    LOGV(3)<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(intermediateID)<<endl;
                                    //get all deformations for full circle
                                    DeformationFieldPointerType deformationSourceIntermed;
                                    DeformationFieldPointerType deformationIntermedTarget;
                                    if (dontCacheDeformations){
                                        deformationSourceIntermed=ImageUtils<DeformationFieldType>::readImage(deformationFilenames[sourceID][intermediateID]);
                                        deformationIntermedTarget=ImageUtils<DeformationFieldType>::readImage(deformationFilenames[intermediateID][targetID]);
                                    }
                                    else{
                                        deformationSourceIntermed = deformationCache[sourceID][intermediateID];
                                        deformationIntermedTarget = deformationCache[intermediateID][targetID];
                                    }

                              

                                    //create mask of valid deformation region
                                    ImagePointerType mask=ImageUtils<ImageType>::createEmpty(sourceImageIterator->second);
                                    mask->FillBuffer(1);
                              
                                    //compute indirect path
                                    DeformationFieldPointerType sourceTargetIndirect=TransfUtils<ImageType>::composeDeformations(deformationIntermedTarget,deformationSourceIntermed);
                                    //add to accumulator
                                    meansShiftEstimator.addImage(sourceTargetIndirect);
                                
                                    double diffNorm=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract(meansShiftEstimator.getMean(),sourceTargetIndirect),1.0);
                                    residual+=diffNorm;
                                    localCount++;
                                }//if
                            }//intermediate image
                            converged=meansShiftEstimator.finalize(1e-1);
                            finalResidual=residual/localCount;
                        }
                        if (maxIter>0)
                            avgIndirectDeformation=meansShiftEstimator.getMean();
                        else
                            avgIndirectDeformation=deformationSourceTarget;
                        //store deformation
                        ostringstream tmpSegmentationFilename;
                        tmpSegmentationFilename<<outputDir<<"/registration-from-"<<sourceID<<"-TO-"<<targetID<<"-hop"<<h+1<<".mha";
                        ImageUtils<DeformationFieldType>::writeImage(tmpSegmentationFilename.str().c_str(),avgIndirectDeformation);
                        if (!dontCacheDeformations){
                            TMPdeformationCache[sourceID][targetID]=avgIndirectDeformation;
                            
                        }

                        
#if 0      
                        //2. compute difference
                        DeformationFieldPointerType deltaNullSourceTarget=TransfUtils<ImageType>::subtract(deformationSourceTarget,avgIndirectDeformation);
                        ostringstream weakEst;
                        weakEst<<outputDir<<"/error-weakEstimate-from-"<<sourceID<<"-TO-"<<targetID<<"-hop"<<h<<".png";
                        ImageUtils<ImageType>::writeImage(weakEst.str().c_str(),FilterUtils<FloatImageType,ImageType>::truncateCast(TransfUtils<ImageType>::computeLocalDeformationNorm(deltaNullSourceTarget,m_sigma)));
#endif                   
                            
                        //TMPdeformationCache[sourceID][targetID]=avgIndirecDeformation;
                        //globalResidual+=1.0*averageResidual;
                        if (trueDefListFilename!=""){
                              //create mask of valid deformation region
                            ImagePointerType mask=ImageUtils<ImageType>::createEmpty(sourceImageIterator->second);
                            mask->FillBuffer(1);
                            ImagePointerType defMask1=TransfUtils<ImageType>::warpImage(mask,deformationCache[sourceID][targetID],true);
                            DeformationFieldPointerType trueErrorSourceTarget=TransfUtils<ImageType>::subtract(deformationCache[sourceID][targetID],trueDeformations[sourceID][targetID]);
                            double prevError=TransfUtils<ImageType>::computeDeformationNormMask(trueErrorSourceTarget,defMask1,1);
                            oldResidual+=prevError;
                            trueErrorSourceTarget=TransfUtils<ImageType>::subtract(avgIndirectDeformation,trueDeformations[sourceID][targetID]);
                            ImagePointerType defMask2=TransfUtils<ImageType>::warpImage(mask,avgIndirectDeformation,true);
                            double newError=TransfUtils<ImageType>::computeDeformationNormMask(trueErrorSourceTarget,defMask2,1);
                            trueResidual+=newError;

                            LOGV(1)<<VAR(prevError)<<" " <<VAR(newError)<<endl;
#if 1
                            ostringstream trueErrorST;
                            trueErrorST<<outputDir<<"/error-TRUE-from-"<<sourceID<<"-TO-"<<targetID<<"-hop"<<h<<".png";
                            ImageUtils<ImageType>::writeImage(trueErrorST.str().c_str(),FilterUtils<FloatImageType,ImageType>::truncateCast(TransfUtils<ImageType>::computeLocalDeformationNorm(trueErrorSourceTarget,m_sigma)));
#endif
                        
                        }
                        globalResidual+=finalResidual;
                        count++;

                    }//if
                   
                }//target images
              
            }//source images
            globalResidual/=count;
            trueResidual/=count;
            oldResidual/=count;
            LOG<<VAR(count)<<" "<<VAR(globalResidual)<<" "<<VAR(trueResidual)<<" "<<VAR(oldResidual)<<endl;
#if 0
            for (ImageListIteratorType sourceImageIterator=inputImages->begin();sourceImageIterator!=inputImages->end();++sourceImageIterator){           
                //iterate over sources
                string sourceID= sourceImageIterator->first;
                for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){                //iterate over targets
                    string targetID= targetImageIterator->first;
                    if (targetID !=sourceID){
#if 0
                        ostringstream tmpdeformed;
                        tmpdeformed<<outputDir<<"/deformed-from-"<<sourceID<<"-TO-"<<targetID<<"-hop"<<h<<".png";
                        ImageUtils<ImageType>::writeImage(tmpdeformed.str().c_str(),TransfUtils<ImageType>::warpImage((*inputImages)[sourceID],deformationCache[sourceID][targetID]));
#endif
                                               
                        if (!dontCacheDeformations){
                            deformationCache[sourceID][targetID]= TMPdeformationCache[sourceID][targetID];
                        }

                    }
                }
            }
#endif

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
