
#pragma once

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


using namespace std;

template <class ImageType>
class RegistrationCircles{
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
    typedef typename DeformationAddFilterType::Pointer DeformationAddFilterPointer;
    typedef typename itk::SubtractImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationSubtractFilterType;

    enum MetricType {NONE,MAD,NCC,MI,NMI,MSD};
    enum WeightingType {UNIFORM,GLOBAL,LOCAL};
protected:
    double m_sigma;
    RadiusType m_patchRadius;
public:
    int run(int argc, char ** argv){
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
        m_sigma=30;
        //(*as) >> parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", true);
        (*as) >> parameter ("T", deformationFileList, " list of deformations", true);
        (*as) >> parameter ("true", trueDefListFilename, " list of TRUE deformations", true);

        (*as) >> parameter ("i", imageFileList, " list of  images", true);
        //(*as) >> parameter ("W", weightListFilename,"list of weights for deformations",false);
        //(*as) >> parameter ("metric", metricName,"metric to be used for global or local weighting, valid: NONE,SAD,MSD,NCC,MI,NMI",false);
        //(*as) >> parameter ("weighting", weightingName,"internal weighting scheme {uniform,local,global}. non-uniform will only work with metric != NONE",false);
        (*as) >> parameter ("s", m_sigma,"sigma for exp(- metric/sigma)",false);
        //(*as) >> parameter ("radius", radius,"patch radius for local metrics",false);
        (*as) >> parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
        //(*as) >> parameter ("radius", radius,"patch radius for NCC",false);
        (*as) >> parameter ("maxHops", maxHops,"maximum number of hops",false);
        (*as) >> parameter ("alpha", alpha,"update rate",false);
        (*as) >> option ("lateFusion", lateFusion,"fuse segmentations late. maxHops=1");
        (*as) >> option ("dontCacheDeformations", dontCacheDeformations,"read deformations only when needed to save memory. higher IO load!");
        //        (*as) >> option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
        //(*as) >> parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
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
            int circles=0;
            double averageResidual=0.0;
            double trueResidual=0.0;
            for (ImageListIteratorType sourceImageIterator=inputImages->begin();sourceImageIterator!=inputImages->end();++sourceImageIterator){           
                //iterate over sources
                string sourceID= sourceImageIterator->first;
                ImageListIteratorType intermediateImageIterator=sourceImageIterator;
                intermediateImageIterator++;
                for (;intermediateImageIterator!=inputImages->end();++intermediateImageIterator){                //iterate over intermediates
                    string intermediateID= intermediateImageIterator->first;
                    ImageListIteratorType targetImageIterator=intermediateImageIterator;
                    targetImageIterator++;
                    for (;targetImageIterator!=inputImages->end();++targetImageIterator){                //iterate over targets
                        string targetID= targetImageIterator->first;
                        std::vector<string> dir(2);
                        dir[0]="forward";
                        dir[1]="backward";
                        for (int d=0;d<2;++d){
                            DeformationFieldPointerType sourceTargetAccumulator;
                            FloatImagePointerType sourceTargetWeightAccumulator;
                            double bestResidual=10000000.0;
                            int count=0;
                            double weight=0.0;
                            //get all deformations for full circles in direction $dir
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


                            //compute circle1
                            //deformatiom
                            DeformationFieldPointerType sourceTarget=TransfUtils<ImageType>::composeDeformations(deformationIntermedTarget,deformationSourceIntermed);
                            DeformationFieldPointerType circle1=TransfUtils<ImageType>::composeDeformations(deformationTargetSource,sourceTarget);
                            //mask
                            ImagePointerType mask1=TransfUtils<ImageType>::warpImage(mask,deformationSourceIntermed,true);
                            mask1=TransfUtils<ImageType>::warpImage(mask,deformationIntermedTarget,true);
                            mask1=TransfUtils<ImageType>::warpImage(mask1,deformationTargetSource,true);
                            //localNorms
                            FloatImagePointerType localDeformationNormWeights1=TransfUtils<ImageType>::computeLocalDeformationNormWeights(circle1,m_sigma);
                            ostringstream tmpSegmentationFilename;
                            tmpSegmentationFilename<<outputDir<<"/error-from-"<<sourceID<<"-VIA-"<<intermediateID<<"-TO-"<<targetID<<"-hop"<<h<<"-1231.png";
                            //    ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(localDeformationNormWeights1,65535)));
                            //store circle
                            ostringstream circle1Filename;
                            circle1Filename<<outputDir<<"/circleError-from-"<<sourceID<<"-VIA-"<<intermediateID<<"-TO-"<<targetID<<"-hop"<<h<<"-1231.mha";
                            //     ImageUtils<DeformationFieldType>::writeImage(circle1Filename.str().c_str(),circle1);
                            //residual
                            double residual1=TransfUtils<ImageType>::computeDeformationNormMask(circle1,mask1,1.0);
                            averageResidual+=residual1;
                            double w1=exp(-residual1/m_sigma);
                            weight+=w1;
                           
                            //compute circle2
                            //deformatiom
                            DeformationFieldPointerType intermedSource=TransfUtils<ImageType>::composeDeformations(deformationTargetSource,deformationIntermedTarget);
                            DeformationFieldPointerType circle2=TransfUtils<ImageType>::composeDeformations(deformationSourceIntermed,intermedSource);
                            //mask
                            ImagePointerType mask2=TransfUtils<ImageType>::warpImage(mask,deformationSourceIntermed,true);
                            mask2=TransfUtils<ImageType>::warpImage(mask,deformationIntermedTarget,true);
                            mask2=TransfUtils<ImageType>::warpImage(mask2,deformationTargetSource,true);
                            //localNorms
                            FloatImagePointerType localDeformationNormWeights2=TransfUtils<ImageType>::computeLocalDeformationNormWeights(circle2,m_sigma);
                            ostringstream tmpSegmentationFilename2;
                            tmpSegmentationFilename2<<outputDir<<"/error-from-"<<intermediateID<<"-VIA-"<<targetID<<"-TO-"<<sourceID<<"-hop"<<h<<"-2312.png";
                            ImageUtils<ImageType>::writeImage(tmpSegmentationFilename2.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(localDeformationNormWeights2,65535)));
                            //store circle
                            ostringstream circle2Filename;
                            circle2Filename<<outputDir<<"/circleError-from-"<<intermediateID<<"-VIA-"<<targetID<<"-TO-"<<sourceID<<"-hop"<<h<<"-2312.mha";
                            ImageUtils<DeformationFieldType>::writeImage(circle2Filename.str().c_str(),circle2);
                            //residual
                            double residual2=TransfUtils<ImageType>::computeDeformationNormMask(circle2,mask2,1.0);
                            averageResidual+=residual2;
                            double w2=exp(-residual2/m_sigma);
                            weight+=w2;
                        
                            //compute circle3
                            //deformatiom
                            DeformationFieldPointerType targetIntermed=TransfUtils<ImageType>::composeDeformations(deformationSourceIntermed,deformationTargetSource);
                            DeformationFieldPointerType circle3=TransfUtils<ImageType>::composeDeformations(deformationIntermedTarget,targetIntermed);
                            //mask
                            ImagePointerType mask3=TransfUtils<ImageType>::warpImage(mask,deformationSourceIntermed,true);
                            mask3=TransfUtils<ImageType>::warpImage(mask,deformationIntermedTarget,true);
                            mask3=TransfUtils<ImageType>::warpImage(mask3,deformationTargetSource,true);
                            //localNorms
                            FloatImagePointerType localDeformationNormWeights3=TransfUtils<ImageType>::computeLocalDeformationNormWeights(circle3,m_sigma);
                            ostringstream tmpSegmentationFilename3;
                            tmpSegmentationFilename3<<outputDir<<"/error-from-"<<targetID<<"-VIA-"<<sourceID<<"-TO-"<<intermediateID<<"-hop"<<h<<"-3123.png";
                            ImageUtils<ImageType>::writeImage(tmpSegmentationFilename3.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(localDeformationNormWeights3,65535)));
                            //store circle
                            ostringstream circle3Filename;
                            circle3Filename<<outputDir<<"/circleError-from-"<<targetID<<"-VIA-"<<sourceID<<"-TO-"<<intermediateID<<"-hop"<<h<<"-3123-1.mha";
                            ImageUtils<DeformationFieldType>::writeImage(circle3Filename.str().c_str(),circle3);
                            //residual
                            double residual3=TransfUtils<ImageType>::computeDeformationNormMask(circle3,mask3,1.0);
                            averageResidual+=residual3;
                            double w3=exp(-residual3/m_sigma);
                            weight+=w3;
                        
                            LOG<<VAR(residual1)<<" "<<VAR(residual2)<<" "<<VAR(residual3)<<endl;
#if 0
                            //deform error maps and store
                            FloatImagePointerType warp1=TransfUtils<FloatImageType>::warpImage(localDeformationNormWeights1,deformationSourceIntermed);
                            ostringstream warpedWeights1;
                            warpedWeights1<<outputDir<<"/warpedErrors-from-"<<sourceID<<"-TO-"<<intermediateID<<"-VIA-"<<targetID<<"-hop"<<h<<"-1231-2.png";
                            ImageUtils<ImageType>::writeImage(warpedWeights1.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(warp1,65535)));
                            FloatImagePointerType warp2=TransfUtils<FloatImageType>::warpImage(localDeformationNormWeights2,deformationIntermedTarget);
                            ostringstream warpedWeights2;
                            warpedWeights2<<outputDir<<"/warpedErrors-from-"<<intermediateID<<"-TO-"<<targetID<<"-VIA-"<<sourceID<<"-hop"<<h<<"-2312-3.png";
                            ImageUtils<ImageType>::writeImage(warpedWeights2.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(warp2,65535)));
                            FloatImagePointerType warp3=TransfUtils<FloatImageType>::warpImage(localDeformationNormWeights3,deformationTargetSource);
                            ostringstream warpedWeights3;
                            warpedWeights3<<outputDir<<"/warpedErrors-from-"<<targetID<<"-TO-"<<sourceID<<"-VIA-"<<intermediateID<<"-hop"<<h<<"-3123-1-.png";
                            ImageUtils<ImageType>::writeImage(warpedWeights3.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(warp3,65535)));

                            
                            //compute Differences between deformed circle residuals and circle residual
                            //circle1
                            DeformationFieldPointerType diffImage1=TransfUtils<ImageType>::subtract(circle2,TransfUtils<ImageType>::warpDeformation(circle1,deformationSourceIntermed));
                            FloatImagePointerType diff1=TransfUtils<ImageType>::computeLocalDeformationNormWeights(differ1->GetOutput(),m_sigma);
                            ostringstream warpedDiff1;
                            warpedDiff1<<outputDir<<"/warpedDiffErrors-from-"<<sourceID<<"-TO-"<<intermediateID<<"-VIA-"<<targetID<<"-hop"<<h<<"-1231-2.png";
                            ImageUtils<ImageType>::writeImage(warpedDiff1.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(diff1,65535)));
                            
                            //circle2
                            DeformationFieldPointerType diffImage2=TransfUtils<ImageType>::subtract(circle3,TransfUtils<ImageType>::warpDeformation(circle2,deformationIntermedTarget));
                            FloatImagePointerType diff2=TransfUtils<ImageType>::computeLocalDeformationNormWeights(differ2->GetOutput(),m_sigma);
                            ostringstream warpedDiff2;
                            warpedDiff2<<outputDir<<"/warpedDiffErrors-from-"<<intermediateID<<"-TO-"<<targetID<<"-VIA-"<<sourceID<<"-hop"<<h<<"-2312-3.png";
                            ImageUtils<ImageType>::writeImage(warpedDiff2.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(diff2,65535)));
                            
                            //circle3
                            DeformationFieldPointerType diffImage3=TransfUtils<ImageType>::subtract(circle1,TransfUtils<ImageType>::warpDeformation(circle3,deformationTargetSource));
                            FloatImagePointerType diff3=TransfUtils<ImageType>::computeLocalDeformationNormWeights(differ3->GetOutput(),m_sigma);
                            ostringstream warpedDiff3;
                            warpedDiff3<<outputDir<<"/warpedDiffErrors-from-"<<targetID<<"-TO-"<<sourceID<<"-VIA-"<<intermediateID<<"-hop"<<h<<"-3123-1.png";
                            ImageUtils<ImageType>::writeImage(warpedDiff3.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(diff3,65535)));
#endif
                            count++;
                            circles+=3;
                            
                            //accumulate error estimates
                            errorAccumulators[sourceID][intermediateID]=TransfUtils<ImageType>::add(circle1,errorAccumulators[sourceID][intermediateID]);
                            errorAccumulators[intermediateID][targetID]=TransfUtils<ImageType>::add(circle2,errorAccumulators[intermediateID][targetID]);
                            errorAccumulators[targetID][sourceID]=TransfUtils<ImageType>::add(circle3,errorAccumulators[targetID][sourceID]);

                            //caclutate true error
                            if (trueDefListFilename!=""){
                                DeformationFieldPointerType trueError1=TransfUtils<ImageType>::subtract(deformationCache[sourceID][intermediateID],trueDeformations[sourceID][intermediateID]);
                                DeformationFieldPointerType trueError2=TransfUtils<ImageType>::subtract(deformationCache[intermediateID][targetID],trueDeformations[intermediateID][targetID]);
                                DeformationFieldPointerType trueError3=TransfUtils<ImageType>::subtract(deformationCache[targetID][sourceID],trueDeformations[targetID][sourceID]);
                                trueResidual+=TransfUtils<ImageType>::computeDeformationNorm(trueError1,1);
                                trueResidual+=TransfUtils<ImageType>::computeDeformationNorm(trueError2,1);
                                trueResidual+=TransfUtils<ImageType>::computeDeformationNorm(trueError3,1);

                            }
                                                                                                    
                                                                                                    
                            //swap intermediate and target to invert circle direction
                            string tmp=intermediateID;
                            intermediateID=targetID;
                            targetID=tmp;


                        }//d
                          
                    
                    }//intermediate image
                    
                }//target images
            }//source images
            averageResidual/=circles;
            trueResidual/=circles;
            LOG<<VAR(circles)<<" "<<VAR(averageResidual)<<" "<<VAR(trueResidual)<<endl;
            for (ImageListIteratorType sourceImageIterator=inputImages->begin();sourceImageIterator!=inputImages->end();++sourceImageIterator){           
                //iterate over sources
                string sourceID= sourceImageIterator->first;
                for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){                //iterate over targets
                    string targetID= targetImageIterator->first;
                    if (targetID !=sourceID){
                        
                        

                        ostringstream tmpSegmentationFilename;
                        tmpSegmentationFilename<<outputDir<<"/registration-from-"<<sourceID<<"-TO-"<<targetID<<"-hop"<<h<<".mha";
                        ImageUtils<DeformationFieldType>::writeImage(tmpSegmentationFilename.str().c_str(),deformationCache[sourceID][targetID]);
                        
                        ImageUtils<DeformationFieldType>::multiplyImage(errorAccumulators[sourceID][targetID],1.0/(nImages-2));
                        ostringstream tmpErrorFilename;
                        tmpErrorFilename<<outputDir<<"/errorEstimate-from-"<<sourceID<<"-TO-"<<targetID<<"-hop"<<h<<".mha";
                        ImageUtils<DeformationFieldType>::writeImage(tmpErrorFilename.str().c_str(),errorAccumulators[sourceID][targetID]);
                        FloatImagePointerType errorNorm=TransfUtils<ImageType>::computeLocalDeformationNormWeights(errorAccumulators[sourceID][targetID],m_sigma);
                        ostringstream errorNormFilename;
                        errorNormFilename<<outputDir<<"/errorEstimate-from-"<<sourceID<<"-TO-"<<targetID<<"-hop"<<h<<".png";
                        ImageUtils<ImageType>::writeImage(errorNormFilename.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(errorNorm,65535)));
                           
                        ImageUtils<DeformationFieldType>::multiplyImage(errorAccumulators[sourceID][targetID],-alpha);
                        DeformationAddFilterPointer updateDef=DeformationAddFilterType::New();
                        updateDef->SetInput1(deformationCache[sourceID][targetID]);
                        updateDef->SetInput2(errorAccumulators[sourceID][targetID]);
                        updateDef->InPlaceOff();
                        updateDef->Update();
                        deformationCache[sourceID][targetID]=updateDef->GetOutput();
                        //reset accumulators
                        errorAccumulators[sourceID][targetID]->FillBuffer(zeroDisp);

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
