
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
#include "Metrics.h"
#include "itkLabelOverlapMeasuresImageFilter.h"
#include "MRFRegistrationFuser.h"
#include <itkDisplacementFieldJacobianDeterminantFilter.h>

using namespace std;

template <class ImageType>
class RegistrationPropagationMRF{
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

    typedef MRFRegistrationFuser<ImageType> RegistrationFuserType;

    typedef typename itk::AddImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationAddFilterType;
    typedef map<string,ImagePointerType> ImageCacheType;
    typedef map<string, map< string, string> > FileListCacheType;
    enum MetricType {NONE,MAD,NCC,MI,NMI,MSD};
    enum WeightingType {UNIFORM,GLOBAL,LOCAL};
protected:
    double m_sigma;
    RadiusType m_patchRadius;
public:
    int run(int argc, char ** argv){
        feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
        argstream * as=new argstream(argc,argv);
        string trueDefListFilename="",landmarkFileList="",groundTruthSegmentationFileList="",deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir="",outputSuffix="",weightListFilename="";
        int verbose=0;
        double pWeight=1.0;
        int radius=3;
        int controlGridSpacingFactor=4;
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
        double m_pairwiseWeight=1.0;
        bool useHardConstraints=false;
        bool m_refineSolution=false;
        bool estimateMRF=false,estimateMean=false;
        int refineIter=0;
        //(*as) >> parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", true);
        (*as) >> option ("MRF", estimateMRF, "use MRF fusion");
        (*as) >> option ("mean", estimateMean, "use (local) mean fusion. Can be used in addition to MRF or stand-alone.");
        (*as) >> parameter ("T", deformationFileList, " list of deformations", true);
        (*as) >> parameter ("i", imageFileList, " list of  images", true);
        (*as) >> parameter ("true", trueDefListFilename, " list of TRUE deformations", false);
        //(*as) >> parameter ("W", weightListFilename,"list of weights for deformations",false);
        (*as) >> parameter ("metric", metricName,"metric to be used for global or local weighting, valid: NONE,SAD,MSD,NCC,MI,NMI",false);
        (*as) >> parameter ("weighting", weightingName,"internal weighting scheme {uniform,local,global}. non-uniform will only work with metric != NONE",false);
        (*as) >> parameter ("s", m_sigma,"sigma for exp(- metric/sigma)",false);
        (*as) >> parameter ("w", m_pairwiseWeight,"pairwise weight for MRF",false);
        (*as) >> parameter ("radius", radius,"patch radius for local metrics",false);
        (*as) >> parameter ("controlGridSpacing", controlGridSpacingFactor,"Resolution of the MRF control grid, obtained by dividing the input image resolution by this factor.",false);
        (*as) >> option ("hardConstraints", useHardConstraints,"Use hard constraints in the MRF to prevent folding.");
        (*as) >> parameter ("refineIter", refineIter,"refine MRF solution by adding the result as new labels to the MRF and re-solving until convergence.",false);
        (*as) >> parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
        (*as) >> parameter ("maxHops", maxHops,"maximum number of hops",false);
        (*as) >> parameter ("alpha", alpha,"pairwise balancing weight (spatial vs label smoothness)",false);
        (*as) >> option ("dontCacheDeformations", dontCacheDeformations,"read deformations only when needed to save memory. higher IO load!");
        (*as) >> parameter ("groundTruthSegmentations",groundTruthSegmentationFileList , "list of groundTruth segmentations <id> <file>", false);
        (*as) >> parameter ("landmarks",landmarkFileList , "list of landmark files <id> <file>", false);       
        //        (*as) >> option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
        //(*as) >> parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
        (*as) >> parameter ("verbose", verbose,"get verbose output",false);
        (*as) >> help();
        as->defaultErrorHandling();
       
        if (!estimateMRF && !estimateMean){
            LOG<<"Neither MRF nor mean estimation activated, assuming MRF estimation is desired!"<<endl;
            LOG<<"Neither MRF nor mean estimation activated, assuming MRF estimation is desired!"<<endl;
            LOG<<"Neither MRF nor mean estimation activated, assuming MRF estimation is desired!"<<endl;
            //estimateMRF=true;
        }
        LOG<<VAR(estimateMRF)<<" "<<VAR(estimateMean)<<endl;
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
                        LOGV(1)<<intermediateID<<" or "<<targetID<<" not in image database, skipping"<<endl;
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
                        LOGV(1)<<intermediateID<<" or "<<targetID<<" not in image database, skipping"<<endl;
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

        //read landmark filenames
        map<string,string> m_landmarkFileList;
        if (landmarkFileList!=""){
            ifstream ifs(landmarkFileList.c_str());
            while (!ifs.eof()){
                string sourceID,landmarkFilename;
                ifs >> sourceID;
                ifs >>landmarkFilename;
                m_landmarkFileList[sourceID]=landmarkFilename;
            }
            
        }
        
        ImageCacheType m_groundTruthSegmentations;
        if (groundTruthSegmentationFileList!=""){
            std::vector<string> buff;
            m_groundTruthSegmentations=ImageUtils<ImageType>::readImageList(groundTruthSegmentationFileList,buff);
        }

        double error;
        double inconsistency;
        error=TransfUtils<ImageType>::computeError(&deformationCache,&trueDeformations,&imageIDs);
        //inconsistency = TransfUtils<ImageType>::computeInconsistency(&deformationCache,&imageIDs, &trueDeformations);
        int iter = 0;
        LOG<<VAR(iter)<<" "<<VAR(error)<<" "<<VAR(inconsistency)<<endl;
        
        double m_oldEnergy=std::numeric_limits<double>::max();
        double m_oldSimilarity=std::numeric_limits<double>::max();
        logSetStage("Zero Hop");
        LOGV(1)<<"Computing"<<std::endl;
        for (iter=1;iter<maxHops+1;++iter){
            map< string, map <string, DeformationFieldPointerType> > TMPdeformationCache;
            int circles=0,count=0;
            double globalResidual=0.0;
            double trueResidual=0.0;
            double m_dice=0.0;
            double m_TRE=0.0;
            double m_energy=0.0;
            double m_similarity=0.0;
            double m_averageMinJac=0;
            double m_minMinJacobian=100000000;

            for (ImageListIteratorType sourceImageIterator=inputImages.begin();sourceImageIterator!=inputImages.end();++sourceImageIterator){           
                //iterate over sources
                string sourceID= sourceImageIterator->first;
                for (ImageListIteratorType targetImageIterator=inputImages.begin();targetImageIterator!=inputImages.end();++targetImageIterator){                //iterate over targets
                    string targetID= targetImageIterator->first;
                    if (targetID !=sourceID){
                        ++count;
                        DeformationFieldPointerType result;
                        DeformationFieldPointerType deformationSourceTarget;
                        deformationSourceTarget = deformationCache[sourceID][targetID];
                        if (estimateMRF || estimateMean){

                            RegistrationFuserType estimator;
                            estimator.setAlpha(alpha);
                            estimator.setPairwiseWeight(m_pairwiseWeight);
                            estimator.setGridSpacing(controlGridSpacingFactor);
                            estimator.setHardConstraints(useHardConstraints);
                       
                            GaussianEstimatorVectorImage<ImageType> meanEstimator;
                            if (radius>0){
                                ImagePointerType warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImageIterator->second,deformationSourceTarget);
                                FloatImagePointerType metric=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                FilterUtils<FloatImageType>::lowerThresholding(metric,0.0001);
                                if (estimateMRF)
                                    estimator.addImage(deformationSourceTarget,metric);
                                if (estimateMean)
                                    meanEstimator.addImage(deformationSourceTarget,metric);
                            }else{
                                if (estimateMRF)
                                    estimator.addImage(deformationSourceTarget);
                                if (estimateMean)
                                    meanEstimator.addImage(deformationSourceTarget);

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
                                        FloatImagePointerType metricImage;
                                        switch(metric){
                                        case NCC:
                                            metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                            break;
                                        case MSD:
                                            metricImage=Metrics<ImageType,FloatImageType>::LSSDNorm(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                            break;
                                        case MAD:
                                            metricImage=Metrics<ImageType,FloatImageType>::LSADNorm(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                            break;
                                        default:
                                            metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImageIterator->second,radius,m_sigma);

                                        }
                                        FilterUtils<FloatImageType>::lowerThresholding(metricImage,0.0001);
                                        if (estimateMRF)
                                            estimator.addImage(indirectDef,metricImage);
                                        if (estimateMean)
                                            meanEstimator.addImage(indirectDef,metricImage);
                                    }else{
                                        if (estimateMRF)
                                            estimator.addImage(indirectDef);
                                        if (estimateMean)
                                            meanEstimator.addImage(indirectDef);
                                    }


                                }//if
                            }//intermediate image
                            ImagePointerType labelImage;
                            
                            if (estimateMean)
                                meanEstimator.finalize();

                            if (! estimateMRF){
                                DeformationFieldPointerType meanDef=meanEstimator.getMean();
                                result=meanDef;
                            }else{

                                if (estimateMean){
                                    DeformationFieldPointerType meanDef=meanEstimator.getMean();
                                    if (radius>0){
                                        ImagePointerType warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImageIterator->second,meanDef);
                                        FloatImagePointerType metricImage;
                                        switch(metric){
                                        case NCC:
                                            metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                            break;
                                        case MSD:
                                            metricImage=Metrics<ImageType,FloatImageType>::LSSDNorm(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                            break;
                                        case MAD:
                                            metricImage=Metrics<ImageType,FloatImageType>::LSADNorm(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                            break;
                                        default:
                                            metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                    
                                        }
                                        FilterUtils<FloatImageType>::lowerThresholding(metricImage,0.0001);
                                        estimator.addImage(meanDef,metricImage);
                                    }else{
                                        estimator.addImage(meanDef);
                                    }
                                }
                                
                        
                                double energy=estimator.finalize(labelImage);
                                result=estimator.getMean();
                                LOGV(1)<<VAR(energy)<<endl;
                                for (int iter=0;iter<refineIter;++iter){
                                    //estimator.setAlpha(pow(2.0,1.0*iter)*alpha);

                                        ImagePointerType warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImageIterator->second,result);
                                        FloatImagePointerType metricImage;
                                        switch(metric){
                                        case NCC:
                                            metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                            break;
                                        case MSD:
                                            metricImage=Metrics<ImageType,FloatImageType>::LSSDNorm(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                            break;
                                        case MAD:
                                            metricImage=Metrics<ImageType,FloatImageType>::LSADNorm(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                            break;
                                        default:
                                            metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImageIterator->second,radius,m_sigma);
                                        }
                                        FilterUtils<FloatImageType>::lowerThresholding(metricImage,0.0001);
                                        estimator.addImage(result,metricImage);
                                        double newEnergy=estimator.finalize(labelImage);
                                        LOGV(1)<<VAR(iter)<<" "<<VAR(newEnergy)<<" "<<(energy-newEnergy)/energy<<endl;
                                        if (newEnergy >energy )
                                            break;
                                        result=estimator.getMean();
                                        if ( (energy-newEnergy)/energy < 1e-4) {
                                            LOGV(1)<<"refinement converged, stopping."<<endl;
                                            break;
                                        }
                                        energy=newEnergy;

                                    }
                                estimator.setAlpha(alpha);

                                m_energy+=energy;
                            }
                            if (outputDir!=""){
                                ostringstream oss;
                                oss<<outputDir<<"/avgDeformation-"<<sourceID<<"-TO-"<<targetID<<".mha";
                                ImageUtils<DeformationFieldType>::writeImage(oss.str(),result);

                                ostringstream oss2;
                                oss2<<outputDir<<"/labelImage-"<<sourceID<<"-TO-"<<targetID<<".nii";
                                ImageUtils<ImageType>::writeImage(oss2.str(),labelImage);

                            }
                        }else{
                            result=deformationSourceTarget;
                        }//if (estimateMean || estimateMRF)


                        TMPdeformationCache[sourceID][targetID]=result;

                        // compare landmarks
                        if (m_landmarkFileList.size()){
                            //hope that all landmark files are available :D
                            m_TRE+=TransfUtils<ImageType>::computeTRE(m_landmarkFileList[targetID], m_landmarkFileList[sourceID],result,targetImageIterator->second);
                        }
                        
                        if (m_groundTruthSegmentations[targetID].IsNotNull() && m_groundTruthSegmentations[sourceID].IsNotNull()){
                            ImagePointerType deformedSeg=TransfUtils<ImageType>::warpImage(  m_groundTruthSegmentations[sourceID] , result ,true);
                            typedef typename itk::LabelOverlapMeasuresImageFilter<ImageType> OverlapMeasureFilterType;
                            typename OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
                            filter->SetSourceImage((m_groundTruthSegmentations)[targetID]);
                            filter->SetTargetImage(deformedSeg);
                            filter->SetCoordinateTolerance(1e-4);
                            filter->Update();
                            double dice=filter->GetDiceCoefficient();
                            LOGV(1)<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(dice)<<endl;
                            m_dice+=dice;
                        }
                        
                        double similarity;
                        ImagePointerType warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImageIterator->second,result);
                        switch(metric){
                        case NCC:
                            similarity=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImageIterator->second);
                            break;
                        case MSD:
                            similarity=Metrics<ImageType,FloatImageType>::msd(warpedSourceImage,targetImageIterator->second);
                            break;
                        case MAD:
                            similarity=Metrics<ImageType,FloatImageType>::mad(warpedSourceImage,targetImageIterator->second);
                            break;
                        }
                        m_similarity+=similarity;
                        //create mask of valid deformation region
                        
                        typedef typename itk::DisplacementFieldJacobianDeterminantFilter<DeformationFieldType,float> DisplacementFieldJacobianDeterminantFilterType;
                        typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
                        jacobianFilter->SetInput(result);
                        jacobianFilter->SetUseImageSpacingOff();
                        jacobianFilter->Update();
                        FloatImagePointerType jac=jacobianFilter->GetOutput();
                        double minJac = FilterUtils<FloatImageType>::getMin(jac);
                        //LOGV(2)<<VAR(sourceID)<<" "<<VAR(targetID)<< " " << VAR(minJac) <<" " <<VAR(nCC)<<endl;
                        m_averageMinJac+=minJac;
                        if (minJac<m_minMinJacobian){
                            m_minMinJacobian=minJac;
                        }
                        
                    }//if

                   
                }//target images
              
            }//source images
            m_dice/=count;
            m_TRE/=count;
            m_energy/=count;
            m_similarity/=count;
            m_averageMinJac/=count;
            error=TransfUtils<ImageType>::computeError(&TMPdeformationCache,&trueDeformations,&imageIDs);
            //inconsistency = TransfUtils<ImageType>::computeInconsistency(&deformationCache,&imageIDs,&trueDeformations);
            LOG<<VAR(iter)<<" "<<VAR(error)<<" "<<VAR(inconsistency)<<" "<<VAR(m_TRE)<<" "<<VAR(m_dice)<<" "<<VAR(m_energy)<<" "<<VAR(m_similarity)<<" "<<VAR(m_averageMinJac)<<" "<<VAR(m_minMinJacobian)<<endl;
            if (m_similarity>m_oldSimilarity){
                LOG<<"Similarity increased, stopping and returning previous estimate."<<endl;
                break;
            }else if ( fabs((m_oldSimilarity-m_similarity)/m_oldSimilarity) < 1e-3) { //else if ( (m_oldEnergy-m_energy)/m_oldEnergy < 1e-2) {
                LOG<<"Optimization converged, stopping."<<endl;
                deformationCache=TMPdeformationCache;
                break;
            }
            m_oldEnergy=m_energy;
            m_oldSimilarity=m_similarity;
            deformationCache=TMPdeformationCache;
          

        }//hops
        
        LOG<<"Storing output."<<endl;
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
