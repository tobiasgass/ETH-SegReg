#pragma once

#include <stdio.h>
#include <iostream>
#include "GCoptimization.h"
#include "ArgumentParser.h"
#include "Log.h"
#include <vector>
#include <map>
#include "itkImageRegionIterator.h"
#include "TransformationUtils.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "bgraph.h"
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
#include <itkLabelOverlapMeasuresImageFilter.h>
#include "Metrics.h"
#include "SegmentationMapper.hxx"
using namespace std;

template <class ImageType, int nSegmentationLabels>
class SegmentationPropagationModular{
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

    typedef  TransfUtils<ImageType,double> TransfUtilsType;

    typedef typename  TransfUtilsType::DisplacementType DisplacementType; typedef typename  TransfUtilsType::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename  itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef typename  itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
    typedef typename  itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
    typedef typename  itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
    typedef   ImageNeighborhoodIteratorType * ImageNeighborhoodIteratorPointerType;
    typedef typename  ImageNeighborhoodIteratorType::RadiusType RadiusType;

    typedef itk::Vector<float,nSegmentationLabels> ProbabilisticPixelType;
    typedef itk::Image<ProbabilisticPixelType,D> ProbabilisticVectorImageType;
    typedef typename ProbabilisticVectorImageType::Pointer ProbabilisticVectorImagePointerType;
    typedef typename itk::ImageRegionIterator<ProbabilisticVectorImageType> ProbImageIteratorType;


    typedef std::vector<std::pair<string,ImagePointerType> > ImageListType;

    enum MetricType {NONE,MAD,NCC,MI,NMI,MSD};
    enum WeightingType {UNIFORM,GLOBAL,LOCAL};
protected:
    double m_sigma;
    RadiusType m_patchRadius;
public:
    int run(int argc, char ** argv){
        feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
        ArgumentParser * as=new ArgumentParser(argc,argv);
        string deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir=".",outputSuffix="",weightListFilename="", imageFileListAtlas="";
        int verbose=0;
        double pWeight=1.0;
        double radius=3;
        int maxHops=1;
        bool uniformUpdate=true;
        string metricName="NCC";
        string weightingName="uniform";
        bool lateFusion=false;
        bool dontCacheDeformations=false;
        bool graphCut=false;
        double smoothness=1.0;
        double m_graphCutSigma=10;
        int useNAtlases=1000000;
        int useNTargets=1000000;
        double globalOneHopWeight=1.0;
        bool AREG= false;
        m_sigma=30;
        as->parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", true);
        as->parameter ("T", deformationFileList, " list of deformations", true);
        as->parameter ("i", imageFileList, " list of target images", true);
        as->parameter ("iAtlas", imageFileListAtlas, " list of atlas images (if not set, target image filelist is assumed to contain both atlas and target images)", false);
        as->parameter ("W", weightListFilename,"list of weights for deformations",false);
        as->parameter ("metric", metricName,"metric to be used for global or local weighting, valid: NONE,SAD,MSD,NCC,MI,NMI",false);
        as->parameter ("weighting", weightingName,"internal weighting scheme {uniform,local,global}. non-uniform will only work with metric != NONE",false);
        as->parameter ("s", m_sigma,"sigma for exp(- metric/sigma)",false);
        as->parameter ("sigmaGC", m_graphCutSigma,"sigma for exp(- contrast/sigma) for graphcut smoothness",false);
        as->parameter ("radius", radius,"patch radius for local metrics",false);
        as->parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
        as->parameter ("radius", radius,"patch radius for NCC",false);
        as->parameter ("maxHops", maxHops,"maximum number of hops",false);
        as->parameter ("useNAtlases", useNAtlases,"use the first N atlases from the list",false);
        as->parameter ("useNTargets", useNTargets,"use the first N targets as intermediate images",false);
        as->parameter ("globalOneHopWeight", globalOneHopWeight,"global weight for one hop segmentations (vs. zero hop)",false);
        as->option ("AREG", AREG,"use AREG to select intermediate targets");
        as->option ("lateFusion", lateFusion,"fuse segmentations late. maxHops=1");
        as->option ("dontCacheDeformations", dontCacheDeformations,"read deformations only when needed to save memory. higher IO load!");
        as->option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
        as->parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
        as->parameter ("verbose", verbose,"get verbose output",false);
        as->help();
        as->parse();
        string suffix;
        if (D==2)
            suffix=".png";
        else
            suffix=".nii.gz";

     

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

      
        
        ImageListType *targetImages,*inputAtlasSegmentations, *atlasImages;
        map<string,int> *targetIDMap,*atlasSegmentationIDMap, *atlasIDMap;
        targetIDMap=new   map<string,int> ;
        atlasSegmentationIDMap=new   map<string,int> ;
        atlasIDMap=new   map<string,int> ;

        typedef typename ImageListType::iterator ImageListIteratorType;
        LOG<<"Reading atlas segmentations."<<endl;
        inputAtlasSegmentations = readImageList( atlasSegmentationFileList, atlasSegmentationIDMap );
        
        LOG<<VAR(atlasSegmentationIDMap)<<endl;

        SegmentationMapper<ImageType> segmentationMapper;
        //map segmentation labels to continuous discrete range
        for (ImageListIteratorType it=inputAtlasSegmentations->begin();it!=inputAtlasSegmentations->end();++it){
            it->second=segmentationMapper.ApplyMap(it->second);
        }
        


        int nAtlases = inputAtlasSegmentations->size();
        useNAtlases=min(useNAtlases,nAtlases);

        LOG<<"Reading input images."<<endl;
        targetImages = readImageList( imageFileList, targetIDMap );
        int nImages = targetImages->size();
        useNTargets=min(nImages,useNTargets);
        if (imageFileListAtlas != ""){
            atlasImages=readImageList(imageFileListAtlas, atlasIDMap);
        }else{
            for (ImageListIteratorType targetImageIterator=targetImages->begin();targetImageIterator!=targetImages->end();++targetImageIterator){
                string targetID = targetImageIterator->first;
                //add target image to atlas images if it is in the list of atlas segmentations
                if ( atlasSegmentationIDMap->find(targetID)!=atlasSegmentationIDMap->end()){
                    atlasImages->push_back(make_pair(targetID,targetImageIterator->second));
                }
            }
        }
        
        LOGV(2)<<VAR(metric)<<" "<<VAR(weighting)<<endl;
        LOGV(2)<<VAR(m_sigma)<<" "<<VAR(lateFusion)<<" "<<VAR(m_patchRadius)<<endl;

        if (dontCacheDeformations){
            LOG<<"Reading deformation file names."<<endl;
        }else{
            LOG<<"CACHING all deformations!"<<endl;
        }
        map< string, map <string, DeformationFieldPointerType> > deformationCache;
        map< string, map <string, string> > deformationFilenames;
        map<string, map<string, float> > globalWeights;
        {
            ifstream ifs(deformationFileList.c_str());
	    LOGV(3)<<"Reading deformation filenames from " << deformationFileList << endl;
            while (!ifs.eof()){
                string intermediateID,targetID,defFileName;
                ifs >> intermediateID;
		LOGV(3)<<VAR(intermediateID)<<endl;
                if (intermediateID!=""){
                    ifs >> targetID;
                    ifs >> defFileName;
                    //skip inter-atlas deformations
                    //if ( (    targetImages->find(intermediateID)==targetImages->end() && atlasImages->find(intermediateID)==targetImages->end()) || targetImages->find(targetID)==targetImages->end()  ){
                    if  (    false && atlasSegmentationIDMap->find(targetID)!=atlasSegmentationIDMap->end() ){
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
        
        if (weightListFilename!=""){
            ifstream ifs(weightListFilename.c_str());
            while (!ifs.eof()){
                string intermediateID,targetID;
                ifs >> intermediateID;
                ifs >> targetID;
                if (false)
                    LOG << intermediateID<<" or "<<targetID<<" not in image database while reading weights, skipping"<<endl;
                else{
                    ifs >> globalWeights[intermediateID][targetID];
                }
            }
        }
    
        logSetStage("Zero Hop");
        LOG<<"Computing"<<std::endl;
        map<string,ProbabilisticVectorImagePointerType> probabilisticSegmentations;
        
        //generate atlas probabilistic segmentations from atlas segmentations
        for (ImageListIteratorType atlasImageIterator=inputAtlasSegmentations->begin();atlasImageIterator!=inputAtlasSegmentations->end();++atlasImageIterator){ 
            //iterate over atlass
            string atlasID = atlasImageIterator->first;
            LOGV(3)<<VAR(atlasID)<<endl;
            probabilisticSegmentations[atlasID]=segmentationToProbabilisticVector(atlasImageIterator->second);
        }


#if 1
        if (AREG){
            LOG<<"Resorting intermediate targets based on ARE-G"<<endl;
            //re-sort target images to yield improving atlas reconstruction error
            double areG=1000000.0;
            int i=0;
            map<string,ProbabilisticVectorImagePointerType> probabilisticAtlasSelfSegmentations;
            int atlasN = 0;
            for (ImageListIteratorType atlasIterator=inputAtlasSegmentations->begin();atlasIterator!=inputAtlasSegmentations->end() && (atlasN<useNAtlases);++atlasIterator,++atlasN){//iterate over atlases
                probabilisticAtlasSelfSegmentations[atlasIterator->first]=  createEmptyProbImageFromImage(atlasIterator->second);
            }

            for (ImageListIteratorType targetImageIterator=targetImages->begin();targetImageIterator!=targetImages->end() && i<useNTargets;++targetImageIterator,++i){
                LOGV(2)<<"Processing "<<i<<"the intermediate image"<<endl;
                double bestARE=100000.0;
                string bestID;
                int bestIdx;
                int j=i;
                for (ImageListIteratorType targetImageIterator2=targetImageIterator;targetImageIterator2!=targetImages->end();++targetImageIterator2,++j){
                    string targetID = targetImageIterator2->first;
                    LOGV(2)<<"Probing image "<<targetID<<endl;
                    double localARE=0.0;
                    atlasN=0;
                    for (ImageListIteratorType atlasIterator=inputAtlasSegmentations->begin();atlasIterator!=inputAtlasSegmentations->end() && (atlasN<useNAtlases);++atlasIterator,++atlasN){//iterate over atlases
                        string atlasID = atlasIterator->first;
                        if (atlasID == targetID){
                            LOGV(2)<<"Target equals atlas, skipping.. " <<endl;
                            localARE=1.0;
                            continue;
                        }else{
                        ProbabilisticVectorImagePointerType probAtlasSeg=ImageUtils<ProbabilisticVectorImageType>::duplicate(probabilisticAtlasSelfSegmentations[atlasID]);
                        //todo: propagate atlas segmentation forward backward
                        //todo accumulate atlas segmentations
                        DeformationFieldPointerType firstDeformation,secondDeformation,deformation;
                        if (dontCacheDeformations){
                            firstDeformation = ImageUtils<DeformationFieldType>::readImage(deformationFilenames[atlasID][targetID]);
                            secondDeformation = ImageUtils<DeformationFieldType>::readImage(deformationFilenames[targetID][atlasID]);
                        }else{
                            firstDeformation = deformationCache[atlasID][targetID];
                            secondDeformation = deformationCache[targetID][atlasID];
                        }
                        
                        deformation = TransfUtilsType::composeDeformations(secondDeformation,firstDeformation);
                        ProbabilisticVectorImagePointerType probSeg=probabilisticSegmentations[atlasID];
                        double weight = 1.0;
                        updateProbabilisticSegmentationLocalMetricNew(probAtlasSeg,probSeg,weight,targetImageIterator->second,(*atlasImages)[(*atlasIDMap)[atlasID]].second,deformation,metric);
                        ImagePointerType outputImage=probSegmentationToSegmentationLocal(probAtlasSeg);
                        double error=1.0-DICE(outputImage,atlasIterator->second);
                        LOGV(3)<<"Reconstructing "<<atlasID<<" using "<<targetID<<"; dice error: "<<error<<endl;
                        localARE+=error;
                        }
                    }
                    
                    //check if ARE improved
                    if (localARE<bestARE){
                        bestID=targetImageIterator2->first;
                        bestIdx=j;
                        bestARE=localARE;
                    }
                }
                LOGV(1)<<"Found best: "<<VAR(bestARE)<<" "<< VAR(bestID)<< endl;
                LOGV(2)<<"Swapping.. "<<endl;
                //swap in best sample at position
                string oldID=targetImageIterator->first;
                int oldIdx=0;
                ImagePointerType tmp=targetImageIterator->second;
                (*targetImages)[i]=(*targetImages)[bestIdx];
                (*targetImages)[bestIdx].first=oldID;
                (*targetImages)[bestIdx].second=tmp;
                //update accumulated atlas probabilities
                int atlasN=0;
                for (ImageListIteratorType atlasIterator=inputAtlasSegmentations->begin();atlasIterator!=inputAtlasSegmentations->end() && (atlasN<useNAtlases);++atlasIterator,++atlasN){//iterate over atlases
                    string atlasID = atlasIterator->first;
                    string targetID = bestID;
                    //todo: propagate atlas segmentation forward backward
                    //todo accumulate atlas segmentations
                    DeformationFieldPointerType firstDeformation,secondDeformation,deformation;
                    if (dontCacheDeformations){
                        firstDeformation = ImageUtils<DeformationFieldType>::readImage(deformationFilenames[atlasID][targetID]);
                        secondDeformation = ImageUtils<DeformationFieldType>::readImage(deformationFilenames[targetID][atlasID]);
                    }else{
                        firstDeformation = deformationCache[atlasID][targetID];
                        secondDeformation = deformationCache[targetID][atlasID];
                    }
                        
                    deformation = TransfUtilsType::composeDeformations(secondDeformation,firstDeformation);
                    ProbabilisticVectorImagePointerType probSeg=probabilisticSegmentations[atlasID];
                    double weight = 1.0;
                    updateProbabilisticSegmentationLocalMetricNew(probabilisticAtlasSelfSegmentations[atlasID],probSeg,weight,tmp,(*atlasImages)[(*atlasIDMap)[atlasID]].second,deformation,metric);
                }
                
            }
        }
#endif


        //generate zero-hop target segmentations
        for (ImageListIteratorType targetImageIterator=targetImages->begin();targetImageIterator!=targetImages->end();++targetImageIterator){                //iterate over targets
            string targetID= targetImageIterator->first;
            if (atlasSegmentationIDMap->find(targetID)==atlasSegmentationIDMap->end()){ //do not calculate segmentation for atlas images
                probabilisticSegmentations[targetID]=createEmptyProbImageFromImage( targetImageIterator->second);
                int atlasN=0;
                for (ImageListIteratorType atlasIterator=inputAtlasSegmentations->begin();
                     atlasIterator!=inputAtlasSegmentations->end() && atlasN<useNAtlases;
                     ++atlasIterator,++atlasN)
                    {//iterate over atlases
                        string atlasID=atlasIterator->first;
                        LOGV(4)<<VAR(atlasID)<<" "<<VAR(targetID)<<endl;
                        DeformationFieldPointerType deformation;
                        if (dontCacheDeformations){
                            deformation = ImageUtils<DeformationFieldType>::readImage(deformationFilenames[atlasID][targetID]);
                        }else{
                            deformation = deformationCache[atlasID][targetID];
                        }
                        ImagePointerType atlasSegmentation=atlasIterator->second;
                        ProbabilisticVectorImagePointerType probAtlasSegmentation=segmentationToProbabilisticVector(atlasSegmentation);
                        double weight=globalWeights[atlasID][targetID];
                   
                        //update
                        if (weighting==UNIFORM || metric == NONE || (lateFusion && nAtlases==1)){
                            updateProbabilisticSegmentationUniform(probabilisticSegmentations[targetID],probAtlasSegmentation,weight,deformation);
                        }else{
                            ImagePointerType targetImage= targetImageIterator->second;
                            ImagePointerType atlasImage=(*atlasImages)[(*atlasIDMap)[atlasID]].second;
                            if (weighting==GLOBAL){
                                updateProbabilisticSegmentationGlobalMetric(probabilisticSegmentations[targetID],probAtlasSegmentation,weight,targetImage,atlasImage,deformation,metric);
                            }else if (weighting==LOCAL){
                                updateProbabilisticSegmentationLocalMetricNew(probabilisticSegmentations[targetID],probAtlasSegmentation,weight,targetImage,atlasImage,deformation,metric);
                            }

                        }
                    }
            }
           
        }//finished zero-hop segmentation
        LOG<<"done"<<endl;

        LOGV(1)<<"Storing zero-hop segmentations."<<endl;
        for (ImageListIteratorType targetImageIterator=targetImages->begin();targetImageIterator!=targetImages->end();++targetImageIterator){
            string targetID= targetImageIterator->first;
            if (atlasSegmentationIDMap->find(targetID)==atlasSegmentationIDMap->end()){ 
                ImagePointerType outputImage;
                if (graphCut)
                    outputImage=probSegmentationToSegmentationGraphcutMultiLabel(probabilisticSegmentations[targetID],targetImageIterator->second,smoothness*inputAtlasSegmentations->size(),m_graphCutSigma);
                else
                    outputImage=probSegmentationToSegmentationLocal(probabilisticSegmentations[targetID]);
                ostringstream tmpSegmentationFilename;
                tmpSegmentationFilename<<outputDir<<"/segmentation-weighting"<<weightingName<<"-metric"<<metricName<<"-target"<<targetID<<"-hop0"<<suffix;
                ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),segmentationMapper.MapInverse(outputImage));
                ostringstream tmpSegmentationFilename2;
                tmpSegmentationFilename2<<outputDir<<"/segmentation-weighting"<<weightingName<<"-metric"<<metricName<<"-target"<<targetID<<"-hop0-ProbImage.mha";
                LOGI(4,ImageUtils<ProbabilisticVectorImageType>::writeImage(tmpSegmentationFilename2.str().c_str(),probabilisticSegmentations[targetID]));
                
            }
        }



        bool converged=false; //no convergence check, yet
        logSetStage("Iterative improvement");
        for (int n=1;n<=maxHops && !converged;++n){
            LOG<<"hop "<<n<<endl;
            //initialize new prob segmentations
            map<string,ProbabilisticVectorImagePointerType> newProbabilisticTargetSegmentations;
            //iterate over atlases
            for (ImageListIteratorType atlasImageIterator=atlasImages->begin();atlasImageIterator!=atlasImages->end();++atlasImageIterator){
                string atlasID = atlasImageIterator->first;
                newProbabilisticTargetSegmentations[atlasID] = probabilisticSegmentations[atlasID];
            }
            //targets
            for (ImageListIteratorType targetImageIterator=targetImages->begin();targetImageIterator!=targetImages->end();++targetImageIterator){
                string targetID = targetImageIterator->first;
                if ( atlasSegmentationIDMap->find(targetID)==atlasSegmentationIDMap->end()){
                    //initialize with previous hop likelihood
                    newProbabilisticTargetSegmentations[targetID]=ImageUtils<ProbabilisticVectorImageType>::duplicate(probabilisticSegmentations[targetID]);
                }
            }
            
            //update!
            int atlasN=0;
            for (ImageListIteratorType atlasIterator=inputAtlasSegmentations->begin();atlasIterator!=inputAtlasSegmentations->end() && (atlasN<useNAtlases);++atlasIterator,++atlasN){//iterate over atlases
                string atlasID=atlasIterator->first;
                int intermediateN=0;
                for (ImageListIteratorType intermediateImageIterator=targetImages->begin();intermediateImageIterator!=targetImages->end() && (intermediateN<useNTargets);++intermediateImageIterator){//iterate over intermediate images
                    string intermediateID= intermediateImageIterator->first;
                    if ( (intermediateID==atlasID) || (atlasSegmentationIDMap->find(intermediateID) == atlasSegmentationIDMap->end()) ){
                        ++intermediateN;
                        //for late fusion, we do also need the deformation from the atlas to the intermediate image
                        DeformationFieldPointerType firstDeformation;
                        if (lateFusion){ 
                            if (intermediateID != atlasID){
                                if (dontCacheDeformations){
                                    LOGV(3)<<VAR(atlasID)<<" "<<VAR(intermediateID)<<" "<<VAR(deformationFilenames[atlasID][intermediateID])<<endl;
                                    firstDeformation = ImageUtils<DeformationFieldType>::readImage(deformationFilenames[atlasID][intermediateID]);
                                }else{
                                    LOGV(3)<<VAR(atlasID)<<" "<<VAR(intermediateID)<<endl;
                                    firstDeformation = deformationCache[atlasID][intermediateID];
                                }
                            }
                        }

                        for (ImageListIteratorType targetImageIterator=targetImages->begin();targetImageIterator!=targetImages->end();++targetImageIterator){//iterate over target images
                            string targetID= targetImageIterator->first;             
                            if ( targetID != atlasID){ //don't update atlas segmentations! :)
                                if (targetID != intermediateID ){ // don't propagate segmentations to the intermediate image.. doesn't make sense, right?
                                    LOGV(3)<<"hop "<<n<<" "<<  VAR(atlasID) << " "<< VAR(intermediateID) <<" "<<VAR(targetID) << endl;


                                    //get auxiliary global weight
                                    double weight=globalWeights[intermediateID][targetID];

                                    //get deformation from intermediate to target image, cached or uncached
                                    DeformationFieldPointerType secondDeformation,deformation;
                                    if (dontCacheDeformations){
                                        LOGV(3)<<VAR(targetID)<<" "<<VAR(intermediateID)<<" "<<VAR(deformationFilenames[intermediateID][targetID])<<endl<<endl;
                                        secondDeformation = ImageUtils<DeformationFieldType>::readImage(deformationFilenames[intermediateID][targetID]);
                                    }else{
                                        LOGV(3)<<VAR(targetID)<<" "<<VAR(intermediateID)<<endl<<endl;
                                        secondDeformation = deformationCache[intermediateID][targetID];
                                    }

                                    ProbabilisticVectorImagePointerType probSeg;
                                    if (lateFusion) {
                                        if ( intermediateID == atlasID ){
                                            deformation = secondDeformation;
                                        }else{
                                            deformation = TransfUtilsType::composeDeformations(secondDeformation,firstDeformation);
                                            weight*=globalWeights[atlasID][intermediateID];
                                        }
                                        probSeg = probabilisticSegmentations[atlasID];
                                    }else{
                                        deformation = secondDeformation;
                                        probSeg = probabilisticSegmentations[intermediateID];
                                    }
                          
                                    //weight*=globalOneHopWeight*1.0/min(useNAtlases,nAtlases);
                                    weight = 1.0/min(useNAtlases,nAtlases);

                                    //UPDATE
                                    if (weighting==UNIFORM || metric == NONE){
                                        updateProbabilisticSegmentationUniform(newProbabilisticTargetSegmentations[targetID],probSeg,weight,deformation);
                                    }else{
                                        ImagePointerType img1=targetImageIterator->second;
                                        ImagePointerType img2;
                                        if (lateFusion){
                                            img2=(*atlasImages)[(*atlasIDMap)[atlasID]].second;
                                        }else{
                                            img2=intermediateImageIterator->second;
                                        }
                                        if (weighting==GLOBAL){
                                            updateProbabilisticSegmentationGlobalMetric(newProbabilisticTargetSegmentations[targetID],probSeg,weight,img1,img2,deformation,metric);
                                    
                                        }else if (weighting==LOCAL){
                                            updateProbabilisticSegmentationLocalMetricNew(newProbabilisticTargetSegmentations[targetID],probSeg,weight,img1,img2,deformation,metric);
                                        }
                                    }
                                    if (verbose>=8){
                                        ImagePointerType outputIntermediateSegmentation = probSegmentationToSegmentationLocal(warpProbImage(probSeg,deformation));
                                        ostringstream tmpSegmentationFilename;
                                        tmpSegmentationFilename<<outputDir<<"/segmentation-intermediate-weighting"<<weightingName<<"-metric"<<metricName<<"-from-"<<atlasID<<"-over-"<<intermediateID<<"-to-"<<targetID<<"-hop"<<n<<suffix;
                                        ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),segmentationMapper.MapInverse(outputIntermediateSegmentation));
                                    }
                                }
                            }
                        }//for targetID
                    }//catch paths over intermediate atlases
                }//for intermediateID

                //iterating over atlases only makes sense for late fusion
                //in the early fusion case, multiple atlases are already incorporated in the zero hop segmentation
                if (!lateFusion)
                    break;
                           
            }//for atlases

            LOG<<"done"<<endl;
            LOG<<"Storing output. and checking convergence"<<endl;
            for (ImageListIteratorType targetImageIterator=targetImages->begin();targetImageIterator!=targetImages->end();++targetImageIterator){
                string id= targetImageIterator->first;
                LOGV(4)<<VAR(id)<<endl;
                if (atlasSegmentationIDMap->find(id)==atlasSegmentationIDMap->end()){ 
                    ImagePointerType outputImage;
                    
                    //ProbabilisticVectorImagePointerType normalizedProbs=normalizeProbs(newProbabilisticTargetSegmentations[id]);
                    ProbabilisticVectorImagePointerType normalizedProbs=normalizeProbsSimple(newProbabilisticTargetSegmentations[id],useNAtlases, useNTargets);
                    //ProbabilisticVectorImagePointerType normalizedProbs=(newProbabilisticTargetSegmentations[id]);

                    if (graphCut)
                        outputImage=probSegmentationToSegmentationGraphcut(newProbabilisticTargetSegmentations[id],smoothness*(nImages-nAtlases)*nAtlases);
                    else
                        outputImage=probSegmentationToSegmentationLocal(newProbabilisticTargetSegmentations[id]);
                    ostringstream tmpSegmentationFilename;
                    tmpSegmentationFilename<<outputDir<<"/segmentation-weighting"<<weightingName<<"-metric"<<metricName<<"-target"<<id<<"-hop"<<n<<suffix;
                    ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),segmentationMapper.MapInverse(outputImage));
                    ostringstream tmpSegmentationFilename2;
                    tmpSegmentationFilename2<<outputDir<<"/segmentation-weighting"<<weightingName<<"-metric"<<metricName<<"-target"<<id<<"-hop1-ProbImage.mha";
                    LOGI(4,ImageUtils<ProbabilisticVectorImageType>::writeImage(tmpSegmentationFilename2.str().c_str(),normalizedProbs));
                }
            }
            probabilisticSegmentations=newProbabilisticTargetSegmentations;
        }// hops
        return 1;
    }//run
protected:
    ImageListType * readImageList(string filename, map<string,int> * indexMap){
        ImageListType * result=new  ImageListType;
        ifstream ifs(filename.c_str());
        if (!ifs){
            LOG<<"could not read "<<filename<<endl;
            exit(0);
        }
        int c=0;
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
                    if (indexMap->find(imageID)==indexMap->end()){
                        (*indexMap)[imageID]=c;
                        result->push_back(make_pair(imageID,img));
                        ++c;
                    }
                    else{
                        LOG<<"duplicate image ID "<<imageID<<", aborting"<<endl;
                        exit(0);
                    }
                }
            }
        return result;
    }        
    ProbabilisticVectorImagePointerType segmentationToProbabilisticVector(ImagePointerType img){
        ProbabilisticVectorImagePointerType result=createEmptyProbImageFromImage(img);
        ProbImageIteratorType probIt(result,result->GetLargestPossibleRegion());
        ImageIteratorType imgIt(img,img->GetLargestPossibleRegion());
        for (imgIt.GoToBegin(),probIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++probIt){
            ProbabilisticPixelType p;
            p.Fill(0.0);
            p[int(imgIt.Get())]=1;
            probIt.Set(p);
        }
        return result;
    }

    ImagePointerType probSegmentationToSegmentationLocal( ProbabilisticVectorImagePointerType img){
        ImagePointerType result=ImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        ImageIteratorType imgIt(result,result->GetLargestPossibleRegion());
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        for (imgIt.GoToBegin(),probIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++probIt){
            float maxProb=-std::numeric_limits<float>::max();
            int maxLabel=0;
            ProbabilisticPixelType p = probIt.Get();
            for (unsigned int s=0;s<nSegmentationLabels;++s){
                if (p[s]>maxProb){
                    maxLabel=s;
                    maxProb=p[s];
                }
            }
            if (D==2){
                imgIt.Set(1.0*std::numeric_limits<PixelType>::max()*maxLabel/(nSegmentationLabels-1));
            }else{
                imgIt.Set(maxLabel);
            }
        }
        return result;
    }
    ImagePointerType probSegmentationToProbImageLocal( ProbabilisticVectorImagePointerType img){
        ImagePointerType result=ImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        ImageIteratorType imgIt(result,result->GetLargestPossibleRegion());
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        for (imgIt.GoToBegin(),probIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++probIt){
            float maxProb=-std::numeric_limits<float>::max();
            int maxLabel=0;
            ProbabilisticPixelType p = probIt.Get();
            double sump=0.0;
            for (unsigned int s=0;s<nSegmentationLabels;++s){
                if (p[s]>maxProb){
                    maxLabel=s;
                    maxProb=p[s];
                }
                sump+=p[s];
            }
            maxProb=p[1]/sump;
            if (D==2){
                imgIt.Set(1.0*std::numeric_limits<PixelType>::max()*pow(maxProb,1.0));
            }else{
                imgIt.Set(maxProb*maxLabel);
            }
        }
        return result;
    }

    ImagePointerType probSegmentationToSegmentationGraphcut( ProbabilisticVectorImagePointerType img, double smooth){
        ImagePointerType result=ImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        typedef Graph<float,float,double> MRFType;
        typedef MRFType::node_id NodeType;
        MRFType* optimizer;
        SizeType size=img->GetLargestPossibleRegion().GetSize();
        int nNodes=1;
        for (unsigned int d=0;d<D;++d) {nNodes*=size[d];}
        int nEdges=D*nNodes;
        for (unsigned int d=0;d<D;++d) {nEdges-=size[d];}
        optimizer = new MRFType(nNodes,nEdges);
        optimizer->add_node(nNodes);
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        int i=0;
        for (probIt.GoToBegin();!probIt.IsAtEnd();++probIt,++i){
            IndexType idx=probIt.GetIndex();
            ProbabilisticPixelType localProbs=probIt.Get();
            ProbabilisticPixelType energies;
            for (unsigned int s=0;s<nSegmentationLabels;++s){
                energies[s]=0;
                for (unsigned int sprime=0;sprime<nSegmentationLabels;++sprime){
                    if (sprime!=s){
                        energies[s]+=localProbs[sprime];
                    }
                }
            }
            LOGV(7)<<VAR(i)<<" "<<VAR(energies)<<endl;
            optimizer->add_tweights(i,energies[0],energies[1]);
            for (unsigned  int d=0;d<D;++d){
                OffsetType off;
                off.Fill(0);
                off[d]+=1;
                IndexType neighborIndex=idx+off;
                bool inside2;
                int withinImageIndex2=ImageUtils<ImageType>::ImageIndexToLinearIndex(neighborIndex,size,inside2);
                if (inside2){
                    optimizer -> add_edge(i,withinImageIndex2,smooth,smooth);
                }
            }
        }
        optimizer->maxflow();
        ImageIteratorType imgIt(result,result->GetLargestPossibleRegion());
        i=0;
        for (imgIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++i){
            int maxLabel=optimizer->what_segment(i)== MRFType::SOURCE ;
            if (D==2){
                imgIt.Set(1.0*std::numeric_limits<PixelType>::max()*maxLabel/(nSegmentationLabels-1));
            }else{
                imgIt.Set(maxLabel);
            }
        }
        return result;
    }
    ProbabilisticVectorImagePointerType normalizeProbs(ProbabilisticVectorImagePointerType img){
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        ProbabilisticVectorImagePointerType result=ProbabilisticVectorImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        ProbImageIteratorType resultIt(result,img->GetLargestPossibleRegion());
        resultIt.GoToBegin();
        for (probIt.GoToBegin();!probIt.IsAtEnd();++probIt,++resultIt){
            ProbabilisticPixelType localProbs=probIt.Get();
            double sum=0.0;
            //compute normalizing sum to normalize probabilities(if they're not yet normalized)
            for (int s2=0;s2<nSegmentationLabels;++s2){
                sum+=localProbs[s2];
            }
            for (int s2=0;s2<nSegmentationLabels;++s2){
                localProbs[s2]/=sum;
            }
            resultIt.Set(localProbs);
        }
        return result;
    }
    ProbabilisticVectorImagePointerType normalizeProbsSimple(ProbabilisticVectorImagePointerType img, int nAtlases, int nImages){
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        ProbabilisticVectorImagePointerType result=ProbabilisticVectorImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        ProbImageIteratorType resultIt(result,img->GetLargestPossibleRegion());
        resultIt.GoToBegin();
        for (probIt.GoToBegin();!probIt.IsAtEnd();++probIt,++resultIt){
            ProbabilisticPixelType localProbs=probIt.Get();
          
            for (int s2=0;s2<nSegmentationLabels;++s2){
                localProbs[s2]/=nAtlases*(nImages-1);
            }
            resultIt.Set(localProbs);
        }
        return result;
    }
    ImagePointerType probSegmentationToSegmentationGraphcutMultiLabel( ProbabilisticVectorImagePointerType img, ImagePointerType segImg, double smooth, double sigma){
        ImagePointerType result=ImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();

        typedef GCoptimizationGeneralGraph MRFType;
        //todo
        MRFType optimizer(result->GetBufferedRegion().GetNumberOfPixels(),nSegmentationLabels);
       
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
       
        //going to use sparse labels
        //iterate over labels
        for (unsigned int s=0;s<nSegmentationLabels;++s){
            std::vector<GCoptimization::SparseDataCost> costs(result->GetBufferedRegion().GetNumberOfPixels());
            //GCoptimization::SparseDataCost costs[result->GetBufferedRegion().GetNumberOfPixels()];
            int n = 0;
            int i=0;
            for (probIt.GoToBegin();!probIt.IsAtEnd();++probIt,++i){
                ProbabilisticPixelType localProbs=probIt.Get();
                double prob=localProbs[s];
                double sum=0.0;
                //compute normalizing sum to normalize probabilities(if they're not yet normalized)
                for (int s2=0;s2<nSegmentationLabels;++s2){
                    sum+=localProbs[s2];
                }
                //only add site do label if prob > 0
                if (prob>0){
                    costs[n].site=i;
                    //LOGV(3)<<VAR(prob)<<" "<<VAR(sum)<<endl;
                    costs[n].cost=-log(prob/sum);
                    ++n;
                }
            }
            //resize to actual number of sites with label s
            costs.resize(n);
            optimizer.setDataCost(s,&costs[0],n);
        }
        float *smoothCosts = new float[nSegmentationLabels*nSegmentationLabels];
        for ( int l1 = 0; l1 < nSegmentationLabels; l1++ )
            for (int l2 = 0; l2 < nSegmentationLabels; l2++ )
                smoothCosts[l1+l2*nSegmentationLabels] =  (l1!=l2);
        optimizer.setSmoothCost(smoothCosts);
        int i=0;
        SizeType size=img->GetLargestPossibleRegion().GetSize();

        ImageIteratorType imageIt(segImg,segImg->GetLargestPossibleRegion());
        for (imageIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++i){
            IndexType idx=imageIt.GetIndex();
            double value=imageIt.Get();
            for (unsigned  int d=0;d<D;++d){
                OffsetType off;
                off.Fill(0);
                off[d]+=1;
                IndexType neighborIndex=idx+off;
                bool inside2;
                int withinImageIndex2=ImageUtils<ImageType>::ImageIndexToLinearIndex(neighborIndex,size,inside2);
                if (inside2){
                    double neighborvalue=segImg->GetPixel(neighborIndex);
                    double weight= exp(-fabs(value-neighborvalue)/sigma);
                    optimizer.setNeighbors(i,withinImageIndex2,smooth*weight);
                }
            }
        }
        LOGV(1)<<"solving graph cut"<<endl;
        optimizer.setVerbosity(mylog.getVerbosity());
        try{
            optimizer.expansion(20);
        }catch (GCException e){
            e.Report();
            exit(-1);
        }
        LOGV(1)<<"done"<<endl;
        ImageIteratorType imgIt(result,result->GetLargestPossibleRegion());
        i=0;
        for (imgIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++i){
            int maxLabel=optimizer.whatLabel(i) ;
            if (D==2){
                imgIt.Set(1.0*std::numeric_limits<PixelType>::max()*maxLabel/(nSegmentationLabels-1));
            }else{
                imgIt.Set(maxLabel);
            }
        }
        delete smoothCosts;
        return result;
    }

    void updateProbabilisticSegmentationUniform(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight,DeformationFieldPointerType deformation){
        ProbabilisticVectorImagePointerType deformedIncrement=warpProbImage(increment,deformation);
        ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());
        for (accIt.GoToBegin(),incIt.GoToBegin();!incIt.IsAtEnd();++incIt,++accIt){
            accIt.Set(accIt.Get()+incIt.Get()*globalWeight);
        }
    }

    void updateProbabilisticSegmentationGlobalMetric(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType targetImage, ImagePointerType movingImage,DeformationFieldPointerType deformation,MetricType metric ){
       
        double metricWeight=0;
        typedef typename itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typename InterpolatorType::Pointer interpolator=InterpolatorType::New();
        interpolator->SetInputImage(movingImage);

        typedef typename itk::DisplacementFieldTransform<typename DisplacementType::ComponentType ,D> DTTransformType;
        typename DTTransformType::Pointer transf=DTTransformType::New();
        transf->SetDisplacementField(deformation);

        switch(metric){
        case NCC:{
            typedef typename itk::NormalizedCorrelationImageToImageMetric<
                ImageType,
                ImageType >    NCCMetricType;
            typename NCCMetricType::Pointer ncc=NCCMetricType::New();
            ncc->SetTransform(transf);
            ncc->SetFixedImage(targetImage);
            ncc->SetMovingImage(movingImage);
            ncc->SetInterpolator(interpolator);
            ncc->SetFixedImageRegion( targetImage->GetLargestPossibleRegion() );
            metricWeight=(1.0-ncc->GetValue(transf->GetParameters()))/2;
            break;
        }
        case MSD:{
            typedef typename itk::MeanSquaresImageToImageMetric<
                ImageType,
                ImageType >    MSDMetricType;
            typename MSDMetricType::Pointer msd=MSDMetricType::New();
            msd->SetFixedImageRegion( targetImage->GetLargestPossibleRegion() );
            msd->SetTransform(transf);
            msd->SetFixedImage(targetImage);
            msd->SetMovingImage(movingImage);
            msd->SetInterpolator(interpolator);
            msd->Initialize();
            metricWeight=(msd->GetValue(transf->GetParameters()));
            metricWeight/=m_sigma*m_sigma;
            metricWeight=exp(-metricWeight);
            break;
        }
        case MAD:{
            metricWeight=globalMAD(targetImage,movingImage,deformation);
            break;
        }
        case MI:{
            typedef itk::MattesMutualInformationImageToImageMetric< 
                ImageType, 
                ImageType >    MIMetricType;
            typename MIMetricType::Pointer       MImetric      = MIMetricType::New();
            MImetric->ReinitializeSeed( 76926294 );
            MImetric->SetNumberOfHistogramBins( 50 );
            MImetric->SetNumberOfSpatialSamples( 100*50 );
            MImetric->SetFixedImageRegion( targetImage->GetLargestPossibleRegion() );
            MImetric->SetTransform(transf);
            MImetric->SetFixedImage(targetImage);
            MImetric->SetMovingImage(movingImage);
            MImetric->SetInterpolator(interpolator);
            MImetric->Initialize();
            metricWeight=-MImetric->GetValue(transf->GetParameters());
          
            break;
        }
        case NMI:{
            typedef itk::NormalizedMutualInformationHistogramImageToImageMetric< 
                ImageType, 
                ImageType >    NMIMetricType;
            typename NMIMetricType::Pointer      NMImetric     = NMIMetricType::New();
            
            typename NMIMetricType::HistogramType::SizeType histSize;
            histSize[0] = 50;
            histSize[1] = 50;
            NMImetric->SetHistogramSize(histSize);           
            NMImetric->SetFixedImageRegion( targetImage->GetLargestPossibleRegion() );
            NMImetric->SetTransform(transf);
            NMImetric->SetFixedImage(targetImage);
            NMImetric->SetMovingImage(movingImage);
            NMImetric->SetInterpolator(interpolator);
            NMImetric->Initialize();
            metricWeight=NMImetric->GetValue(transf->GetParameters());
            break;
        }
        }   //switch

        ProbabilisticVectorImagePointerType deformedIncrement=warpProbImage(increment,deformation);
        ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());
     
        LOGV(10)<<VAR(metricWeight)<<endl;
        for (accIt.GoToBegin(),incIt.GoToBegin();!incIt.IsAtEnd();++incIt,++accIt){
            accIt.Set(accIt.Get()+incIt.Get()*globalWeight*metricWeight);
        }
    }
    void updateProbabilisticSegmentationLocalMetric(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType targetImage, ImagePointerType movingImage,DeformationFieldPointerType deformation,MetricType metric ){
        ProbabilisticVectorImagePointerType deformedIncrement=warpProbImage(increment,deformation);
        ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());


        std::pair<ImagePointerType,ImagePointerType> deformedMoving = TransfUtilsType::warpImageWithMask(movingImage,deformation);
        
        ImageNeighborhoodIteratorPointerType tIt=new ImageNeighborhoodIteratorType(m_patchRadius,targetImage,targetImage->GetLargestPossibleRegion());
        ImageNeighborhoodIteratorPointerType aIt=new ImageNeighborhoodIteratorType(m_patchRadius,deformedMoving.first,deformedMoving.first->GetLargestPossibleRegion());
        ImageNeighborhoodIteratorPointerType mIt=new ImageNeighborhoodIteratorType(m_patchRadius,deformedMoving.second,deformedMoving.second->GetLargestPossibleRegion());
        accIt.GoToBegin();incIt.GoToBegin();tIt->GoToBegin();mIt->GoToBegin(); aIt->GoToBegin();
        for (;!accIt.IsAtEnd();++accIt,++incIt,++(*tIt),++(*mIt),++(*aIt)){
            double metricWeight=1;
            switch (metric){
            case MSD:
                metricWeight=localMSD(tIt,aIt,mIt);
                break;
            case MAD:
                metricWeight=localMAD(tIt,aIt,mIt);
                break;
            case NCC:
                metricWeight=localNCC(tIt,aIt,mIt);
                break;
            default:
                metricWeight=1;
            }
            LOGV(10)<<VAR(metricWeight)<<endl;

            accIt.Set(accIt.Get()+incIt.Get()*globalWeight*metricWeight);
        }
        delete tIt; delete aIt; delete mIt;
    }
    void updateProbabilisticSegmentationLocalMetricNew(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType targetImage, ImagePointerType movingImage,DeformationFieldPointerType deformation,MetricType metric ){
        ProbabilisticVectorImagePointerType deformedIncrement=warpProbImage(increment,deformation);
        ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());


        std::pair<ImagePointerType,ImagePointerType> deformedMoving = TransfUtilsType::warpImageWithMask(movingImage,deformation);
        FloatImagePointerType metricImage;
        switch (metric){
        case MSD:
            metricImage=Metrics<ImageType,FloatImageType>::LSSDAutoNorm(deformedMoving.first, targetImage,m_patchRadius[0],m_sigma);
            break;
        case MAD:
            metricImage=Metrics<ImageType,FloatImageType>::LSADAutoNorm(deformedMoving.first, targetImage,m_patchRadius[0],m_sigma);
            break;
        case NCC:
            metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(deformedMoving.first, targetImage,m_patchRadius[0], m_sigma);
            break;
        default:
            LOG<<"no valid metric, aborting"<<endl;
            exit(0);
        }
        LOGI(8,ImageUtils<FloatImageType>::writeImage("weightImage.nii",metricImage));
        FloatImageIteratorType weightIt(metricImage,metricImage->GetLargestPossibleRegion());
        weightIt.GoToBegin();
        accIt.GoToBegin();incIt.GoToBegin();
        for (;!accIt.IsAtEnd();++accIt,++incIt, ++weightIt){
            accIt.Set(accIt.Get()+incIt.Get()*globalWeight*weightIt.Get());
        }
    }
    ProbabilisticVectorImagePointerType createEmptyProbImageFromImage(ImagePointerType input){
        ProbabilisticVectorImagePointerType output=ProbabilisticVectorImageType::New();
        output->SetOrigin(input->GetOrigin());
        output->SetSpacing(input->GetSpacing());
        output->SetDirection(input->GetDirection());
        output->SetRegions(input->GetLargestPossibleRegion());
        output->Allocate();
        ProbabilisticPixelType p;
        p.Fill(0.0);
        output->FillBuffer(p);
        return output;
        
    }

    ProbabilisticVectorImagePointerType warpProbImage(ProbabilisticVectorImagePointerType input, DeformationFieldPointerType deformation){
        ProbabilisticVectorImagePointerType output=ProbabilisticVectorImageType::New();
        output->SetOrigin(deformation->GetOrigin());
        output->SetSpacing(deformation->GetSpacing());
        output->SetDirection(deformation->GetDirection());
        output->SetRegions(deformation->GetLargestPossibleRegion());
        output->Allocate();
        ProbImageIteratorType outIt(output,output->GetLargestPossibleRegion());
        typedef itk::VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<
            ProbabilisticVectorImageType ,double> DefaultFieldInterpolatorType;
        typename DefaultFieldInterpolatorType::Pointer interpolator=DefaultFieldInterpolatorType::New();
        interpolator->SetInputImage(input);

        DeformationIteratorType deformationIt(deformation,deformation->GetLargestPossibleRegion());
        for (outIt.GoToBegin(),deformationIt.GoToBegin();!outIt.IsAtEnd();++outIt,++deformationIt){
            IndexType index=deformationIt.GetIndex();
            typename DefaultFieldInterpolatorType::ContinuousIndexType idx(index);
            DisplacementType displacement=deformationIt.Get();
            PointType p;
            output->TransformIndexToPhysicalPoint(index,p);
            p+=displacement;
            input->TransformPhysicalPointToContinuousIndex(p,idx);
            outIt.Set(interpolator->EvaluateAtContinuousIndex(idx));
        }
        return output;
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
        std::pair<ImagePointerType,ImagePointerType> deformedMoving = TransfUtilsType::warpImageWithMask(moving,deformation);
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
    
    double DICE(ImagePointerType seg1, ImagePointerType seg2){
        typedef typename itk::LabelOverlapMeasuresImageFilter<ImageType> OverlapMeasureFilterType;
        typename OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
        seg1=FilterUtils<ImageType>::binaryThresholdingLow(seg1,1.0);
        seg2=FilterUtils<ImageType>::binaryThresholdingLow(seg2,1.0);
        filter->SetSourceImage(seg1);
        filter->SetTargetImage(seg2);
        filter->Update();
        return filter->GetDiceCoefficient(1);
    }
};//class
