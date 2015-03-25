/**
 * @file   Segmentation-Fusion-Modular.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 14:17:19 2015
 * 
 * @brief   Modular Segmentation Fusion.
 * 
 * 
 */

#pragma once

#include <stdio.h>
#include <iostream>
#ifdef WITH_GCO
#include "GCoptimization.h"
#include "graph.h"
#endif

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
#include <itkLabelOverlapMeasuresImageFilter.h>
#include "Metrics.h"

namespace SSSP{
  ///\brief Modular Segmentation Fusion.
  ///
  /// Allows for choosing between (locally) weighted averaging and GC-based fusion (if available)
  template <class ImageType, int nSegmentationLabels>
    class SegmentationFusionModular{
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
      feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
      ArgumentParser * as=new ArgumentParser(argc,argv);
      std::string deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir=".",outputSuffix="",weightListFilename="", imageFileListAtlas="";
      int verbose=0;
      double radius=3;
      int maxHops=1;
      std::string metricName="NCC";
      std::string weightingName="uniform";
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
      as->parameter ("globalOneHopWeight", globalOneHopWeight,"global weight for one hop segmentations (vs. zero hop)",false);
      as->option ("AREG", AREG,"use AREG to select intermediate targets");
      as->option ("lateFusion", lateFusion,"fuse segmentations late. maxHops=1");
      as->option ("dontCacheDeformations", dontCacheDeformations,"read deformations only when needed to save memory. higher IO load!");
      as->option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
      as->parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
      as->parameter ("verbose", verbose,"get verbose output",false);
      as->help();
      as->parse();
      std::string suffix;
      if (D==2)
	suffix=".png";
      else
	suffix=".nii";

     

      //late fusion is only well defined for maximal 1 hop.
      //it requires to explicitly compute all n!/(n-nHops) deformation paths to each image and is therefore infeasible for nHops>1
      //also strange to implement
      if (lateFusion)
	maxHops=min(maxHops,1);

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
      std::map<std::string,int> *targetIDMap,*atlasSegmentationIDMap, *atlasIDMap;
      targetIDMap=new   std::map<std::string,int> ;
      atlasSegmentationIDMap=new   std::map<std::string,int> ;
      atlasIDMap=new   std::map<std::string,int> ;

      typedef typename ImageListType::iterator ImageListIteratorType;
      LOG<<"Reading atlas segmentations."<<endl;
      inputAtlasSegmentations = readImageList( atlasSegmentationFileList,atlasSegmentationIDMap );
        
      LOG<<VAR(atlasSegmentationIDMap)<<endl;
      if (D==2){
	//fix png segmentations
	for (ImageListIteratorType it=inputAtlasSegmentations->begin();it!=inputAtlasSegmentations->end();++it){
	  std::ostringstream tmpSegmentationFilename;
	  tmpSegmentationFilename<<outputDir<<"/atlassegmentation-weighting"<<weightingName<<"-metric"<<metricName<<"-ATLAS"<<it->first<<"-hop0"<<suffix;
	  ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),it->second);
	  ImageUtils<ImageType>::multiplyImage(it->second,1.0*(nSegmentationLabels-1)/(std::numeric_limits<PixelType>::max()-1));
	  std::ostringstream tmpSegmentationFilename2;
	  tmpSegmentationFilename2<<outputDir<<"/atlassegmentation-weighting"<<weightingName<<"-metric"<<metricName<<"-ATLAS"<<it->first<<"-afterNorm"<<suffix;
	  ImageUtils<ImageType>::writeImage(tmpSegmentationFilename2.str().c_str(),it->second);
	}
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
	atlasImages=targetImages;
	atlasIDMap=targetIDMap;
      }
        
      LOGV(2)<<VAR(metric)<<" "<<VAR(weighting)<<endl;
      LOGV(2)<<VAR(m_sigma)<<" "<<VAR(lateFusion)<<" "<<VAR(m_patchRadius)<<endl;

      if (dontCacheDeformations){
	LOG<<"Reading deformation file names."<<endl;
      }else{
	LOG<<"CACHING all deformations!"<<endl;
      }
      std::map< std::string, map <std::string, DeformationFieldPointerType> > deformationCache;
      std::map< std::string, map <std::string, std::string> > deformationFilenames;
      std::map<std::string, std::map<std::string, float> > globalWeights;
       
      
      logSetStage("Zero Hop");
      LOG<<"Computing"<<std::endl;
      std::map<std::string,ProbabilisticVectorImagePointerType> probabilisticSegmentations;
        
      //generate atlas probabilistic segmentations from atlas segmentations
      for (ImageListIteratorType atlasImageIterator=inputAtlasSegmentations->begin();atlasImageIterator!=inputAtlasSegmentations->end();++atlasImageIterator){ 
	//iterate over atlass
	std::string atlasID = atlasImageIterator->first;
	LOGV(3)<<VAR(atlasID)<<endl;
	probabilisticSegmentations[atlasID]=segmentationToProbabilisticVector(atlasImageIterator->second);
	std::ostringstream tmpSegmentationFilename2;
	tmpSegmentationFilename2<<outputDir<<"/ATLASsegmentation-weighting"<<weightingName<<"-metric"<<metricName<<"-target"<<atlasID<<"-hop0-ProbImage.mha";
	LOGI(4,ImageUtils<ProbabilisticVectorImageType>::writeImage(tmpSegmentationFilename2.str().c_str(),probabilisticSegmentations[atlasID]));
      }



      //generate zero-hop target segmentations
      for (ImageListIteratorType targetImageIterator=targetImages->begin();targetImageIterator!=targetImages->end();++targetImageIterator){                //iterate over targets
	std::string targetID= targetImageIterator->first;
	if (atlasSegmentationIDMap->find(targetID)==atlasSegmentationIDMap->end()){ //do not calculate segmentation for atlas images
	  probabilisticSegmentations[targetID]=createEmptyProbImageFromImage( targetImageIterator->second);
	  int atlasN=0;
	  for (ImageListIteratorType atlasIterator=inputAtlasSegmentations->begin();
	       atlasIterator!=inputAtlasSegmentations->end() && atlasN<useNAtlases;
	       ++atlasIterator,++atlasN)
	    {//iterate over atlases
	      std::string atlasID=atlasIterator->first;
	      LOGV(4)<<VAR(atlasID)<<" "<<VAR(targetID)<<endl;
                       
	      ImagePointerType atlasSegmentation=atlasIterator->second;
	      ProbabilisticVectorImagePointerType probAtlasSegmentation= probabilisticSegmentations[atlasID];
	      double weight=1.0;
                   
	      //update
	      if (weighting==UNIFORM || metric == NONE || (lateFusion && nAtlases==1)){
		updateProbabilisticSegmentationUniform(probabilisticSegmentations[targetID],probAtlasSegmentation,weight);
	      }else{
		ImagePointerType targetImage= targetImageIterator->second;
		ImagePointerType atlasImage=(*atlasImages)[(*atlasIDMap)[atlasID]].second;
		if (weighting==GLOBAL){
		  updateProbabilisticSegmentationGlobalMetric(probabilisticSegmentations[targetID],probAtlasSegmentation,weight,targetImage,atlasImage,metric);
		}else if (weighting==LOCAL){
		  updateProbabilisticSegmentationLocalMetricNew(probabilisticSegmentations[targetID],probAtlasSegmentation,weight,targetImage,atlasImage,metric);
		}

	      }
	    }
	}
           
      }//finished zero-hop segmentation
      LOG<<"done"<<endl;

      LOGV(1)<<"Storing zero-hop segmentations."<<endl;
      for (ImageListIteratorType targetImageIterator=targetImages->begin();targetImageIterator!=targetImages->end();++targetImageIterator){
	std::string targetID= targetImageIterator->first;
	if (atlasSegmentationIDMap->find(targetID)==atlasSegmentationIDMap->end()){ 
	  ImagePointerType outputImage;
	  if (graphCut){
#ifdef WITH_GCO
	    outputImage=probSegmentationToSegmentationGraphcutMultiLabel(probabilisticSegmentations[targetID],targetImageIterator->second,smoothness*inputAtlasSegmentations->size(),m_graphCutSigma);
#else
	    LOG<<"GC library not available, output will be empty!"<<endl;
#endif
	  }else{
	    outputImage=probSegmentationToSegmentationLocal(probabilisticSegmentations[targetID]);
	  }

	  std::ostringstream tmpSegmentationFilename;
	  tmpSegmentationFilename<<outputDir<<"/segmentation-weighting"<<weightingName<<"-metric"<<metricName<<"-target"<<targetID<<"-hop0"<<suffix;
	  ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),outputImage);
	  std::ostringstream tmpSegmentationFilename2;
	  tmpSegmentationFilename2<<outputDir<<"/segmentation-weighting"<<weightingName<<"-metric"<<metricName<<"-target"<<targetID<<"-hop0-ProbImage.mha";
	  LOGI(4,ImageUtils<ProbabilisticVectorImageType>::writeImage(tmpSegmentationFilename2.str().c_str(),probabilisticSegmentations[targetID]));
                
	}
      }



      return 1;
    }//run
  protected:
    ImageListType * readImageList(std::string filename, std::map<std::string,int> * indexMap){
      ImageListType * result=new  ImageListType;
      ifstream ifs(filename.c_str());
      if (!ifs){
	LOG<<"could not read "<<filename<<endl;
	exit(0);
      }
      int c=0;
      while( ! ifs.eof() ) 
	{
	  std::string imageID;
	  ifs >> imageID;                
	  if (imageID!=""){
	    ImagePointerType img;
	    std::string imageFileName ;
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

#ifdef WITH_GCO
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
#endif

    ///normalize probabilities in prob image by sum(p)
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

    ///normalize probabilities by dividing by nAtlases
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

#ifdef WITH_GCO
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
#endif

    void updateProbabilisticSegmentationUniform(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight){
      ProbabilisticVectorImagePointerType deformedIncrement=increment;
      ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
      ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());
      for (accIt.GoToBegin(),incIt.GoToBegin();!incIt.IsAtEnd();++incIt,++accIt){
	accIt.Set(accIt.Get()+incIt.Get()*globalWeight);
      }
    }

    void updateProbabilisticSegmentationGlobalMetric(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType targetImage, ImagePointerType movingImage,MetricType metric ){
       
      double metricWeight=0;
      typedef typename itk::LinearInterpolateImageFunction<ImageType> InterpolatorType;
      typename InterpolatorType::Pointer interpolator=InterpolatorType::New();
      interpolator->SetInputImage(movingImage);

      typedef typename itk::IdentityTransform<float,D> DTTransformType;
      typename DTTransformType::Pointer transf=DTTransformType::New();
      ProbabilisticVectorImagePointerType deformedIncrement=increment;
      ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
      ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());
     
      LOGV(10)<<VAR(metricWeight)<<endl;
      for (accIt.GoToBegin(),incIt.GoToBegin();!incIt.IsAtEnd();++incIt,++accIt){
	accIt.Set(accIt.Get()+incIt.Get()*globalWeight*metricWeight);
      }
    }
    
    void updateProbabilisticSegmentationLocalMetric(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType targetImage, ImagePointerType movingImage,MetricType metric ){
      ProbabilisticVectorImagePointerType deformedIncrement=increment;
      ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
      ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());


      std::pair<ImagePointerType,ImagePointerType> deformedMoving;
        
        
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

    
    void updateProbabilisticSegmentationLocalMetricNew(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType targetImage, ImagePointerType movingImage,MetricType metric ){
      ProbabilisticVectorImagePointerType deformedIncrement=increment;
      ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
      ProbImageIteratorType incIt(deformedIncrement,deformedIncrement->GetLargestPossibleRegion());


      std::pair<ImagePointerType,ImagePointerType> deformedMoving;
        
      deformedMoving.first=movingImage;
        
      FloatImagePointerType metricImage;
      switch (metric){
      case MSD:
	metricImage=Metrics<ImageType,FloatImageType>::LSSDNorm(deformedMoving.first, targetImage,m_patchRadius[0],m_sigma);
	break;
      case MAD:
	metricImage=Metrics<ImageType,FloatImageType>::LSADNorm(deformedMoving.first, targetImage,m_patchRadius[0],m_sigma);
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

}//namespace
