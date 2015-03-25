/**
 * @file   MRFRegistrationFuser.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 11:56:13 2015
 * 
 * @brief  This class implements the actual MRF fusion
 *
 * It needs as inputs a set of (lowres) deformations, and additionally a set of images in the same resolution which contain the local weights (potentials)
 * TRWS(required) is used to combine the hypotheses
 */
#pragma once


#include <limits.h>
#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include "itkGaussianImage.h"

#ifndef WITH_TRWS
#error "TRWS required, but not found"
#else
#include "typeGeneral.h"
#include "MRFEnergy.h"
#include "minimize.cpp"
#include "treeProbabilities.cpp"
#endif

#include <itkVectorGradientMagnitudeImageFilter.h>
#include "itkGaussianImage.h"

namespace MRegFuse{
  /**
   * @brief  This class implements the actual MRF fusion
   *
   * It needs as inputs a set of (lowres) deformations, and additionally a set of images in the same resolution which contain the local weights (potentials)
   * TRWS(required) is used to combine the hypotheses
   */
  template<class ImageType,class FloatPrecision=float>
    class MRFRegistrationFuser : public GaussianEstimatorVectorImage<ImageType>{

  public:
  typedef typename ImageType::Pointer  ImagePointerType;
  typedef typename ImageType::ConstPointer  ConstImagePointerType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::PixelType PixelType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::SpacingType SpacingType;
  typedef typename TransfUtils<ImageType>::DisplacementType DeformationType;
  typedef typename TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
  typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
  typedef typename ImageUtils<ImageType,FloatPrecision>::FloatImageType FloatImageType;
  typedef typename FloatImageType::Pointer FloatImagePointerType;
  typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
  typedef typename itk::ImageRegionIterator<ImageType> ImageIteratorType;
  typedef typename itk::ImageRegionIterator<DeformationFieldType> DeformationImageIteratorType;
  static const int D=ImageType::ImageDimension;
  typedef TypeGeneral TRWType;
  typedef MRFEnergy<TRWType> MRFType;
  typedef typename TRWType::REAL Real;
  typedef typename MRFType::NodeId NodeType;
  typedef typename ImageType::PointType PointType;
  typedef typename ImageType::OffsetType OffsetType;
  private:
  std::vector<DeformationFieldPointerType> m_lowResDeformations;
  std::vector<FloatImagePointerType> m_lowResLocalWeights;
  double m_gridSpacing,m_pairwiseWeight,m_alpha;
  DeformationFieldPointerType m_lowResResult, m_result;
  FloatImagePointerType m_gridImage,m_highResGridImage;
  int m_count;
  bool m_hardConstraints;
  double m_relativeLB;
  SpacingType m_gridSpacings;
  bool m_anisotropicSmoothing;
  GaussianEstimatorScalarImage<FloatImageType> m_smoothingEstimator;
  ImagePointerType m_mask,m_labelImage;
  public:
  MRFRegistrationFuser(){
    m_gridSpacing=8;
    m_pairwiseWeight=1.0;
    m_gridImage=NULL;
    m_count=0;
    m_hardConstraints=false;
    m_alpha=1.0;
    m_anisotropicSmoothing=false;
    m_mask=NULL;
  }
  void setPairwiseWeight(double w){m_pairwiseWeight=w;}
  void setAlpha(double a){m_alpha=a;}
  void setGridSpacing(double s){m_gridSpacing=s;}
  void setHardConstraints(bool b){m_hardConstraints=b;}
  void setAnisoSmoothing(bool a){m_anisotropicSmoothing=a;}
  void setMask(ImagePointerType mask){m_mask=mask;}

    
  //add deformation (with optinal weights)
  void addImage(DeformationFieldPointerType img,FloatImagePointerType weights){
    if (!m_gridImage.IsNotNull()){
      if (m_gridSpacing<=0){
	LOG<<VAR(m_gridSpacing)<<endl;
	exit(0);
      }
      //initialize
      m_highResGridImage=ImageUtils<FloatImageType>::createEmpty(weights);//TransfUtils<FloatImageType>::createEmptyImage(img);
      LOGV(7)<<"Initializing MRF Grid by downsampling the input grid of size "<<m_highResGridImage->GetLargestPossibleRegion().GetSize()<<" by a factor of "<<1.0/m_gridSpacing<<std::endl;
      m_gridImage=FilterUtils<FloatImageType>::NNResample(m_highResGridImage,
							  1.0/m_gridSpacing,
							  false);
      LOGV(7)<<"Resulting grid size is "<<m_gridImage->GetLargestPossibleRegion().GetSize()<< "with a spacing of "<<m_gridImage->GetSpacing()<<"mm."<<std::endl;
      m_gridImage->FillBuffer(0.0);
      m_highResGridImage->FillBuffer(0.0);
      m_gridSpacings=m_gridImage->GetSpacing();
    }            
      
    DeformationFieldPointerType def=img;
    if (def->GetLargestPossibleRegion().GetSize()!=m_gridImage->GetLargestPossibleRegion().GetSize())
      def=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(img,m_gridImage);
    m_lowResDeformations.push_back(def);
    //m_lowResDeformations.push_back(TransfUtils<FloatImageType>::computeBSplineTransformFromDeformationField(img,m_gridImage));
    //m_lowResDeformations.push_back(TransfUtils<FloatImageType>::computeBSplineTransformFromDeformationField(img,m_gridImage));
    ++m_count;
    if (weights.IsNotNull()){
      m_lowResLocalWeights.push_back(FilterUtils<FloatImageType>::LinearResample(weights,m_gridImage,false));
    }
    if (m_anisotropicSmoothing){
      typedef  typename itk::VectorGradientMagnitudeImageFilter<DeformationFieldType,double> FilterType;
      typename FilterType::Pointer filter=FilterType::New();
      filter->SetInput(def);
      filter->Update();
      m_smoothingEstimator.addImage(filter->GetOutput());//FilterUtils<FloatImageType>::LinearResample(weights,m_gridImage,false));
    }
  }
  void replaceFirstImage(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        
    DeformationFieldPointerType def=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(img,m_gridImage);
    m_lowResDeformations[0]=(def);
     
    if (weights.IsNotNull()){
      m_lowResLocalWeights[0]=(FilterUtils<FloatImageType>::LinearResample(weights,m_gridImage,false));
    }
     
  }

  void replaceLastImage(DeformationFieldPointerType img,FloatImagePointerType weights=NULL){
        
    DeformationFieldPointerType def=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(img,m_gridImage);
    m_lowResDeformations[m_count-1]=(def);
     
    if (weights.IsNotNull()){
      m_lowResLocalWeights[m_count-1]=(FilterUtils<FloatImageType>::LinearResample(weights,m_gridImage,false));
    }
     
  }
  //build and solve graph
  double finalize(){
    m_lowResResult=ImageUtils<DeformationFieldType>::duplicate(m_lowResDeformations[0]);
    m_labelImage=FilterUtils<FloatImageType,ImageType>::createEmptyFrom(m_gridImage);
    m_labelImage->FillBuffer(0);
      return 0.0;

  }
  double solve(){
    TRWType::REAL energy=-1, lowerBound=-1;

        
    int nRegLabels=m_count;
    bool useAuxLabel=false;
    if (useAuxLabel){
      nRegLabels++;
      LOG<<"Adding auxiliary lables to avoid problems with over-constrained hard constraints for positive jacobians"<<endl;
    }
    //build MRF
    FloatImagePointerType anisoSmoothingWeights;
    if (m_anisotropicSmoothing){
      m_smoothingEstimator.finalize();
      anisoSmoothingWeights=m_smoothingEstimator.getMean();
    }
            
    MRFType * m_optimizer;
    m_optimizer= new MRFType(TRWType::GlobalSize());
    TRWType::REAL D1[nRegLabels];
    //
    SizeType size=m_gridImage->GetLargestPossibleRegion().GetSize();
    int nRegNodes=m_gridImage->GetLargestPossibleRegion().GetNumberOfPixels();
    vector<NodeType> regNodes(nRegNodes,NULL);
        
    if (m_mask.IsNull()){
      m_mask=FilterUtils<FloatImageType,ImageType>::createEmpty(m_gridImage);
      m_mask->FillBuffer(1);
    }else{
      if (m_mask->GetLargestPossibleRegion().GetSize()!=m_gridImage->GetLargestPossibleRegion().GetSize())
	m_mask=FilterUtils<ImageType>::NNResample(m_mask,1.0/m_gridSpacing,false);
    }
        
        
    //iterate coarse grid for unaries
    FloatImageIteratorType gridIt(m_gridImage,m_gridImage->GetLargestPossibleRegion());
    ImageIteratorType maskIt;
    maskIt=ImageIteratorType(m_mask,m_mask->GetLargestPossibleRegion());
    maskIt.GoToBegin();

    bool buff;
    gridIt.GoToBegin();
    int countInside=0,countFringe=0;
    for (int d=0;!gridIt.IsAtEnd();++gridIt,++d){
            
      bool insideMask=maskIt.Get()>0;
      ++maskIt;
      if (!insideMask)
	continue;
        
      ++countInside;
      IndexType idx=gridIt.GetIndex();
      d=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx,size,buff);

      if (maskIt.Get()!=2){
	for (int l1=0;l1<nRegLabels;++l1) {
	  if (l1<m_count){
	    //D1[l1]=-log(m_lowResLocalWeights[l1]->GetPixel(idx));
	    D1[l1]=1.0-(m_lowResLocalWeights[l1]->GetPixel(idx));
	    LOGV(7)<<l1<<" "<<VAR(D1[l1])<<" "<<m_lowResLocalWeights[l1]->GetPixel(idx)<<endl;
	  }else{
	    //penalty for aux label
	    D1[l1]=2;
	  }
	}
                
	regNodes[d] = 
	  m_optimizer->AddNode(TRWType::LocalSize(nRegLabels), TRWType::NodeData(D1));
      }else{
	//get label from previous estimate and fix to that label :)
	int label=  m_labelImage->GetPixel(idx);
                
	for (int l1=0;l1<nRegLabels;++l1) {
	  if (l1==label){
	    //D1[l1]=-log(m_lowResLocalWeights[l1]->GetPixel(idx));
	    D1[l1]=1.0-(m_lowResLocalWeights[l1]->GetPixel(idx));
	    LOGV(7)<<l1<<" "<<VAR(D1[l1])<<" "<<m_lowResLocalWeights[l1]->GetPixel(idx)<<endl;
	  }else{
	    //penalty for aux label
	    D1[l1]=10000000;
	  }
	}
	regNodes[d] = 
	  m_optimizer->AddNode(TRWType::LocalSize(nRegLabels), TRWType::NodeData(D1));
	++countFringe;

      }
    }
    LOGV(1)<<VAR(countInside)<<" "<<VAR(countFringe)<<endl;
     
    //iterate coarse grid for pairwises
    gridIt.GoToBegin();
    if (m_mask.IsNotNull()){
      maskIt.GoToBegin();
    }
    for (int d=0;!gridIt.IsAtEnd();++gridIt,++d){
      bool insideMask=maskIt.Get()>0;
      ++maskIt;
      if (!insideMask)
	continue;
      int nRegLabelsNode1=nRegLabels;//maskIt.Get()!=2?nRegLabels:1;

      IndexType idx=gridIt.GetIndex();
      //LOGV(1)<<VAR(idx)<<endl;
      std::vector<DeformationType> displacements(nRegLabels);
      for (int l1=0;l1<nRegLabels;++l1){
	if (l1<m_lowResLocalWeights.size()){
	  displacements[l1]=m_lowResDeformations[l1]->GetPixel(idx);
	}
      }
      PointType point,neighborPoint;
      m_gridImage->TransformIndexToPhysicalPoint(idx,point);
      d=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx,size,buff);
      double normalizer=-1.0;
      if (anisoSmoothingWeights.IsNotNull()){
	normalizer=anisoSmoothingWeights->GetPixel(idx);
      }
            
      for (int i=0;i<D;++i){
	OffsetType off;
	off.Fill(0);
	off[i]=1;
	IndexType neighborIndex=idx+off;
	//LOGV(2)<<m_mask->GetLargestPossibleRegion().GetSize()<<" "<<VAR(neighborIndex)<<" "<<(m_mask->GetPixel(neighborIndex)>0)<<" "<<m_mask->GetPixel(neighborIndex)<<" "<<m_mask->GetPixel(idx)<<" "<<VAR((m_mask->GetPixel(idx)>0))<<endl;
	if (!(m_mask->GetPixel(neighborIndex)>0)){
	  //skip edges where neighbor pixel is outside mask, if mask is available
	  continue;
	}
	if (m_gridImage->GetLargestPossibleRegion().IsInside(neighborIndex)){
	  //LOGV(1)<<VAR(neighborIndex)<<endl;
                    
	  int nRegLabelsNode2=nRegLabels;//m_mask->GetPixel(idx)!=2?nRegLabels:1;
                    
                    
	  TRWType::REAL Vreg[nRegLabelsNode1*nRegLabelsNode2];

	  for (int l1=0;l1<nRegLabelsNode1;++l1){
	    int label1=l1;//nRegLabelsNode1!=1?l1:m_labelImage->GetPixel(idx);
	    for (int l2=0;l2<nRegLabelsNode2;++l2){
	      int label2=l2;//nRegLabelsNode2!=1?l2:m_labelImage->GetPixel(neighborIndex);
	      double weight=0.0;
	      DeformationType neighborDisplacement=m_lowResDeformations[label2]->GetPixel(neighborIndex);
	      m_gridImage->TransformIndexToPhysicalPoint(neighborIndex,neighborPoint);
	      double distanceNormalizer=(point-neighborPoint).GetNorm();
	      DeformationType displacementDifference=(displacements[label1]-neighborDisplacement);
	      if (label1<m_count && label2<m_count){
                                
                         
		{
                                  
           
		  weight=(1.0-m_alpha)*(displacementDifference.GetSquaredNorm()/(distanceNormalizer)) + (m_alpha)*(label1!=label2);
                                
		
		}
                         
	      }else if (label1<m_count || label2<m_count){
		weight=0;
	      }else{
		//do not allow aux labels next to each other? or should one?
		weight=0;//1000000;
	      }
	      if (normalizer>0){
		weight=(weight*distanceNormalizer- (normalizer*normalizer));
		weight*=weight;
                                         
	      }
	      //label1=nRegLabelsNode1>1?label1:0;
	      //                            label2=nRegLabelsNode2>1?label2:0;
	      Vreg[label1+label2*nRegLabelsNode1]=m_pairwiseWeight*weight;
                           
	    }
	  }
	  int d2=ImageUtils<ImageType>::ImageIndexToLinearIndex(neighborIndex,size,buff);
	  m_optimizer->AddEdge(regNodes[d], regNodes[d2], TRWType::EdgeData(TRWType::GENERAL,Vreg));
	}         
      }
    }

    //solve MRF
    MRFEnergy<TRWType>::Options options;
    options.m_iterMax = 1000; // maximum number of iterations
    options.m_printMinIter=1;
    options.m_printIter=1;
    options.m_eps=1e-7;
    clock_t opt_start=clock();
    m_optimizer->Minimize_TRW_S(options, lowerBound, energy);
    clock_t finish = clock();
    float t = (float) ((double)(finish - opt_start) / CLOCKS_PER_SEC);
    LOGV(2)<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound <<std::endl;

    m_relativeLB=lowerBound/energy;
    //get output and upsample

          
    DeformationImageIteratorType resIt(m_lowResResult, m_lowResResult->GetLargestPossibleRegion());
    resIt.GoToBegin();
    maskIt.GoToBegin();
    for (;!resIt.IsAtEnd();++resIt){
      IndexType idx=resIt.GetIndex();
      int linearIndex=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx,size,buff);
      bool insideMask=true;
      insideMask=maskIt.Get()!=2 && maskIt.Get();
      ++(maskIt);
      int label;
      if (insideMask){
	label=m_optimizer->GetSolution(regNodes[linearIndex]);
	//resIt.Set(m_lowResDeformations[label]->GetPixel(idx));
	m_lowResResult->SetPixel(idx,m_lowResDeformations[label]->GetPixel(idx));
	m_labelImage->SetPixel(idx,label);
      }
    }
      
     
    //m_result=TransfUtils<FloatImageType>::linearInterpolateDeformationField(m_lowResResult,m_highResGridImage);
    //m_result=TransfUtils<FloatImageType>::computeDeformationFieldFromBSplineTransform(m_lowResResult,m_highResGridImage);
    delete m_optimizer;
    return energy;
  }

  DeformationFieldPointerType getMean(){
    m_result=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(m_lowResResult,m_highResGridImage);
    return m_result;
  }
  //DeformationFieldPointerType getMean(){m_result=TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(     m_lowResDeformations[0],m_highResGridImage);return m_result;}

  DeformationFieldPointerType getLowResResult(){return    m_lowResResult;}
  ImagePointerType getLabelImage(){return m_labelImage;}
  DeformationFieldPointerType getVariance(){return NULL;}
  DeformationFieldPointerType getStdDev(){return NULL;}
  double getRelativeLB(){return m_relativeLB;}
  FloatImagePointerType getLikelihood(DeformationFieldPointerType img, double s=1.0){
    FloatImagePointerType result=TransfUtils<FloatImageType>::createEmptyFloat(img);
    return result;
  }

#define USELOCALSIGMASFORDILATION
#ifdef USELOCALSIGMASFORDILATION
  double solveUntilPosJacDet(int maxIter,double increaseSmoothing, bool useJacMask,FloatImagePointerType localDilationRadii=NULL){
    double energy;
    double minJac=-1;
    finalize();
    double mmJac=0;
    int iter=0;
    if (useJacMask && localDilationRadii->GetSpacing()!=m_gridImage->GetSpacing()){
      localDilationRadii=FilterUtils<FloatImageType>::maximumResample(localDilationRadii,m_gridImage,m_gridImage->GetSpacing()[0]/localDilationRadii->GetSpacing()[0]);
    }
    LOGI(3,ImageUtils<FloatImageType>::writeImage("resampledDilationRadii.nii",localDilationRadii));

    for (;iter<maxIter;++iter){
          
      FloatImagePointerType jacDets=TransfUtils<ImageType,float,double,double>::getJacDets(m_lowResResult);
      minJac=FilterUtils<FloatImageType>::getMin(jacDets);
      if (minJac>0.0){
	LOGV(2)<<"MinJac of coarse test was positive ("<<minJac<<"); now testing high resolution deformation.."<<endl;
	jacDets=TransfUtils<ImageType,float,double,double>::getJacDets(TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(m_lowResResult,m_highResGridImage));
	LOGI(3,ImageUtils<ImageType>::writeImage("highResNegJac.nii",FilterUtils<FloatImageType,ImageType>::binaryThresholdingHigh(jacDets,0.0)));
	minJac=FilterUtils<FloatImageType>::getMin(jacDets);
	if (jacDets->GetSpacing()!=m_gridImage->GetSpacing()){
	  jacDets=FilterUtils<FloatImageType>::minimumResample(jacDets,m_gridImage,m_gridImage->GetSpacing()[0]/jacDets->GetSpacing()[0]);
	}
      }
           
      double negJacFrac=FilterUtils<FloatImageType>::sum(FilterUtils<FloatImageType>::binaryThresholdingHigh(jacDets,0.0));
            
      negJacFrac/=jacDets->GetLargestPossibleRegion().GetNumberOfPixels();
      LOGV(2)<<VAR(iter)<<" "<<VAR(minJac)<<" "<<VAR(negJacFrac)<<endl;
         

      double fac=1.0;
      if (minJac<0.0) fac=-1.0;
      if (minJac>mmJac)
	break;
      if (useJacMask){
	int sumPixels=0;
	//ImagePointerType mask= FilterUtils<FloatImageType,ImageType>::cast(FilterUtils<FloatImageType>::binaryThresholdingHigh(jacDets,mmJac));
	ImagePointerType mask= FilterUtils<FloatImageType,ImageType>::myBinaryThresholdingHigh(jacDets,mmJac);

	if (mask->GetLargestPossibleRegion().GetSize()!=m_gridImage->GetLargestPossibleRegion().GetSize()){
	  LOG<<"SHOULD NOT HAPPEN!"<<endl;
	}


	LOGI(3,ImageUtils<ImageType>::writeImage("mask.nii",mask));
	do{
                    
	  LOGV(2)<<"Locally dilating mask with a ball of 3sigma  mm."<<endl;

	  //dilation is in pixel units -.-
	  ImagePointerType testMask=computeLocallyDilatedMask(mask,localDilationRadii);
	  LOGI(3,ImageUtils<ImageType>::writeImage("mask-dilated.nii",testMask));

	  sumPixels=FilterUtils<ImageType>::sum(testMask);
	  LOGV(3)<<VAR(sumPixels)<<endl;
	  mask=testMask;
                    
	}while (sumPixels<2);
                
                
	ImagePointerType oneMoreMask=FilterUtils<ImageType>::dilation(mask,m_gridSpacings[0]/mask->GetSpacing()[0]);
	oneMoreMask=FilterUtils<ImageType>::substract(oneMoreMask,mask);
	ImageUtils<ImageType>::multiplyImage(oneMoreMask,2);
	mask=FilterUtils<ImageType>::add(mask,oneMoreMask);

	ostringstream oss2;
	oss2<<"mask-dilated-iter"<<iter<<".mha";
	LOGV(3)<<"Saving mask to " << oss2.str()<<endl;
	LOGI(3,ImageUtils<ImageType>::writeImage(oss2.str(),mask));
	//LOGI(3,ImageUtils<ImageType>::writeImage("mask-dilated-iter0.mha",mask));
                    
	setMask(mask);
                                                        
      }
      energy=solve();
      m_pairwiseWeight*=increaseSmoothing;
                    
    }
    LOGV(1)<<"SSR iterations :"<<iter<<endl;
    return energy;

  }


  ImagePointerType computeLocallyDilatedMask(ImagePointerType mask,FloatImagePointerType dilationRadii){
       
    //LOGV(2)<<"Locally dilating mask with a ball of "<<min(100.0,50.0)<<"*minJac px."<<endl;

    typedef itk::ConnectedComponentImageFilter<ImageType,ImageType>  ConnectedComponentImageFilterType;
    typedef typename ConnectedComponentImageFilterType::Pointer ConnectedComponentImageFilterPointer;
        
    ConnectedComponentImageFilterPointer filter =
    ConnectedComponentImageFilterType::New();
        
    filter->SetInput(mask);
    filter->Update();
    ImagePointerType components=filter->GetOutput();
    int nComponents=filter->GetObjectCount();

    std::vector<double> maxSigmaPerComp(nComponents,0);

    FloatImageIteratorType jacIt(dilationRadii,dilationRadii->GetLargestPossibleRegion());
    ImageIteratorType it(components,components->GetLargestPossibleRegion());
    jacIt.GoToBegin();
    it.GoToBegin();
    for (;!it.IsAtEnd();++it,++jacIt){
      int comp=it.Get();
      if (comp>0){
	double jac=jacIt.Get();
	if (jac>maxSigmaPerComp[comp-1])
	  maxSigmaPerComp[comp-1]=jac;

      }

    } 
    for (int c=0;c<nComponents;++c){
      if (maxSigmaPerComp[c]<1) maxSigmaPerComp[c]=16;
      double dilation=max(1.0,ceil(maxSigmaPerComp[c]/m_gridSpacings[0]));
      LOGV(3)<<VAR(c)<<" "<<VAR(maxSigmaPerComp[c])<<" "<<VAR(dilation)<<endl;
      components=FilterUtils<ImageType>::dilation(components,dilation,c+1);
    }
    return FilterUtils<ImageType>::binaryThresholdingLow(components,1);
        

  }
#else
  double solveUntilPosJacDet(int maxIter,double increaseSmoothing,bool useJacMask, double ballRadius, ImagePointerType mask=NULL){
    double energy;
    double minJac=-1;
    finalize();
    double mmJac=0;
    int iter=0;
    if (mask.IsNotNull()){
      useJacMask=false;
      setMask(mask);
    }
    for (;iter<maxIter;++iter){
      //FloatImagePointerType jacDets=TransfUtils<ImageType>::getJacDets(m_lowResResult);
         
      FloatImagePointerType jacDets=TransfUtils<ImageType,float,double,double>::getJacDets(m_lowResResult);
      minJac=FilterUtils<FloatImageType>::getMin(jacDets);
      //bool lowResNegJacTest=(FilterUtils<FloatImageType>::getMin(TransfUtils<ImageType>::getJacDets(m_lowResResult))<=0);
      if (minJac>0.0){
	//            LOGV(3)<<VAR(lowResNegJacTest)<<" "<<(minJac<=0)<<endl;
	LOGV(2)<<"MinJac of coarse test was positive ("<<minJac<<"); now testing high resolution deformation.."<<endl;
	jacDets=TransfUtils<ImageType,float,double,double>::getJacDets(TransfUtils<FloatImageType>::bSplineInterpolateDeformationField(m_lowResResult,m_highResGridImage));
	LOGI(3,ImageUtils<ImageType>::writeImage("highResNegJac.nii",FilterUtils<FloatImageType,ImageType>::binaryThresholdingHigh(jacDets,0.0)));
	minJac=FilterUtils<FloatImageType>::getMin(jacDets);
	if (jacDets->GetSpacing()!=m_gridImage->GetSpacing()){
	  jacDets=FilterUtils<FloatImageType>::minimumResample(jacDets,m_gridImage,m_gridImage->GetSpacing()[0]/jacDets->GetSpacing()[0]);
	}
      }
           
      double negJacFrac=FilterUtils<FloatImageType>::sum(FilterUtils<FloatImageType>::binaryThresholdingHigh(jacDets,0.0));
            
      negJacFrac/=jacDets->GetLargestPossibleRegion().GetNumberOfPixels();
      LOGV(2)<<VAR(iter)<<" "<<VAR(minJac)<<" "<<VAR(negJacFrac)<<endl;
         

      double fac=1.0;
      if (minJac<0.0) fac=-1.0;
      if (minJac>mmJac)
	break;
      if (useJacMask){
	int sumPixels=0;
	//ImagePointerType mask= FilterUtils<FloatImageType,ImageType>::cast(FilterUtils<FloatImageType>::binaryThresholdingHigh(jacDets,mmJac));
	ImagePointerType mask= FilterUtils<FloatImageType,ImageType>::myBinaryThresholdingHigh(jacDets,mmJac);

	if (mask->GetLargestPossibleRegion().GetSize()!=m_gridImage->GetLargestPossibleRegion().GetSize()){
	  //if mask needs to be resampled, we first dilate it with a small ball to avoid 'forgetting' negative pixels due to the resampling
	  //mask=FilterUtils<ImageType>::dilation(mask,m_gridSpacings[0]);
	  //mask= FilterUtils<ImageType>::NNResample(mask,FilterUtils<FloatImageType,ImageType>::cast(m_gridImage),false);
	  LOG<<"SHOULD NOT HAPPEN!"<<endl;
	}


	LOGI(3,ImageUtils<ImageType>::writeImage("mask.nii",mask));
	do{
	  //double ballRadius=min(100.0,fac*50.0*minJac);
	  //ballRadius=m_gridSpacings[0];
	  LOGV(2)<<"dilating mask with a ball of "<<ballRadius<<" mm."<<endl;

	  //dilation is in pixel units -.-
	  //ImagePointerType testMask=FilterUtils<ImageType>::dilation(mask,ballRadius/m_gridSpacings[0]);
	  ImagePointerType testMask=computeLocallyDilatedMask(jacDets, mmJac, ballRadius);
	  LOGI(3,ImageUtils<ImageType>::writeImage("mask-dilated.nii",testMask));

	  sumPixels=FilterUtils<ImageType>::sum(testMask);
	  LOGV(3)<<VAR(sumPixels)<<endl;
	  if (sumPixels<2){
	    ballRadius*=2;
	  }else{
	    mask=testMask;
	  }
	}while (sumPixels<2);
                
                
	ImagePointerType oneMoreMask=FilterUtils<ImageType>::dilation(mask,m_gridSpacings[0]/mask->GetSpacing()[0]);
	oneMoreMask=FilterUtils<ImageType>::substract(oneMoreMask,mask);
	ImageUtils<ImageType>::multiplyImage(oneMoreMask,2);
	mask=FilterUtils<ImageType>::add(mask,oneMoreMask);
	LOGI(3,ImageUtils<ImageType>::writeImage("mask-final.nii",mask));
                    
	setMask(mask);
                                                        
      }
      energy=solve();
      m_pairwiseWeight*=increaseSmoothing;
                    
    }
    LOGV(1)<<"SSR iterations :"<<iter<<endl;
    return energy;

  }


  ImagePointerType computeLocallyDilatedMask(FloatImagePointerType jac, double jacThresh,double dilateFactor){
       
    LOGV(2)<<"Locally dilating mask with a ball of "<<min(100.0,50.0)<<"*minJac px."<<endl;
    ImagePointerType mask= (FilterUtils<FloatImageType,ImageType>::myBinaryThresholdingHigh(jac,jacThresh));
    mask=FilterUtils<ImageType>::dilation(mask,m_gridSpacings[0]/mask->GetSpacing()[0]);//,min(100.0,50.0*minJacPerComp[c])),c+1);
    typedef itk::ConnectedComponentImageFilter<ImageType,ImageType>  ConnectedComponentImageFilterType;
    typedef typename ConnectedComponentImageFilterType::Pointer ConnectedComponentImageFilterPointer;
        
    ConnectedComponentImageFilterPointer filter =
    ConnectedComponentImageFilterType::New();
        
    filter->SetInput(mask);
    filter->Update();
    ImagePointerType components=filter->GetOutput();
    int nComponents=filter->GetObjectCount();
    std::vector<double> minJacPerComp(nComponents,jacThresh);

    FloatImageIteratorType jacIt(jac,jac->GetLargestPossibleRegion());
    ImageIteratorType it(components,components->GetLargestPossibleRegion());
    jacIt.GoToBegin();
    it.GoToBegin();
    for (;!it.IsAtEnd();++it,++jacIt){
      int comp=it.Get();
      if (comp>0){
	double jac=jacIt.Get();
	if (jac<minJacPerComp[comp-1])
	  minJacPerComp[comp-1]=jac;

      }

    }
     
    for (int c=0;c<nComponents;++c){
      double dilation=max(1.0,min(50.0,fabs(dilateFactor*minJacPerComp[c]/mask->GetSpacing()[0])));
      LOGV(3)<<VAR(c)<<" "<<VAR(minJacPerComp[c])<<" "<<VAR(dilation)<<endl;
      if (dilation>0.0)
	components=FilterUtils<ImageType>::dilation(components,dilation,c+1);
    }
    return FilterUtils<ImageType>::binaryThresholdingLow(components,1);
        

  }
#endif
  double fesslerTest(){
        
    double minDiff=100000000;
    DeformationImageIteratorType gridIt(m_lowResResult, m_lowResResult->GetLargestPossibleRegion());

    for (int d=0;!gridIt.IsAtEnd();++gridIt,++d){
          
      IndexType idx=gridIt.GetIndex();
            
      PointType point,neighborPoint;
      m_gridImage->TransformIndexToPhysicalPoint(idx,point);
      DeformationType disp=gridIt.Get();
      for (int i=0;i<D;++i){
	OffsetType off;
	off.Fill(0);
	off[i]=1;
	IndexType neighborIndex=idx+off;
	if (m_gridImage->GetLargestPossibleRegion().IsInside(neighborIndex)){
	  DeformationType neighborDisplacement=m_lowResResult->GetPixel(neighborIndex);
	  m_gridImage->TransformIndexToPhysicalPoint(neighborIndex,neighborPoint);
	  DeformationType displacementDifference=(disp-neighborDisplacement);
	  //fessler penalty 
	  double k=1.0/D+0.0000001;
	  for (int d2=0;d2<D;++d2){
	    double dispDiff=-displacementDifference[d2] ;
	    if ( dispDiff < -1.0*k*m_gridSpacings[d2] ){
	      double pen=(-dispDiff + 1.0*k*m_gridSpacings[d2] );
	      if (pen<minDiff)minDiff=pen;
	    }
	  }
                    
	}         
      }
    }

    return minDiff;
  }
    
  };//class

}//namespace
