/**
 * @file   Classifier-Segmentation-Unary.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar  5 15:52:07 2015
 * 
 * @brief  Classifier interfaces and some general classes to penalize different segmentation labels of neighboring nodes.
 * 
 * 
 */

#pragma once
#include "Log.h"

#include <vector>
#include "data.h"
#include "forest.h"
#include "randomnaivebayes.h"
#include "pairforest.h"
#include "tree.h"
#include "data.h"
#include "utilities.h"
#include "hyperparameters.h"
#include <libconfig.h++>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include "itkImageDuplicator.h"
#include "itkConstNeighborhoodIterator.h"
#include <time.h>
#include <itkImageRandomConstIteratorWithIndex.h>
#include <itkImageRandomNonRepeatingConstIteratorWithIndex.h>

#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionIterator.h>

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "ImageUtils.h"


namespace SRS{
  template<class ImageType>
    class ClassifierSegmentationUnaryBase: public itk::Object{
  protected:
    int m_nSegmentationLabels;
  public:
    virtual void setNSegmentationLabels(int n){
      m_nSegmentationLabels=n;
    }
    virtual ~ClassifierSegmentationUnaryBase(){}

  }; 

  

  ///This classifier does a simple estimation of the parameters of a gaussian distribution based on intensity and gradient features. the potential then becomes the neg log of the likelihood.    
  template<class ImageType>
    class ClassifierSegmentationUnaryGaussianWithGradient: public ClassifierSegmentationUnaryBase<ImageType> {
  public:
    typedef ClassifierSegmentationUnaryGaussianWithGradient            Self;
    typedef ClassifierSegmentationUnaryBase<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
  protected:
    std::vector<double> m_meanIntens, m_meanGrad,m_varianceIntens,m_varianceGrad,m_covariance;
  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(ClassifierSegmentationUnaryGaussianWithGradient, Object);
    itkNewMacro(Self);

    ///estimate parameters of densities (one for each label).
    virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
       
      int maxTrain=10000000;
      //maximal size
      long int nData=1;
      for (int d=0;d<ImageType::ImageDimension;++d)
	nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
            
      maxTrain=maxTrain>nData?nData:maxTrain;
      LOG<<maxTrain<<" computed"<<std::endl;
      int nFeatures=2;
      typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
      //		typedef typename itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;
      IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
      ImageIterator.SetNumberOfSamples(maxTrain);
      int i=0;
      ImageIterator.GoToBegin();
      this->m_counts= std::vector<int>(2,0);
      //this->m_intensCounts= std::vector<int>(this->m_nIntensities,0);
      this->m_meanIntens=std::vector<double>(2,0.0); this->m_meanGrad=std::vector<double>(2,0.0);this->m_varianceIntens=std::vector<double>(2,0.0);this->m_varianceGrad=std::vector<double>(2,0.0);this->m_covariance=std::vector<double>(2,0.0);

      for (;!ImageIterator.IsAtEnd() ;
	   ++ImageIterator)
	{
	  typename ImageType::IndexType idx=ImageIterator.GetIndex();
	  int grad=gradient->GetPixel(idx);
	  int label=labels->GetPixel(idx)>0;
	  int intens=mapIntensity(ImageIterator.Get());
	  this->m_counts[label]++;
	  i++;
	  this->m_meanIntens[label]+=intens;
	  this->m_meanGrad[label]+=grad;
	  this->m_varianceIntens[label]+=intens*intens;
	  this->m_varianceGrad[label]+=grad*grad;
	  this->m_covariance[label]+=intens*grad;
	}

      for (int s=0;s<2;++s){
	this->m_meanIntens[s]/=i;
	this->m_meanGrad[s]/=i;
	this->m_varianceIntens[s]= this->m_varianceIntens[s]/i- this->m_meanIntens[s];
	this->m_varianceGrad[s]=this->m_varianceGrad[s]/i-this->m_meanGrad[s];
	this->m_covariance[s]= this->m_covariance[s]/i-this->m_meanGrad[s]*this->m_meanIntens[s];
      }
    };

    virtual void computeProbabilities(){
      LOG<<"ERROR, NOT IMPLEMENTED"<<std::endl;
    }
    virtual void train(){
      computeProbabilities();
    }


  };
  
  template<class ImageType>
    class ClassifierSegmentationUnaryHandcraftedBone: public ClassifierSegmentationUnaryBase<ImageType> {
  public:
    typedef ClassifierSegmentationUnaryHandcraftedBone            Self;
    typedef ClassifierSegmentationUnaryBase<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
  protected:
    
    
    int m_nIntensities;
    int bone;
    int tissue;
  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(ClassifierSegmentationUnaryHandcraftedBone, Object);
    itkNewMacro(Self);

    ClassifierSegmentationUnaryHandcraftedBone(){
      bone=(100+1000)*255.0/2000;
      tissue=(-500+1000)*255.0/2000;
    };
    virtual void setNIntensities(int n){
      m_nIntensities=n;
    }
    virtual void freeMem(){
                
    }
    virtual void save(string filename){
    }
    virtual void load(string filename){
    }

    virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
   
    };

    virtual void computeProbabilities(){
 
    }
        
    inline virtual int mapIntensity(float intensity){
      return intensity;
    }
    virtual double px_l(float imageIntensity,int segmentationLabel, int s){
      int bone=(300+1000)*255.0/2000;
      int tissue=(-500+1000)*255.0/2000;
      double segmentationProb=1;
      if (segmentationLabel>0) {
	if (imageIntensity < tissue)
	  segmentationProb =fabs(imageIntensity-tissue);
	else if (imageIntensity < bone) 
	  segmentationProb = 0.69; //log (0.5);
	else
	  segmentationProb = 0.00000000001;
      }else{
	if ((imageIntensity >  bone)  && s>128)
	  segmentationProb = fabs(imageIntensity-bone);
	else if (imageIntensity >tissue)
	  segmentationProb =0.69 ;
	else
	  segmentationProb = 0.00000000001;
                
      }
      return exp(-segmentationProb);
    }
       
    virtual void train(){
    }

    virtual void evalImage(ImageConstPointerType im, ImageConstPointerType gradient){
      ImagePointerType result0=ImageUtils<ImageType>::createEmpty(im);
      ImagePointerType result1=ImageUtils<ImageType>::createEmpty(im);
      typename itk::ImageRegionConstIterator<ImageType> it(im,im->GetLargestPossibleRegion());
      typename itk::ImageRegionConstIterator<ImageType> itGrad(gradient,gradient->GetLargestPossibleRegion());
      for (it.GoToBegin();!it.IsAtEnd(); ++it,++itGrad){
	PixelType val=it.Get();
	PixelType grad=itGrad.Get();
	double prob0=px_l(val,0,grad);
	double prob1=px_l(val,1,grad);
	//                LOG<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
	result0->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob0);
	result1->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob1);
      }
      if (false){
	if (ImageType::ImageDimension==2){
	  ImageUtils<ImageType>::writeImage("p0-marcel.nii",result0);
	  ImageUtils<ImageType>::writeImage("p1-marcel.nii",result1);
	}else{
	  ImageUtils<ImageType>::writeImage("p0-marcel.nii",result0);
	  ImageUtils<ImageType>::writeImage("p1-marcel.nii",result1);
	}
            
      }
    }
  };
  
  template<class ImageType>
    class HandcraftedBoneSegmentationClassifierMarcel: public ClassifierSegmentationUnaryBase<ImageType> {
  public:
    typedef HandcraftedBoneSegmentationClassifierMarcel            Self;
    typedef ClassifierSegmentationUnaryBase<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
  protected:
    
    
    int m_nIntensities;
    int bone;
    int tissue;
  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(HandcraftedBoneSegmentationClassifierMarcel, Object);
    itkNewMacro(Self);

    HandcraftedBoneSegmentationClassifierMarcel(){
      bone=(300);//+1000)*255.0/2000;
      tissue=(-500);//+1000)*255.0/2000;
      if (numeric_limits<PixelType>::min() == 0){
	bone+=1000;
	tissue+=1000;
      }
      if (numeric_limits<PixelType>::max() == 256){
	bone=255.0*bone/2000;
	tissue=255.0*tissue/2000;
      }
            
    };
    virtual void setNIntensities(int n){
      m_nIntensities=n;
    }
    virtual void freeMem(){
                
    }
    virtual void save(string filename){
    }
    virtual void load(string filename){
    }

    virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
   
    };

    virtual void computeProbabilities(){
 
    }
        
    inline virtual int mapIntensity(float intensity){
      return intensity;
    }
    virtual double px_l(float imageIntensity,int segmentationLabel, int s){
      //int bone=(300+1000)*255.0/2000;
      //int tissue=(-500+1000)*255.0/2000;
      double segmentationProb=1;
      if (segmentationLabel>0) {
	if (imageIntensity < tissue)
	  segmentationProb =fabs(imageIntensity-tissue);
	else if (imageIntensity < bone) 
	  segmentationProb = 0.69; //log (0.5);
	else
	  segmentationProb = 0.00000000001;
      }else{
	if ((imageIntensity >  bone)  && s>128)
	  segmentationProb = fabs(imageIntensity-bone);
	else if (imageIntensity >tissue)
	  segmentationProb =0.69 ;
	else
	  segmentationProb = 0.00000000001;
                
      }
      return exp(-segmentationProb);
    }
       
    virtual void train(){
    }

    virtual void evalImage(ImageConstPointerType im, ImageConstPointerType gradient){
      ImagePointerType result0=ImageUtils<ImageType>::createEmpty(im);
      ImagePointerType result1=ImageUtils<ImageType>::createEmpty(im);
      typename itk::ImageRegionConstIterator<ImageType> it(im,im->GetLargestPossibleRegion());
      typename itk::ImageRegionConstIterator<ImageType> itGrad(gradient,gradient->GetLargestPossibleRegion());
      for (it.GoToBegin();!it.IsAtEnd(); ++it,++itGrad){
	PixelType val=it.Get();
	PixelType grad=itGrad.Get();
	double prob0=px_l(val,0,grad);
	double prob1=px_l(val,1,grad);
	//                LOG<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
	result0->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob0);
	result1->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob1);
      }
      if (false){
	if (ImageType::ImageDimension==2){
	  ImageUtils<ImageType>::writeImage("p0-marcel.nii",result0);
	  ImageUtils<ImageType>::writeImage("p1-marcel.nii",result1);
	}else{
	  ImageUtils<ImageType>::writeImage("p0-marcel.nii",result0);
	  ImageUtils<ImageType>::writeImage("p1-marcel.nii",result1);
	}
            
      }
    }
  };
  

  template<class ImageType>
    class SegmentationClassifierProbabilityImage: public ClassifierSegmentationUnaryBase<ImageType> {
  public:
    typedef SegmentationClassifierProbabilityImage            Self;
    typedef ClassifierSegmentationUnaryBase<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
  protected:
    
    
    int m_nIntensities;
      
  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(SegmentationClassifierProbabilityImage, Object);
    itkNewMacro(Self);

    SegmentationClassifierProbabilityImage(){
          
    };
    virtual void setNIntensities(int n){
      m_nIntensities=n;
    }
    virtual void freeMem(){
                
    }
    virtual void save(string filename){
    }
    virtual void load(string filename){
    }

    virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
   
    };

    virtual void computeProbabilities(){
 
    }
        
    inline virtual int mapIntensity(float intensity){
      return intensity;
    }
    virtual double px_l(float imageIntensity,int segmentationLabel, int s){
           
      double segmentationProb=imageIntensity;
      if (ImageType::ImageDimension==2) segmentationProb/=m_nIntensities;
            
      if (segmentationLabel==0) {
	segmentationProb=1-segmentationProb;
      }
      //segmentationProb=1.0/(1.0+exp(-30.0*(segmentationProb-0.5)));
      //segmentationProb=tan(3*(segmentationProb-0.5));
      return segmentationProb;
    }
       
    virtual void train(){
    }

    virtual void evalImage(ImageConstPointerType im, ImageConstPointerType gradient){
      ImagePointerType result0=ImageUtils<ImageType>::createEmpty(im);
      ImagePointerType result1=ImageUtils<ImageType>::createEmpty(im);
      typename itk::ImageRegionConstIterator<ImageType> it(im,im->GetLargestPossibleRegion());
      typename itk::ImageRegionConstIterator<ImageType> itGrad(gradient,gradient->GetLargestPossibleRegion());
      for (it.GoToBegin();!it.IsAtEnd(); ++it,++itGrad){
	PixelType val=it.Get();
	PixelType grad=itGrad.Get();
	double prob0=px_l(val,0,grad);
	double prob1=px_l(val,1,grad);
	//                LOG<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
	result0->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob0);
	result1->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob1);
      }
      if (false){
	if (ImageType::ImageDimension==2){
	  ImageUtils<ImageType>::writeImage("p0-marcel.nii",result0);
	  ImageUtils<ImageType>::writeImage("p1-marcel.nii",result1);
	}else{
	  ImageUtils<ImageType>::writeImage("p0-marcel.nii",result0);
	  ImageUtils<ImageType>::writeImage("p1-marcel.nii",result1);
	}
            
      }
    }
  };
  
     


}//namespace SRS
