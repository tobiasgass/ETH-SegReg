/**
 * @file   Classifier-Segmentation-Pairwise.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Fri Mar  6 16:42:44 2015
 * 
 * @brief  Base for classifiers to return segmentation pairwise potentials
 * 
 * 
 */

#pragma once
#include "Log.h"

#include <iostream>

#include "itkObject.h"




namespace SRS{
  
  ///\brief abstract lass for segmentation pairwise classifiers
  template<class ImageType>
    class ClassifierSegmentationPairwiseBase: public itk::Object{
  protected	:
    int m_nSegmentationLabels;
  public:
    virtual void setNSegmentationLabels(int n){
      m_nSegmentationLabels=n;
    }
    virtual ~ClassifierSegmentationPairwiseBase(){}

    ///function that allows mapping the intensities in a unified way to a usable range
    virtual int mapIntensity(float intensity)=0;

    ///returns the likelihood of observing intensity/gradient difference between neighboring points given their segmentation labels
    virtual double px_l(float intensityDiff,int label, int gradientDiff, int label2=0)=0;
    
    ///training method
    virtual void train(bool t, string g=""){     
    };
  };

  ///\brief classifier 'stump', returns exp(-contrast) if neihboring labels are different and 0 otherwise. Fixed sigma
  template<class ImageType>
    class ClassifierSegmentationUnaryLocalContrast: public ClassifierSegmentationPairwiseBase<ImageType> {
  public:
    typedef ClassifierSegmentationUnaryLocalContrast            Self;
    typedef ClassifierSegmentationPairwiseBase<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
   

  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(ClassifierSegmentationUnaryLocalContrast, Object);
    itkNewMacro(Self);

   

        
    inline virtual int mapIntensity(float intensity){
      return intensity;
    }
    virtual double px_l(float intensityDiff,int label, int gradientDiff, int label2=0){
              
      label=label!=label2;
      //            LOG<<intensityDiff<<" "<<label<<" "<<gradientDiff<<endl;
      //double prob=this->m_probs[(label>0)*this->m_nIntensities*this->m_nIntensities+intensityDiff*this->m_nIntensities+gradientDiff];
      intensityDiff=fabs(intensityDiff);
      gradientDiff=fabs(gradientDiff);
      double prob;
      if (gradientDiff<0){
	//LOG<<gradientDiff<<endl;
	prob=0;
      }else{
	//intensityDiff*=intensityDiff;
	gradientDiff=(gradientDiff*gradientDiff);
	prob=0.95*exp(-this->m_weight*0.00005*gradientDiff);
	//LOG<<gradientDiff<<" "<<prob<<endl;
      }
      if (label){
	prob=1-prob;
      }
      return prob;
    }
     
     

   


  };

///\brief classifier 'stump', returns 1 if neighboring labels are not equal und 0 otherwise
  template<class ImageType>
    class ClassifierSegmentationPairwiseUniform: public ClassifierSegmentationPairwiseBase<ImageType> {
  public:
    typedef ClassifierSegmentationPairwiseUniform            Self;
    typedef ClassifierSegmentationPairwiseBase<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
   

  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(ClassifierSegmentationPairwiseUniform, Object);
    itkNewMacro(Self);

    virtual double px_l(float intensityDiff,int label, int gradientDiff, int label2=0){
      label=label!=label2;
      return label;
    }
     
  
  };
    

  


}//namespace SRS
