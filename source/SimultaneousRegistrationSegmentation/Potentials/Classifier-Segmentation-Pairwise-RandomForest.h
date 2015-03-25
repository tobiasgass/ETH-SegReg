/**
 * @file   Classifier-Segmentation-Pairwise-RandomForest.h
 * @author gasst <gasst@ETHSEGREG>
 * @date   Thu Mar  5 13:17:53 2015
 * 
 * @brief  Several classes that learn pairwise potentials from a labelled image using random forests
 * 
 * 
 */

#pragma once

#ifndef WITH_RF
#error "Compilation without RF library will not work, ERROR!"
#endif

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

#include "Classifier-Segmentation-Pairwise.h"
namespace SRS{
  
  ///\brief random forest classifier which learns pairwise potentials from image intensity and an additional (gradient or other) image  
  ///Probabilities are cached by discretizing the intensity space, and storing the associated likelihoods in a table
  template<class ImageType>  
    class ClassifierSegmentationPairwiseRandomForestWithGradient: public ClassifierSegmentationPairwiseBase<ImageType> {
  public:
    typedef ClassifierSegmentationPairwiseRandomForestWithGradient            Self;
    typedef ClassifierSegmentationPairwiseBase<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;    
    typedef typename FloatImageType::ConstPointer FloatImageConstPointerType;
  protected:
    int m_nIntensities;
    FileData m_TrainData;
    Forest * m_Forest;
    std::vector<double> m_weights;
    std::vector<int> m_labelVector;
    matrix<float> m_data;
    matrix<float> m_conf;
    int m_nData, m_totalCount;
    std::vector<int> m_counts, m_intensCounts, m_jointCounts;
    std::vector<float> m_probs;
    double m_meanIntens, m_meanGrad,m_varianceIntens,m_varianceGrad,m_covariance;
    double m_weight;
    std::vector<FloatImagePointerType> potentials;
  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(ClassifierSegmentationPairwiseRandomForestWithGradient, Object);
    itkNewMacro(Self);
        
    ClassifierSegmentationPairwiseRandomForestWithGradient(){
      m_data=matrix<float>(1,3);
      m_conf=matrix<float>(1,2);
      m_labelVector=std::vector<int>(1);
      m_weight=1.0;
    }
    virtual void setNIntensities(int n){
      this->m_nIntensities=n;
      m_probs= std::vector<float> (2*this->m_nIntensities*this->m_nIntensities,0);
    }
    virtual void freeMem(){
      delete m_Forest;
      m_data=matrix<float>(0,0);
      m_conf=matrix<float>(0,0);
    }
    virtual void save(string filename){
      m_Forest->save(filename);
    }
    virtual void load(string filename){
      m_Forest->load(filename);
    }
    virtual void SetWeight(double w){ m_weight=w;}

    ///Convert image data to random forest data structure
    virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
       
      int maxTrain=3000000;
      //maximal size
      long int nData=1;
      for (int d=0;d<ImageType::ImageDimension;++d)
	nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
      nData*=ImageType::ImageDimension;

      maxTrain=maxTrain>nData?nData:maxTrain;
      LOG<<maxTrain<<" computed"<<std::endl;
      int nFeatures=2;
      matrix<float> data(maxTrain,nFeatures);
      LOG<<maxTrain<<" matrix allocated"<<std::endl;
      std::vector<int> labelVector(maxTrain);
      typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
      IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
      ImageIterator.SetNumberOfSamples(maxTrain);
      int i=0;
      ImageIterator.GoToBegin();
      this->m_counts= std::vector<int>(2,0);
      //this->m_intensCounts= std::vector<int>(this->m_nIntensities,0);
      for (;!ImageIterator.IsAtEnd()&&i<maxTrain ;
	   ++ImageIterator)
	{
	  typename ImageType::IndexType idx=ImageIterator.GetIndex();
	  for ( int d=0;d<ImageType::ImageDimension && i<maxTrain;++d){
	    typename ImageType::OffsetType off;
	    off.Fill(0);
	    if ((int)idx[d]<(int)intensities->GetLargestPossibleRegion().GetSize()[d]-1){
	      off[d]+=1;
	      //    LOG<<d<<" "<<i<<" "<<idx+off<<" "<<intensities->GetLargestPossibleRegion().GetSize()<<endl;
	      int grad1=gradient->GetPixel(idx);
	      int label1=labels->GetPixel(idx);
	      int intens1=mapIntensity(ImageIterator.Get());
	      int grad2=gradient->GetPixel(idx+off);
	      int label2=labels->GetPixel(idx+off);
	      int intens2=mapIntensity(intensities->GetPixel(idx+off));
	      data(i,0)=fabs(intens1-intens2);
	      data(i,1)=fabs(grad1-grad2);
                    
	      labelVector[i]=label1!=label2;
	      this->m_counts[label1!=label2]++;
	      i++;                
	    }
	  }
                    
	}
      LOG<<"finished adding data" <<endl;
      m_totalCount=i;
      data.resize(i,nFeatures);
      std::vector<int> copy=labelVector;
      labelVector.resize(i);

      this->m_nData=i;
      std::vector<double> weights(labelVector.size());
      for (i=0;i<(int)labelVector.size();++i){
                
	weights[i]=1.0;
             
      }
      this->m_weights=weights;
      LOG<<"done adding data. "<<std::endl;
      this->m_TrainData.setData(data);
      this->m_TrainData.setLabels(labelVector);
    }

    virtual void computeProbabilities(){
      this->m_probs= std::vector<float> (2*this->m_nIntensities*this->m_nIntensities,0);
      matrix<float> data(this->m_nIntensities*this->m_nIntensities,2);
      std::vector<int> labelVector(this->m_nIntensities*this->m_nIntensities,0);
      int c=0;
      for (int i=0;i<this->m_nIntensities;++i){
	for (int j=0;j<this->m_nIntensities;++j,++c){
	  data(c,0)=i;
	  data(c,1)=j;
                    
	  labelVector[c]=i>0;
	}
      }
      LOG<<"evaluating forest "<<std::endl;
      this->m_Forest->eval(data,labelVector,false);
      matrix<float> conf = this->m_Forest->getConfidences();
      LOG<<conf.size1()<<" "<<conf.size2()<<std::endl;
      c=0;
      for (int i=0;i<this->m_nIntensities;++i){
	for (int j=0;j<this->m_nIntensities;++j,++c){
	  for (int s=0;s<2;++s){
	    // p(s) = relative frequency
	    //double p_s=1.0*this->m_counts[s] / ( this->m_counts[0] +  this->m_counts[1]);
	    double p=conf(c,s) ;/// p_s  * p_x2 ; 
	    //LOG<<p<<std::endl;
	    p=p>0?p:0.0000001;
	    this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p;
	  }
	}
      }

    }
        
    inline virtual int mapIntensity(float intensity){
      return intensity;
    }
    virtual double px_l(float intensityDiff,int label, int gradientDiff, int label2=-1){
      //            LOG<<intensityDiff<<" "<<label<<" "<<gradientDiff<<endl;
      label=(label!=label2);
      intensityDiff=fabs(intensityDiff);
      gradientDiff=fabs(gradientDiff);
      double prob=this->m_probs[(label>0)*this->m_nIntensities*this->m_nIntensities+intensityDiff*this->m_nIntensities+gradientDiff];
      return prob;
    }

    virtual void evalImage(ImageConstPointerType im, ImageConstPointerType gradient){
      LOG<<"EVALIMAGE is deprecated"<<std::endl;
    }

    ///compute and cache potential values for input image+gradient image
    virtual void cachePotentials(ImageConstPointerType im,ImageConstPointerType grad){

      typedef typename itk::ImageRegionConstIterator< ImageType > IteratorType;
      typedef itk::Image<float, ImageType::ImageDimension> FloatImageType;
      typedef typename FloatImageType::Pointer FloatImagePointerType;
      typedef typename FloatImageType::ConstPointer FloatImageConstPointerType;
      typedef typename itk::ImageRegionIterator< FloatImageType > NewIteratorType;
      IteratorType iterator(im, im->GetLargestPossibleRegion());
      IteratorType iterator2(grad, grad->GetLargestPossibleRegion());
             
      int nPixels=1;
      for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	nPixels*=im->GetLargestPossibleRegion().GetSize()[d];
      }
      matrix<float> data(nPixels*ImageType::ImageDimension,2);
      std::vector<int> labelVector(nPixels*ImageType::ImageDimension,0);
      int c=0;
      //store observations
      typedef typename ImageType::OffsetType OffsetType;
      for (iterator.GoToBegin(),iterator2.GoToBegin();!iterator.IsAtEnd();++iterator,++iterator2){
	typename ImageType::IndexType idx1=iterator.GetIndex();
                 
	for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	  OffsetType off;
	  off.Fill(0);
	  float val11=iterator.Get();
	  float val21=iterator2.Get();
	  float val12=0;
	  float val22=0;
	  if (idx1[d]<(int)im->GetLargestPossibleRegion().GetSize()[d]-1){
	    off[d]+=1;
	    typename ImageType::IndexType idx2=idx1+off;
	    val12=im->GetPixel(idx2);
	    val22=grad->GetPixel(idx2);
	  }
	  float absDiff1=fabs(val11-val12);
	  float absDiff2=fabs(val21-val22);
	  data(c,0)=absDiff1;
	  data(c,1)=absDiff2;
	  c++;
	}
      }

      //eval observations
      this->m_Forest->eval(data,labelVector,false);
      matrix<float> conf = this->m_Forest->getConfidences();

      //store result in images
      potentials=std::vector<FloatImagePointerType>(ImageType::ImageDimension);
      std::vector<NewIteratorType *> potentialIterators(ImageType::ImageDimension);
      for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	potentials[d]=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
	potentialIterators[d] = new   NewIteratorType( potentials[d],  potentials[d]->GetLargestPossibleRegion());
	potentialIterators[d]->GoToBegin();
      }
      c=0;
      for (iterator.GoToBegin(),iterator2.GoToBegin();!iterator.IsAtEnd();++iterator,++iterator2){
	for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	  potentialIterators[d]->Set((conf(c,0)));
	  ++c;
	  ++(*potentialIterators[d]);
	}
      }
      for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	typedef itk::RescaleIntensityImageFilter<FloatImageType,ImageType> CasterType;
	typename CasterType::Pointer caster=CasterType::New();
	caster->SetOutputMinimum( numeric_limits<typename ImageType::PixelType>::min() );
	caster->SetOutputMaximum( numeric_limits<typename ImageType::PixelType>::max() );
	caster->SetInput(potentials[d]);
	caster->Update();
	ostringstream smoothFilename;
	smoothFilename<<"smoothRF-d"<<d;
	if (ImageType::ImageDimension == 2){
	  smoothFilename<<".nii";}
	else{
	  smoothFilename<<".nii";}
	//ImageUtils<ImageType>::writeImage(smoothFilename.str(),(ImageConstPointerType)caster->GetOutput());
	delete  potentialIterators[d];
      }
    }

    ///do the same as above but return the images with the probabilities to the caller
    virtual std::vector<FloatImagePointerType> getProbabilities(ImageConstPointerType im,ImageConstPointerType grad){
      typedef typename itk::ImageRegionConstIterator< ImageType > IteratorType;
      typedef typename itk::ImageRegionIterator< FloatImageType > NewIteratorType;
      IteratorType iterator(im, im->GetLargestPossibleRegion());
      IteratorType iterator2(grad, grad->GetLargestPossibleRegion());
             
      int nPixels=1;
      for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	nPixels*=im->GetLargestPossibleRegion().GetSize()[d];
      }
      matrix<float> data(nPixels*ImageType::ImageDimension,2);
      std::vector<int> labelVector(nPixels*ImageType::ImageDimension,0);
      int c=0;
      //store observations
      typedef typename ImageType::OffsetType OffsetType;
      for (iterator.GoToBegin(),iterator2.GoToBegin();!iterator.IsAtEnd();++iterator,++iterator2){
	typename ImageType::IndexType idx1=iterator.GetIndex();
                 
	for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	  OffsetType off;
	  off.Fill(0);
	  float val11=iterator.Get();
	  float val21=iterator2.Get();
	  float val12=0;
	  float val22=0;
	  if (idx1[d]<(int)im->GetLargestPossibleRegion().GetSize()[d]-1){
	    off[d]+=1;
	    typename ImageType::IndexType idx2=idx1+off;
	    val12=im->GetPixel(idx2);
	    val22=grad->GetPixel(idx2);
	  }
	  float absDiff1=fabs(val11-val12);
	  float absDiff2=fabs(val21-val22);
	  data(c,0)=absDiff1;
	  data(c,1)=absDiff2;
	  c++;
	}
      }

      //eval observations
      this->m_Forest->eval(data,labelVector,false);
      matrix<float> conf = this->m_Forest->getConfidences();

      //store result in images
      potentials=std::vector<FloatImagePointerType>(ImageType::ImageDimension);
      std::vector<NewIteratorType *> potentialIterators(ImageType::ImageDimension);
      for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	potentials[d]=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
	potentialIterators[d] = new   NewIteratorType( potentials[d],  potentials[d]->GetLargestPossibleRegion());
	potentialIterators[d]->GoToBegin();
      }
      c=0;
      for (iterator.GoToBegin(),iterator2.GoToBegin();!iterator.IsAtEnd();++iterator,++iterator2){
	for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	  potentialIterators[d]->Set((conf(c,0)));
	  ++c;
	  ++(*potentialIterators[d]);
	}
      }
      for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	typedef itk::RescaleIntensityImageFilter<FloatImageType,ImageType> CasterType;
	typename CasterType::Pointer caster=CasterType::New();
	caster->SetOutputMinimum( numeric_limits<typename ImageType::PixelType>::min() );
	caster->SetOutputMaximum( numeric_limits<typename ImageType::PixelType>::max() );
	caster->SetInput(potentials[d]);
	caster->Update();
	ostringstream smoothFilename;
	smoothFilename<<"smoothRF-d"<<d;
	if (ImageType::ImageDimension == 2){
	  smoothFilename<<".nii";}
	else{
	  smoothFilename<<".nii";}
	LOGI(10,ImageUtils<ImageType>::writeImage(smoothFilename.str(),(ImageConstPointerType)caster->GetOutput()));
	delete  potentialIterators[d];
      }
      return potentials;
    }


    virtual double getCachedPotential(IndexType idx1,IndexType idx2){          
      for (unsigned int d=0;d<ImageType::ImageDimension;++d){
	int diff= idx1[d]-idx2[d];
	if (diff>0){
	  return potentials[d]->GetPixel(idx1);
	}else if (diff<0){
	  return potentials[d]->GetPixel(idx2);
	}
      }
    }
    virtual void loadProbs(string filename){

      ifstream myFile (filename.c_str(), ios::in | ios::binary);
      if (myFile){
	myFile.read((char*)(&this->m_probs[0]),2*this->m_nIntensities*this->m_nIntensities*sizeof(float) );
	LOG<<" read m_segmentationPosteriorProbs from disk"<<std::endl;
      }else{
	LOG<<" error reading m_segmentationPosteriorProbs"<<std::endl;
	exit(0);

      }

    }
    virtual void saveProbs(string filename){
      ofstream myFile (filename.c_str(), ios::out | ios::binary);
      myFile.write ((char*)(&this->m_probs[0]),2*this->m_nIntensities*this->m_nIntensities*sizeof(float) );
    }

    virtual void train( bool train,string filename=""){
      LOG<<"reading config"<<std::endl;
      string confFile("randomForest.conf");
      HyperParameters hp;
      libconfig::Config configFile;

      configFile.readFile(confFile.c_str());

      // DATA
      hp.trainData = (const char*) configFile.lookup("Data.trainData");
      hp.trainLabels = (const char*) configFile.lookup("Data.trainLabels");
      hp.testData = (const char*) configFile.lookup("Data.testData");
      hp.testLabels = (const char*) configFile.lookup("Data.testLabels");
      hp.numLabeled = m_nData;//configFile.lookup("Data.numLabeled");
      hp.numClasses = configFile.lookup("Data.numClasses");

      // TREE
      hp.maxTreeDepth = configFile.lookup("Tree.maxDepth");
      hp.bagRatio = configFile.lookup("Tree.bagRatio");
      hp.numRandomFeatures = configFile.lookup("Tree.numRandomFeatures");
      hp.numProjFeatures = configFile.lookup("Tree.numProjFeatures");
      hp.useRandProj = configFile.lookup("Tree.useRandProj");
      hp.useGPU = configFile.lookup("Tree.useGPU");
      hp.useSubSamplingWithReplacement = configFile.lookup("Tree.subSampleWR");
      hp.verbose = configFile.lookup("Tree.verbose");
      hp.useInfoGain = configFile.lookup("Tree.useInfoGain");


      // FOREST
      hp.numTrees = configFile.lookup("Forest.numTrees");
      hp.useSoftVoting = configFile.lookup("Forest.useSoftVoting");
      hp.saveForest = configFile.lookup("Forest.saveForest");

      if (train){
	LOG<<"creating forest"<<std::endl;
	m_Forest= new Forest(hp);
	LOG<<"training forest"<<std::endl;
	m_Forest->train(m_TrainData.getData(),m_TrainData.getLabels(),m_weights);
	LOG<<"done"<<std::endl;
	computeProbabilities();
	if ((filename!=""))
	  m_Forest->save(filename);
      }else{
	m_Forest=new Forest(hp);
	if ((filename!=""))
	  m_Forest->load(filename);
	else{
	  LOG<<"NO FILENAME GIVEN FOR RF, aborting"<<endl;
	  exit(0);
	}
	computeProbabilities();
      }
    } // train
  };//class


 
  template<class ImageType>
    class ClassifierSegmentationPairwiseRandomForestSignedGradient: public ClassifierSegmentationPairwiseRandomForestWithGradient<ImageType> {
  public:
    typedef ClassifierSegmentationPairwiseRandomForestSignedGradient            Self;
    typedef ClassifierSegmentationPairwiseRandomForestWithGradient<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
    
  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(ClassifierSegmentationPairwiseRandomforestSignedGradient, Object);
    itkNewMacro(Self);
        
    virtual void setNIntensities(int n){
      this->m_nIntensities=n;
    }
          
          
    virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
       
      int maxTrain=3000000;
      //maximal size
      long int nData=1;
      for (int d=0;d<ImageType::ImageDimension;++d)
	nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
      nData*=ImageType::ImageDimension;

      maxTrain=maxTrain>nData?nData:maxTrain;
      LOG<<maxTrain<<" computed"<<std::endl;
      int nFeatures=2;
      matrix<float> data(maxTrain,nFeatures);
      LOG<<maxTrain<<" matrix allocated"<<std::endl;
      std::vector<int> labelVector(maxTrain);
      typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
      IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
      ImageIterator.SetNumberOfSamples(maxTrain);
      int i=0;
      ImageIterator.GoToBegin();
      this->m_counts= std::vector<int>(3,0);
      //this->m_intensCounts= std::vector<int>(this->m_nIntensities,0);
      for (;!ImageIterator.IsAtEnd()&&i<maxTrain ;
	   ++ImageIterator)
	{
	  typename ImageType::IndexType idx=ImageIterator.GetIndex();
	  for ( int d=0;d<ImageType::ImageDimension && i<maxTrain;++d){
	    typename ImageType::OffsetType off;
	    off.Fill(0);
	    if ((int)idx[d]<(int)intensities->GetLargestPossibleRegion().GetSize()[d]-1){
	      off[d]+=1;
	      //    LOG<<d<<" "<<i<<" "<<idx+off<<" "<<intensities->GetLargestPossibleRegion().GetSize()<<endl;
	      int grad1=gradient->GetPixel(idx);
	      int label1=labels->GetPixel(idx)>0;
	      int intens1=mapIntensity(ImageIterator.Get());
	      int grad2=gradient->GetPixel(idx+off);
	      int label2=labels->GetPixel(idx+off)>0;
	      int intens2=mapIntensity(intensities->GetPixel(idx+off));
	      data(i,0)=(intens1-intens2)+this->m_nIntensities;
	      data(i,1)=(grad1-grad2)+this->m_nIntensities;
                               
	      int label=0;
	      if (label1 &&!label2)
		label=1;
	      else if (label2 &&!label1)
		label=2;
	      labelVector[i]=label;
	      this->m_counts[label]++;
	      i++;                
	    }
	  }
                    
	}
      LOG<<"finished adding data" <<endl;
      this->m_totalCount=i;
      data.resize(i,nFeatures);
      std::vector<int> copy=labelVector;
      labelVector.resize(i);

      this->m_nData=i;
      std::vector<double> weights(labelVector.size());
      for (i=0;i<(int)labelVector.size();++i){
                
	weights[i]=1.0;
             
      }
      this->m_weights=weights;
      LOG<<"done adding data. "<<std::endl;
      this->m_TrainData.setData(data);
      this->m_TrainData.setLabels(labelVector);
    }

    virtual void computeProbabilities(){
      this->m_probs= std::vector<float> (3*2*2*this->m_nIntensities*this->m_nIntensities,0);
      matrix<float> data(2*2*this->m_nIntensities*this->m_nIntensities,2);
      std::vector<int> labelVector(2*2*this->m_nIntensities*this->m_nIntensities,0);
      int c=0;
      for (int i=0;i<2*this->m_nIntensities;++i){
	for (int j=0;j<2*this->m_nIntensities;++j,++c){
	  data(c,0)=i;
	  data(c,1)=j;
                    
	  labelVector[c]=i>0;
	}
      }
      LOG<<"evaluating forest "<<std::endl;
      this->m_Forest->eval(data,labelVector,false);
      matrix<float> conf = this->m_Forest->getConfidences();
      LOG<<conf.size1()<<" "<<conf.size2()<<std::endl;
      c=0;
      for (int i=0;i<2*this->m_nIntensities;++i){
	for (int j=0;j<2*this->m_nIntensities;++j,++c){
	  for (int s=0;s<3;++s){
	    // p(s) = relative frequency
	    //double p_s=1.0*this->m_counts[s] / ( this->m_counts[0] +  this->m_counts[1]);
	    double p=conf(c,s) ;/// p_s  * p_x2 ; 
	    //LOG<<p<<std::endl;
	    p=p>0?p:0.0000001;
	    this->m_probs[s*2*2*this->m_nIntensities*this->m_nIntensities+i*2*this->m_nIntensities+j]=p;
	  }
	}
      }

    }
        
    inline virtual int mapIntensity(float intensity){
      return intensity;
    }
    virtual double px_l(float intensityDiff,int label1, int gradientDiff, int label2=-1){
      //            LOG<<intensityDiff<<" "<<label<<" "<<gradientDiff<<endl;
      int label=0;
      if (label1 &&!label2)
	label=1;
      else if (label2 &&!label1)
	label=2;
      intensityDiff=fabs(intensityDiff)+this->m_nIntensities;
      gradientDiff=fabs(gradientDiff)+this->m_nIntensities;
      double prob=this->m_probs[(label>0)*this->m_nIntensities*2*2*this->m_nIntensities+intensityDiff*2*this->m_nIntensities+gradientDiff];
      return prob;
    }
    virtual void evalImage(ImageConstPointerType im, ImageConstPointerType gradient){
      ImagePointerType result0=ImageUtils<ImageType>::createEmpty(im);
      ImagePointerType result1=ImageUtils<ImageType>::createEmpty(im);
      typename itk::ImageRegionConstIterator<ImageType> it(im,im->GetLargestPossibleRegion());
      typename itk::ImageRegionConstIterator<ImageType> itGrad(gradient,gradient->GetLargestPossibleRegion());
      PixelType multiplier=1;
      if (ImageType::ImageDimension !=0 ){
	multiplier=std::numeric_limits<PixelType>::max();
      }
      for (it.GoToBegin();!it.IsAtEnd(); ++it,++itGrad){
	PixelType val=it.Get();
	PixelType grad=itGrad.Get();
	double prob0=px_l(val,0,grad);
	double prob1=px_l(val,1,grad);
	//                LOG<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
	result0->SetPixel(it.GetIndex(),(PixelType)(multiplier*prob0));
	result1->SetPixel(it.GetIndex(),(PixelType)(multiplier*prob1));
      }
      if(true){
	if (ImageType::ImageDimension ==2 ){
	  LOGI(10,ImageUtils<ImageType>::writeImage("p0-rfGradient.nii",result0));
	  LOGI(10,ImageUtils<ImageType>::writeImage("p1-rfGradient.nii",result1));
	}else{
	  LOGI(10,ImageUtils<ImageType>::writeImage("p0-rfGradient.nii",result0));
	  LOGI(10,ImageUtils<ImageType>::writeImage("p1-rfGradient.nii",result1));
     
	}}
            
    }

    virtual void loadProbs(string filename){
                
      ifstream myFile (filename.c_str(), ios::in | ios::binary);
      if (myFile){
	myFile.read((char*)(&this->m_probs[0]),3*2*2*this->m_nIntensities*this->m_nIntensities*sizeof(float) );
	LOG<<" read m_segmentationPosteriorProbs from disk"<<std::endl;
      }else{
	LOG<<" error reading m_segmentationPosteriorProbs"<<std::endl;
	exit(0);

      }

    }
    virtual void saveProbs(string filename){
      ofstream myFile (filename.c_str(), ios::out | ios::binary);
      myFile.write ((char*)(&this->m_probs[0]),3*2*2*this->m_nIntensities*this->m_nIntensities*sizeof(float) );
    }
    virtual void train(){
      LOG<<"reading config"<<std::endl;
      string confFile("randomForest.conf");///home/gasst/work/progs/rf/src/randomForest.conf");
      HyperParameters hp;
      libconfig::Config configFile;

      configFile.readFile(confFile.c_str());

      // DATA
      hp.trainData = (const char*) configFile.lookup("Data.trainData");
      hp.trainLabels = (const char*) configFile.lookup("Data.trainLabels");
      hp.testData = (const char*) configFile.lookup("Data.testData");
      hp.testLabels = (const char*) configFile.lookup("Data.testLabels");
      hp.numLabeled = this->m_nData;//configFile.lookup("Data.numLabeled");
      hp.numClasses =3;// configFile.lookup("Data.numClasses");

      // TREE
      hp.maxTreeDepth = configFile.lookup("Tree.maxDepth");
      hp.bagRatio = configFile.lookup("Tree.bagRatio");
      hp.numRandomFeatures = configFile.lookup("Tree.numRandomFeatures");
      hp.numProjFeatures = configFile.lookup("Tree.numProjFeatures");
      hp.useRandProj = configFile.lookup("Tree.useRandProj");
      hp.useGPU = configFile.lookup("Tree.useGPU");
      hp.useSubSamplingWithReplacement = configFile.lookup("Tree.subSampleWR");
      hp.verbose = configFile.lookup("Tree.verbose");
      hp.useInfoGain = configFile.lookup("Tree.useInfoGain");


      // FOREST
      hp.numTrees = configFile.lookup("Forest.numTrees");
      hp.useSoftVoting = configFile.lookup("Forest.useSoftVoting");
      hp.saveForest = configFile.lookup("Forest.saveForest");

      LOG<<"creating forest"<<std::endl;
      this->m_Forest= new Forest(hp);
      LOG<<"training forest"<<std::endl;
      this->m_Forest->train(this->m_TrainData.getData(),this->m_TrainData.getLabels(),this->m_weights);
      LOG<<"done"<<std::endl;
      computeProbabilities();
    }
  };//class


    ///\brief Train a full posterior model of all label combinations given the intensity and gradient images. This contrasts the other classifiers, which only train models of the labels of neighboring nodes being the same (or different)
    ///This classifier also uses several non-linear combinations of intensity/gradient features to avoid overfitting (too much)
  template<class ImageType>
    class ClassifierSegmentationPairwiseRandomForestMultilabelPosterior: public ClassifierSegmentationPairwiseRandomForestWithGradient<ImageType> {
  public:
    typedef ClassifierSegmentationPairwiseRandomForestMultilabelPosterior            Self;
    typedef ClassifierSegmentationPairwiseRandomForestWithGradient<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
    
  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(ClassifierSegmentationPairwiseRandomForestMultilabelPosterior, Object);
    itkNewMacro(Self);
        
    virtual void setNIntensities(int n){
      this->m_nIntensities=n;
    }
          
          
    virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
       
      int maxTrain=3000000;
      //maximal size
      long int nData=1;
      for (int d=0;d<ImageType::ImageDimension;++d)
	nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
      nData*=ImageType::ImageDimension;

      maxTrain=maxTrain>nData?nData:maxTrain;
      LOG<<maxTrain<<" computed"<<std::endl;
      int nFeatures=8;
      matrix<float> data(maxTrain,nFeatures);
      LOG<<maxTrain<<" matrix allocated"<<std::endl;
      std::vector<int> labelVector(maxTrain);
      typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
      IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
      ImageIterator.SetNumberOfSamples(maxTrain);
      int i=0;
      ImageIterator.GoToBegin();
      this->m_counts= std::vector<int>(this->m_nSegmentationLabels*this->m_nSegmentationLabels,0);
      //this->m_intensCounts= std::vector<int>(this->m_nIntensities,0);
      for (;!ImageIterator.IsAtEnd()&&i<maxTrain ;
	   ++ImageIterator)
	{
	  typename ImageType::IndexType idx=ImageIterator.GetIndex();
	  for ( int d=0;d<ImageType::ImageDimension && i<maxTrain;++d){
	    typename ImageType::OffsetType off;
	    off.Fill(0);
	    off[d]+=pow(int(-1),int(i%3));
	    typename ImageType::IndexType newInd=idx+off;
	    if ((int)newInd[d]<(int)intensities->GetLargestPossibleRegion().GetSize()[d] &&(int)newInd[d]>=0 ){
	      //alternate between +1 and -1

	      //    LOG<<d<<" "<<i<<" "<<idx+off<<" "<<intensities->GetLargestPossibleRegion().GetSize()<<endl;
	      int grad1=mapGradient(gradient->GetPixel(idx));
	      int label1=labels->GetPixel(idx);
	      int intens1=mapIntensity(ImageIterator.Get());
	      int grad2=mapGradient(gradient->GetPixel(newInd));
	      int label2=labels->GetPixel(newInd);
	      int intens2=mapIntensity(intensities->GetPixel(newInd));
	      data(i,0)=(intens1-intens2);
	      data(i,1)=(grad1-grad2);
	      data(i,2)=intens1;
	      data(i,3)=grad1;
	      data(i,4)=intens2;
	      data(i,5)=grad2;
	      data(i,6)=fabs(intens1-intens2);
	      data(i,7)=fabs(grad1-grad2);
	      int label=label1+this->m_nSegmentationLabels*label2;
	      labelVector[i]=label;
	      this->m_counts[label]++;
	      i++;                
	    }
	  }
                    
	}
      LOG<<"finished adding data" <<endl;
      this->m_totalCount=i;
      data.resize(i,nFeatures);
      std::vector<int> copy=labelVector;
      labelVector.resize(i);

      this->m_nData=i;
      std::vector<double> weights(labelVector.size());
      for (i=0;i<(int)labelVector.size();++i){
                
	weights[i]=1.0;
             
      }
      this->m_weights=weights;
      LOG<<"done adding data. "<<std::endl;
      this->m_TrainData.setData(data);
      this->m_TrainData.setLabels(labelVector);
    }

    ///likelihood of observing image given the image appearance.
    ///This function signature is INCOMPATIBLE with current calls
    virtual double px_l(float intens1,float intens2, float grad1, float grad2,int label1,int label2){
      
      //            LOG<<intensityDiff<<" "<<label<<" "<<gradientDiff<<endl;
      intens1=this->mapIntensity(intens1);
      intens2=this->mapIntensity(intens2); 
      grad1=this->mapGradient(grad1);
      grad2=this->mapGradient(grad2);

      matrix<float> data(1,8);
      std::vector<int> labelVector(1,label1*this->m_nSegmentationLabels +label2);
      data(0,0)=(intens1-intens2);
      data(0,1)=(grad1-grad2);
      data(0,2)=intens1;
      data(0,3)=grad1;
      data(0,4)=intens2;
      data(0,5)=grad2;
      data(0,6)=fabs(intens1-intens2);
      data(0,7)=fabs(grad1-grad2);
      this->m_Forest->eval(data,labelVector,false);
      double prob= this->m_Forest->getConfidences()(0,label1*this->m_nSegmentationLabels +label2);
      return prob;
    }

    virtual void loadProbs(string filename){
                
      ifstream myFile (filename.c_str(), ios::in | ios::binary);
      if (myFile){
	myFile.read((char*)(&this->m_probs[0]),3*2*2*this->m_nIntensities*this->m_nIntensities*sizeof(float) );
	LOG<<" read m_segmentationPosteriorProbs from disk"<<std::endl;
      }else{
	LOG<<" error reading m_segmentationPosteriorProbs"<<std::endl;
	exit(0);

      }

    }
    virtual void saveProbs(string filename){
      ofstream myFile (filename.c_str(), ios::out | ios::binary);
      myFile.write ((char*)(&this->m_probs[0]),3*2*2*this->m_nIntensities*this->m_nIntensities*sizeof(float) );
    }
    virtual void train(bool t,string f=""){
      LOG<<"reading config"<<std::endl;
      string confFile("randomForest.conf");
      HyperParameters hp;
      libconfig::Config configFile;

      configFile.readFile(confFile.c_str());

      // DATA
      hp.trainData = (const char*) configFile.lookup("Data.trainData");
      hp.trainLabels = (const char*) configFile.lookup("Data.trainLabels");
      hp.testData = (const char*) configFile.lookup("Data.testData");
      hp.testLabels = (const char*) configFile.lookup("Data.testLabels");
      hp.numLabeled = this->m_nData;//configFile.lookup("Data.numLabeled");
      hp.numClasses =this->m_nSegmentationLabels*this->m_nSegmentationLabels;// configFile.lookup("Data.numClasses");

      // TREE
      hp.maxTreeDepth = configFile.lookup("Tree.maxDepth");
      hp.bagRatio = configFile.lookup("Tree.bagRatio");
      hp.numRandomFeatures = configFile.lookup("Tree.numRandomFeatures");
      hp.numProjFeatures = configFile.lookup("Tree.numProjFeatures");
      hp.useRandProj = configFile.lookup("Tree.useRandProj");
      hp.useGPU = configFile.lookup("Tree.useGPU");
      hp.useSubSamplingWithReplacement = configFile.lookup("Tree.subSampleWR");
      hp.verbose = configFile.lookup("Tree.verbose");
      hp.useInfoGain = configFile.lookup("Tree.useInfoGain");


      // FOREST
      hp.numTrees = configFile.lookup("Forest.numTrees");
      hp.useSoftVoting = configFile.lookup("Forest.useSoftVoting");
      hp.saveForest = configFile.lookup("Forest.saveForest");

      LOG<<"creating forest"<<std::endl;
      this->m_Forest= new Forest(hp);
      LOG<<"training forest"<<std::endl;
      this->m_Forest->train(this->m_TrainData.getData(),this->m_TrainData.getLabels(),this->m_weights);
      LOG<<"done"<<std::endl;
      this->computeProbabilities();
    }
  };//class

  

}//namespace SRS
