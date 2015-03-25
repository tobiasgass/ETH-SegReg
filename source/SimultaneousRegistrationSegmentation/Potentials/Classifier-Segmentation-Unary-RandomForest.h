/**
 * @file   Classifier-Segmentation-Unary-RandomForest.h
 * @author Tobias Gass <gass@i6.informatik.rwth-aachen.de>
 * @date   Thu Mar  5 17:20:01 2015
 * 
 * @brief  Specialized segmentation pairwise classifiers using random forests
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



#include <iostream>
#include "itkImageDuplicator.h"
#include "itkConstNeighborhoodIterator.h"
#include <time.h>
#include <itkImageRandomConstIteratorWithIndex.h>
#include <itkImageRandomNonRepeatingConstIteratorWithIndex.h>

#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <sstream>

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"

#include "Classifier-Segmentation-Unary.h"

namespace SRS{

  using namespace libconfig;

///\brief Standard classifier that learns local posterior probabilities from intensities+class labels
  /// ...
  template<class ImageType>
    class ClassifierSegmentationUnaryRandomForest: public ClassifierSegmentationUnaryBase<ImageType>{
  protected:
    FileData m_data;
    Forest * m_Forest;
    int m_nData;
  public:
    typedef ClassifierSegmentationUnaryRandomForest            Self;
    typedef ClassifierSegmentationUnaryBase<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
    typedef typename itk::ImageRegionConstIteratorWithIndex< ImageType > ConstImageIteratorType;
    typedef typename itk::ImageRegionIteratorWithIndex< FloatImageType > FloatIteratorType;
        
  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(ClassifierSegmentationUnaryRandomForest, Object);
    itkNewMacro(Self);

    ClassifierSegmentationUnaryRandomForest(){
      LOGV(5)<<"Initializing intensity based segmentation classifier" << endl;         
         
    };
     
         
    virtual void freeMem(){
      delete m_Forest;
      m_data=FileData();
    }
    virtual void save(string filename){
      m_Forest->save(filename);
    }
    virtual void load(string filename){
      m_Forest->load(filename);
    }
    virtual void setData(std::vector<ImageConstPointerType> inputImage, ImageConstPointerType labels=NULL){
      LOGV(5)<<"Setting up data for intensity based segmentation classifier" << endl;
      long int maxTrain=std::numeric_limits<long int>::max();
      //maximal size
      long int nData=1;
      for (int d=0;d<ImageType::ImageDimension;++d)
	nData*=inputImage[0]->GetLargestPossibleRegion().GetSize()[d];
      maxTrain=maxTrain>nData?nData:maxTrain;
      LOGV(5)<<maxTrain<<" computed"<<std::endl;
      unsigned int nFeatures=inputImage.size();
      matrix<float> data(maxTrain,nFeatures);
      LOGV(5)<<maxTrain<<" matrix allocated"<<std::endl;
      std::vector<int> labelVector(maxTrain);
      std::vector<ConstImageIteratorType> iterators;
      for (unsigned int s=0;s<nFeatures;++s){
	iterators.push_back(ConstImageIteratorType(inputImage[s],inputImage[s]->GetLargestPossibleRegion()));
	iterators[s].GoToBegin();
      }
      int i=0;
      for (;!iterators[0].IsAtEnd() ; ++i)
	{
	  int label=0;
	  if (labels)
	    label=labels->GetPixel(iterators[0].GetIndex())>0;
	  //LOGV(10)<<i<<" "<<label<<" "<<nFeatures<<std::endl;
	  for (unsigned int f=0;f<nFeatures;++f){
	    int intens=(iterators[f].Get());
	    data(i,f)=intens;
	    ++iterators[f];
	  }
	  labelVector[i]=label;
                  
                  
	}

   
      m_nData=i;
      LOG<<"done adding data. "<<std::endl;
      LOG<<"stored "<<m_nData<<" samples "<<std::endl;
      m_data.setData(data);
      m_data.setLabels(labelVector);
    }

    virtual std::vector<FloatImagePointerType> evalImage(std::vector<ImageConstPointerType> inputImage){
      LOGV(5)<<"Evaluating intensity based segmentation classifier" << endl;

      setData(inputImage);
      m_Forest->eval(m_data.getData(),m_data.getLabels(),false);
      matrix<float> conf = m_Forest->getConfidences();
      std::vector<FloatImagePointerType> result(this->m_nSegmentationLabels);
      for ( int s=0;s<this->m_nSegmentationLabels;++s){
	result[s]=FilterUtils<ImageType,FloatImageType>::createEmpty(inputImage[0]);
      }
           
          
      std::vector<FloatIteratorType> iterators;
      for ( int s=0;s<this->m_nSegmentationLabels;++s){
	iterators.push_back(FloatIteratorType(result[s],result[s]->GetLargestPossibleRegion()));
	iterators[s].GoToBegin();
      }
            
      for (int i=0;!iterators[0].IsAtEnd() ; ++i){
	for ( int s=0;s<this->m_nSegmentationLabels;++s){
	  iterators[s].Set((conf(i,s)));
	  ++iterators[s];
	}
      }
      std::string suff;
      if (ImageType::ImageDimension==2){
	suff=".nii";
      }
      if (ImageType::ImageDimension==3){
	suff=".nii";
      }
      for ( int s=0;s<this->m_nSegmentationLabels;++s){
	ostringstream probabilityfilename;
	probabilityfilename<<"prob-rf-c"<<s<<suff;

	ImageUtils<FloatImageType>::writeImage(probabilityfilename.str().c_str(),result[s]);
      }
      return result;
    }
   

    virtual void train(){
      LOG<<"reading config"<<std::endl;
      string confFile("/home/gasst/work/progs/rf/randomForest.conf");///home/gasst/work/progs/rf/src/randomForest.conf");
      HyperParameters hp;
      Config configFile;

      configFile.readFile(confFile.c_str());

      // DATA
      hp.numLabeled = m_nData;//configFile.lookup("Data.numLabeled");
      hp.numClasses = this->m_nSegmentationLabels;//configFile.lookup("Data.numClasses");
      LOGV(9)<<VAR(hp.numLabeled)<<" "<<VAR(hp.numClasses)<<std::endl;
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
      LOGV(9)<<VAR(hp.numTrees)<<" "<<VAR(hp.maxTreeDepth)<<std::endl;

      LOG<<"creating forest"<<std::endl;
      m_Forest= new Forest(hp);
      LOG<<"training forest"<<std::endl;
      std::vector<double> weights(m_data.getLabels().size(),1.0);
      m_Forest->train(m_data.getData(),m_data.getLabels(),weights);
      LOG<<"done"<<std::endl;
    };



  };//class
    
  ///\brief train a random forest classifier and estimate class likelihoods in order to convert the posterior probability to a likelihood
  template<class ImageType>
    class ClassifierSegmentationUnaryRandomForestGenerativeWithGradient: public ClassifierSegmentationUnaryRandomForest<ImageType> {
  public:
    typedef ClassifierSegmentationUnaryRandomForestGenerativeWithGradient            Self;
    typedef ClassifierSegmentationUnaryRandomForest<ImageType> Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::ConstPointer ImageConstPointerType;
    typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
  public:
    /** Standard part of every itk Object. */
    itkTypeMacro(ClassifierSegmentationUnaryGenerativeWithGradient, Object);
    itkNewMacro(Self);

    virtual int mapIntensity(float intens){
      return int(min(1.0*this->m_nIntensities,1.0*this->m_nIntensities*(intens+1000)/1800));
    }
    
    virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
      LOGV(5)<<"Preparing data for intensity and gradient based segmentation classifier" << endl;
      long int maxTrain=30000000;
      //maximal size
      long int nData=1;
      for (int d=0;d<ImageType::ImageDimension;++d)
	nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
            
      maxTrain=maxTrain>nData?nData:maxTrain;
      LOG<<maxTrain<<" computed"<<std::endl;
      int nFeatures=2;
      matrix<float> data(maxTrain,nFeatures);
      LOG<<maxTrain<<" matrix allocated"<<std::endl;
      std::vector<int> labelVector(maxTrain);
      //typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
      typedef typename itk::ImageRegionConstIteratorWithIndex< ImageType > IteratorType;
      IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
      //ImageIterator.SetNumberOfSamples(maxTrain);
      int i=0;
      ImageIterator.GoToBegin();
      this->m_counts= std::vector<int>(2,0);
      //this->m_intensCounts= std::vector<int>(this->m_nIntensities,0);
      this->m_mean=0.0;
      this->m_variance=0.0;
      this->m_meanIntens=0; this->m_meanGrad=0;this->m_varianceIntens=0;this->m_varianceGrad=0;this->m_covariance=0;
      //this->m_jointCounts=std::vector<int>(this->m_nIntensities/10*this->m_nIntensities/10,0);
      this->m_jointCounts=std::vector<int>(this->m_nIntensities*this->m_nIntensities,0);
      for (;!ImageIterator.IsAtEnd() ;
	   ++ImageIterator)
	{
	  typename ImageType::IndexType idx=ImageIterator.GetIndex();
	  float grad=mapGradient(gradient->GetPixel(idx));
	  int label=labels->GetPixel(idx)>0;
	  int intens=mapIntensity(ImageIterator.Get());
	  data(i,0)=intens;
	  data(i,1)=grad;
	  labelVector[i]=label;
	  this->m_counts[label]++;
	  i++;

	  //this->m_jointCounts[floor(intens/10) + this->m_nIntensities/10*floor(grad/10)]++;
	  this->m_jointCounts[floor(intens) + this->m_nIntensities*floor(grad)]++;

	}
      std::vector<int> tmpCounts(this->m_jointCounts.size(),0);
      int kernelwidth=2;
      for (unsigned int i=0;i< this->m_jointCounts.size();++i){
	int sumCount=0;
	for (int k=-kernelwidth;k<=kernelwidth;++k){
	  int newIndex=i+k;
	  if (newIndex>0 &&newIndex<int(this->m_jointCounts.size())){
	    int w=kernelwidth-fabs(k)+1;
	    tmpCounts[i]+=w*this->m_jointCounts[newIndex];
	    sumCount+=w;
	  }
	}
	tmpCounts[i]=floor(0.5+1.0*tmpCounts[i]/sumCount);
      }
      this->m_jointCounts=tmpCounts;
      this->m_totalCount=i;
      this->m_meanIntens/=i;
      this->m_meanGrad/=i;
      LOG<<i<<" "<< this->m_varianceIntens<<" "<<  this->m_meanIntens<<" "<<this->m_varianceIntens/i- this->m_meanIntens<<endl;
      this->m_varianceIntens= this->m_varianceIntens/i- this->m_meanIntens;
      this->m_varianceGrad=this->m_varianceGrad/i-this->m_meanGrad;
      this->m_covariance= this->m_covariance/i-this->m_meanGrad*this->m_meanIntens;
      data.resize(i,nFeatures);
      std::vector<int> copy=labelVector;
      labelVector.resize(i);

      this->m_nData=i;
      std::vector<double> weights(labelVector.size());
      for (i=0;i<(int)labelVector.size();++i){
                
	weights[i]=1.0;
	//      weights[i]=1-1.0*this->m_counts[labelVector[i]]/(this->m_counts[0]+this->m_counts[1]);
      }
      this->m_weights=weights;
      LOG<<"done adding data. "<<std::endl;
      this->m_TrainData.setData(data);
      this->m_TrainData.setLabels(labelVector);
    }

    virtual void computeProbabilities(){
      LOGV(5)<<"Storing probabilities for intensity and gradient based segmentation classifier" << endl;
      this->m_probs= std::vector<float> (2*this->m_nIntensities*this->m_nIntensities,0);

      matrix<float> data(this->m_nIntensities*this->m_nIntensities,2);
      std::vector<int> labelVector(this->m_nIntensities*this->m_nIntensities,0);
      int c=0;
      for (int i=0;i<this->m_nIntensities;++i){
	for (int j=0;j<this->m_nIntensities;++j,++c){
	  data(c,0)=i;
	  data(c,1)=j;
                    
	  labelVector[c]=0;
	}
      }
      this->m_Forest->eval(data,labelVector,false);
      matrix<float> conf = this->m_Forest->getConfidences();
      LOGV(5)<<conf.size1()<<" "<<conf.size2()<<std::endl;
      c=0;
      //double min=9999;
      //double max=-1;
      std::vector<double> mins(2,99999);
      std::vector<double> maxs(2,-99999);
      LOGV(5)<<this->m_varianceIntens<<" "<<this->m_varianceGrad<<" "<<this->m_covariance<<" "<<this->m_counts[0] <<" "<<  this->m_counts[1]<<endl;
      double det=this->m_varianceIntens*this->m_varianceGrad-this->m_covariance*this->m_covariance;
      LOGV(5)<<det<<" "<<this->m_counts[0] +  this->m_counts[1]<<endl;
      //double norm=1.0/sqrt(2*3.14*det);
      for (int i=0;i<this->m_nIntensities;++i){
	for (int j=0;j<this->m_nIntensities;++j,++c){
	  for (int s=0;s<2;++s){
	    // p(s) = relative frequency
	    double p_s=  1.0*this->m_counts[s]/( this->m_counts[0]+ this->m_counts[1]);
	    //double p_x=  1.0*this->m_jointCounts[floor(i/10) + this->m_nIntensities/10*floor(j/10)]/this->m_totalCount;
	    double p_x=  1.0*this->m_jointCounts[floor(i) + this->m_nIntensities*floor(j)]/this->m_totalCount;
	    double p= conf(c,s) / p_s  * p_x ; 
	    this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p;
	    if (p<mins[0]) mins[0]=p;
	    if (p>maxs[0]) maxs[0]=p;
	  }
	}
      }
#if 1
      for (int i=0;i<this->m_nIntensities;++i){
	for (int j=0;j<this->m_nIntensities;++j,++c){
	  //  for (int s=0;s<2;++s){
	  double p0,p1;
	  p0=( this->m_probs[0*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]);//(maxs[0]);
	  if (p0==0.0) p0= 0.0000000001;
	  p1=( this->m_probs[1*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]);//(maxs[0]);
	  if (p1==0.0) p1= 0.0000000001;
	  this->m_probs[0*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p0;//<0.5?0.5:p0;
	  this->m_probs[1*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p1;//<0.5?0.5:p1;
	}
      }
#endif
    }
  

  };//class


#if 0
  ///\brief Standard classifier that learns local posterior probabilities from intensities+class labels
  /// ...
     template<class ImageType>
    class ClassifierSegmentationUnaryRandomForest: public ClassifierSegmentationUnaryBase<ImageType>{
    protected:
    
        FileData m_TrainData;
        Forest * m_Forest;
        std::vector<double> m_weights;
        std::vector<int> m_labelVector;
        matrix<float> m_data;
        matrix<float> m_conf;
        int m_nData;
        std::vector<int> m_counts, m_intensCounts;
        int m_nIntensities;
        std::vector<float> m_probs;
        double m_mean,m_variance;

    public:
        typedef ClassifierSegmentationUnaryRandomForest            Self;
        typedef ClassifierSegmentationUnaryBase<ImageType> Superclass;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;

    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(ClassifierSegmentationUnaryRandomForest, Object);
        itkNewMacro(Self);

        ClassifierSegmentationUnaryRandomForest(){
            LOGV(5)<<"Initializing intensity based segmentation classifier" << endl;
            m_data=matrix<float>(1,3);
            m_conf=matrix<float>(1,2);
            m_labelVector=std::vector<int>(1);
        }
        virtual void setNIntensities(int n){
            m_nIntensities=n;
            m_probs= std::vector<float> (2*m_nIntensities,0);
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
        virtual void evalImage(ImageConstPointerType im, ImageConstPointerType grad=NULL){
            LOGV(5)<<"Evaluating intensity based segmentation classifier" << endl;
            typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
            typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
            FloatImagePointerType result0=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
            FloatImagePointerType result1=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
            typename itk::ImageRegionConstIterator<ImageType> it(im,im->GetLargestPossibleRegion());
            for (it.GoToBegin();!it.IsAtEnd(); ++it){
                PixelType val=it.Get();
                double prob0=px_l(val,0);
                double prob1=px_l(val,1);
                result0->SetPixel(it.GetIndex(),prob0);
                result1->SetPixel(it.GetIndex(),prob1);
            }
            ImageUtils<FloatImageType>::writeImage("p0-rf.nii",result0);
            ImageUtils<FloatImageType>::writeImage("p1-rf.nii",result1);
            
        }
        virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient=NULL){
            LOGV(5)<<"Setting up data for intensity based segmentation classifier" << endl;
            int maxTrain=1000000;
            //maximal size
            long int nData=1;
            for (int d=0;d<ImageType::ImageDimension;++d)
                nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
            maxTrain=maxTrain>nData?nData:maxTrain;
            LOG<<maxTrain<<" computed"<<std::endl;
            int nFeatures=1;
            matrix<float> data(maxTrain,nFeatures);
            LOG<<maxTrain<<" matrix allocated"<<std::endl;
            std::vector<int> labelVector(maxTrain);
            typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
            //		typedef typename itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;
            IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
            ImageIterator.SetNumberOfSamples(maxTrain);
            int i=0;
            ImageIterator.GoToBegin();
            m_counts= std::vector<int>(2,0);
            //m_intensCounts= std::vector<int>(m_nIntensities,0);
            m_mean=0.0;
            m_variance=0.0;
            for (;!ImageIterator.IsAtEnd() ;
                 ++ImageIterator)
                {
                    int label=labels->GetPixel(ImageIterator.GetIndex())>0;
                    //			if (label || (m_counts[1] && 1.0*m_counts[1]/(m_counts[1]+m_counts[0]) > 0.5 )){
                    int intens=mapIntensity(ImageIterator.Get());
                    data(i,0)=intens;
                    labelVector[i]=label;
                    i++;
                    m_counts[label]++;
                    //m_intensCounts[intens]++;
                    m_mean+=intens;
                    m_variance+=intens*intens;
                    
                
                }

            m_mean/=i;
            m_variance=m_variance/i-m_mean*m_mean;
            LOG<<"data mean :" <<m_mean<<" data variance: "<<m_variance<<std::endl;
            data.resize(i,nFeatures);
            std::vector<int> copy=labelVector;
            labelVector.resize(i);

            m_nData=i;
            std::vector<double> weights(labelVector.size());
            for (i=0;i<(int)labelVector.size();++i){
                weights[i]=1.0;
            }
            m_weights=weights;
            LOG<<"done adding data. "<<std::endl;
            m_TrainData.setData(data);
            m_TrainData.setLabels(labelVector);
        };

        virtual void computeProbabilities(){
            LOGV(5)<<"Storing probabilities for intensity based segmentation classifier" << endl;
            matrix<float> data(m_nIntensities,1);
            std::vector<int> labelVector(m_nIntensities,0);
            for (int i=0;i<m_nIntensities;++i){
                double intens=1.0*i;
                data(i,0)=intens;
                labelVector[i]=0;
            }
            m_Forest->eval(data,labelVector,false);
            matrix<float> conf = m_Forest->getConfidences();
            LOG<<conf.size1()<<" "<<conf.size2()<<std::endl;
            std::vector<float> p_S(2);
            p_S[0]=1.0*m_counts[0]/(m_counts[0]+m_counts[1]);
            p_S[1]=1.0*m_counts[1]/(m_counts[0]+m_counts[1]);

            LOG<<p_S[0]<<" "<<p_S[1]<<std::endl;
            double min=9999;
            double max=-1;
            for (int i=0;i<m_nIntensities;++i){
                for (int s=0;s<2;++s){

                    //double p_x=1.0/sqrt(2*3.14*m_variance)*exp(-0.5*(i-m_mean)*(i-m_mean)/m_variance);
                    m_probs[s*m_nIntensities+i]=conf(i,s)>0?conf(i,s):0.0000001;// / p_S[s] ;//*p_x;
                    if ( m_probs[s*m_nIntensities+i] <min)
                        min=m_probs[s*m_nIntensities+i];
                    else if ( m_probs[s*m_nIntensities+i] >max){
                        max=m_probs[s*m_nIntensities+i];
                    }
                    //LOG<<i<<" "<<s<<" "<<conf(i,s)<<" "<<p_S[s]<<" "<<p_x<<" "<<  m_probs[s*m_nIntensities+i] <<" "<<-log(m_probs[s*m_nIntensities+i])<<std::endl;
                }
            }
#if 1
            for (int i=0;i<m_nIntensities;++i){
                for (int s=0;s<2;++s){
                    
                    //m_probs[s*m_nIntensities+i]=(m_probs[s*m_nIntensities+i]-min)/(max-min);
                    m_probs[s*m_nIntensities+i]=(m_probs[s*m_nIntensities+i]);///(max);
                    //LOG<<i<<" "<<s<<" "<< m_probs[s*m_nIntensities+i]<<endl;
                }
            }
#endif       
        }
        

        inline virtual int mapIntensity(float intensity){
            return intensity;
        }
        inline virtual int mapGradient(float gradient){
            return m_nIntensities*(gradient+100)/200;
        }
        virtual double px_l(float intensity,int label,float grad=0){
            int intens=mapIntensity(intensity);
            return m_probs[(label>0)*m_nIntensities+intens];
        }

        virtual void loadProbs(string filename){

        }
        virtual void saveProbs(string filename){

        }

        virtual void train(){
            LOG<<"reading config"<<std::endl;
            string confFile("/home/gasst/work/progs/rf/randomForest.conf");///home/gasst/work/progs/rf/src/randomForest.conf");
            HyperParameters hp;
            Config configFile;

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

            LOG<<"creating forest"<<std::endl;
            m_Forest= new Forest(hp);
            LOG<<"training forest"<<std::endl;
            m_Forest->train(m_TrainData.getData(),m_TrainData.getLabels(),m_weights);//	,m_weights);
            LOG<<"done"<<std::endl;
            computeProbabilities();
        };


    };
#endif

    
    template<class ImageType>
    class ClassifierSegmentationUnaryRandomforestWithGradient: public ClassifierSegmentationUnaryRandomForest<ImageType> {
    public:
        typedef ClassifierSegmentationUnaryRandomforestWithGradient            Self;
        typedef ClassifierSegmentationUnaryRandomForest<ImageType> Superclass;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
    protected:
    
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

    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(ClassifierSegmentationUnaryRandomforestWithGradient, Object);
        itkNewMacro(Self);

        ClassifierSegmentationUnaryRandomforestWithGradient(){
            LOGV(5)<<"Initializing intensity and gradient based segmentation classifier" << endl;
                        
            this->m_data=matrix<float>(1,3);
            this->m_conf=matrix<float>(1,2);
            this->m_labelVector=std::vector<int>(1);
        };
        virtual void setNIntensities(int n){
            this->m_nIntensities=n;
            this->m_probs= std::vector<float> (2*this->m_nIntensities*this->m_nIntensities,0);
        }
        virtual void freeMem(){
            delete this->m_Forest;
            this->m_data=matrix<float>(0,0);
            this->m_conf=matrix<float>(0,0);
        }
        virtual void save(string filename){
            this->m_Forest->save(filename);
        }
        virtual void load(string filename){
            this->m_Forest->load(filename);
        }
        inline virtual int mapGradient(float gradient){
            return this->m_nIntensities*(gradient+100)/200;
        }
        virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
            LOGV(5)<<"Preparing data for intensity and gradient based segmentation classifier" << endl;
            int maxTrain=3000000000;
            //maximal size
            long int nData=1;
            for (int d=0;d<ImageType::ImageDimension;++d)
                nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
            
            maxTrain=maxTrain>nData?nData:maxTrain;
            LOG<<maxTrain<<" computed"<<std::endl;
int nFeatures=2;
            matrix<float> data(maxTrain,nFeatures);
            LOG<<maxTrain<<" matrix allocated"<<std::endl;
            std::vector<int> labelVector(maxTrain);
            //typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
            typedef typename itk::ImageRegionConstIteratorWithIndex< ImageType > IteratorType;
            IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
            //ImageIterator.SetNumberOfSamples(maxTrain);
            int i=0;
            ImageIterator.GoToBegin();
            this->m_counts= std::vector<int>(2,0);
            //this->m_intensCounts= std::vector<int>(this->m_nIntensities,0);
            this->m_mean=0.0;
            this->m_variance=0.0;
            this->m_meanIntens=0; this->m_meanGrad=0;this->m_varianceIntens=0;this->m_varianceGrad=0;this->m_covariance=0;
            this->m_jointCounts=std::vector<int>(26*26,0);
            for (;!ImageIterator.IsAtEnd() ;
                 ++ImageIterator)
                {
                    typename ImageType::IndexType idx=ImageIterator.GetIndex();
                    float grad=mapGradient(gradient->GetPixel(idx));
                    int label=labels->GetPixel(idx)>0;
                    int intens=mapIntensity(ImageIterator.Get());
                    data(i,0)=intens;
                    data(i,1)=grad;
                    labelVector[i]=label;
                    this->m_counts[label]++;
                    i++;
#if 0
                    this->m_meanIntens+=intens;
                    this->m_meanGrad+=grad;
                    this->m_varianceIntens+=intens*intens;
                    this->m_varianceGrad+=grad*grad;
                    this->m_covariance+=intens*grad;
                    //joint histogram, 10 bins
                    this->m_jointCounts[floor(intens/10) + 26*floor(grad/10)]++;
#endif
                }
            this->m_totalCount=i;
            this->m_meanIntens/=i;
            this->m_meanGrad/=i;
            LOG<<i<<" "<< this->m_varianceIntens<<" "<<  this->m_meanIntens<<" "<<this->m_varianceIntens/i- this->m_meanIntens<<endl;
            this->m_varianceIntens= this->m_varianceIntens/i- this->m_meanIntens;
            this->m_varianceGrad=this->m_varianceGrad/i-this->m_meanGrad;
            this->m_covariance= this->m_covariance/i-this->m_meanGrad*this->m_meanIntens;
            data.resize(i,nFeatures);
            std::vector<int> copy=labelVector;
            labelVector.resize(i);

            this->m_nData=i;
            std::vector<double> weights(labelVector.size());
            for (i=0;i<(int)labelVector.size();++i){
                
                weights[i]=1.0;
                //      weights[i]=1-1.0*this->m_counts[labelVector[i]]/(this->m_counts[0]+this->m_counts[1]);
            }
            this->m_weights=weights;
            LOG<<"done adding data. "<<std::endl;
            this->m_TrainData.setData(data);
            this->m_TrainData.setLabels(labelVector);
        };

        virtual void computeProbabilities(){
            LOGV(5)<<"Storing probabilities for intensity and gradient based segmentation classifier" << endl;
            this->m_probs= std::vector<float> (2*this->m_nIntensities*this->m_nIntensities,0);

            matrix<float> data(this->m_nIntensities*this->m_nIntensities,2);
            std::vector<int> labelVector(this->m_nIntensities*this->m_nIntensities,0);
            int c=0;
            for (int i=0;i<this->m_nIntensities;++i){
                for (int j=0;j<this->m_nIntensities;++j,++c){
                    data(c,0)=i;
                    data(c,1)=j;
                    
                    labelVector[c]=0;
                }
            }
            this->m_Forest->eval(data,labelVector,false);
            matrix<float> conf = this->m_Forest->getConfidences();
            LOGV(5)<<conf.size1()<<" "<<conf.size2()<<std::endl;
            c=0;
            //double min=9999;
            //double max=-1;
            std::vector<double> mins(2,99999);
            std::vector<double> maxs(2,-99999);
            LOGV(5)<<this->m_varianceIntens<<" "<<this->m_varianceGrad<<" "<<this->m_covariance<<" "<<this->m_counts[0] <<" "<<  this->m_counts[1]<<endl;
            double det=this->m_varianceIntens*this->m_varianceGrad-this->m_covariance*this->m_covariance;
            LOGV(5)<<det<<" "<<this->m_counts[0] +  this->m_counts[1]<<endl;
            //double norm=1.0/sqrt(2*3.14*det);
            for (int i=0;i<this->m_nIntensities;++i){
                for (int j=0;j<this->m_nIntensities;++j,++c){
                    for (int s=0;s<2;++s){
                        // p(s) = relative frequency
                        
                        double p=conf(c,s) ;/// p_s  * p_x2 ; 
                        this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p;
                        if (p<mins[0]) mins[0]=p;
                        if (p>maxs[0]) maxs[0]=p;
                    }
                }
            }
#if 1
            for (int i=0;i<this->m_nIntensities;++i){
                for (int j=0;j<this->m_nIntensities;++j,++c){
                    for (int s=0;s<2;++s){
                        double p;
                        p=( this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]);//(maxs[0]);
                        if (p==0.0) p= 0.0000000001;
                        this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p;
                    }
                }
            }
#endif
        }
        
        inline virtual int mapIntensity(float intensity){
            return intensity;
        }
        virtual double px_l(float intensity,int label, float gradient){
            int intens=mapIntensity(intensity);
            gradient=mapGradient(gradient);
            return this->m_probs[(label>0)*this->m_nIntensities*this->m_nIntensities+intens*this->m_nIntensities+gradient];
        }
        virtual void evalImage(ImageConstPointerType im, ImageConstPointerType gradient){
            LOGV(5)<<"Evaluating intensity and gradient based segmentation classifier" << endl;
            ImageUtils<ImageType>::writeImage("komischergradient.nii",gradient);
            typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
            typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
            FloatImagePointerType result0=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
            FloatImagePointerType result1=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
            FloatImagePointerType result2=FilterUtils<ImageType,FloatImageType>::createEmpty(im);

            typename itk::ImageRegionConstIterator<ImageType> it(im,im->GetLargestPossibleRegion());
            typename itk::ImageRegionConstIterator<ImageType> itGrad(gradient,gradient->GetLargestPossibleRegion());

            for (it.GoToBegin();!it.IsAtEnd(); ++it){
                PixelType val=(it.Get());
                PixelType grad=(itGrad.Get());
                double prob0=px_l(val,0,grad);
                double prob1=px_l(val,1,grad);
                result0->SetPixel(it.GetIndex(),prob0);
                result1->SetPixel(it.GetIndex(),prob1);
                if (prob0>0 ){
                    result2->SetPixel(it.GetIndex(),prob1/prob0);
                }else{
                    result2->SetPixel(it.GetIndex(),0);
                }
            }
            ImageUtils<FloatImageType>::writeImage("p0-rf.nii",result0);
            ImageUtils<FloatImageType>::writeImage("p1-rf.nii",result1);
            ImageUtils<FloatImageType>::writeImage("prelative-rf.nii",result2);
          
        }

        virtual void loadProbs(string filename){

        }
        virtual void saveProbs(string filename){

        }

        virtual void train(){
            LOG<<"reading config"<<std::endl;
            string confFile("/home/gasst/work/progs/rf/randomForest.conf");///home/gasst/work/progs/rf/src/randomForest.conf");
            HyperParameters hp;
            Config configFile;

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

            LOG<<"creating forest"<<std::endl;
            m_Forest= new Forest(hp);
            LOG<<"training forest"<<std::endl;
            m_Forest->train(m_TrainData.getData(),m_TrainData.getLabels(),m_weights);
            LOG<<"done"<<std::endl;
            computeProbabilities();
        };


    };

}//namespace
