#pragma once
#include "Log.h"
/*
 * Classifier.h
 *
 *  Created on: Feb 14, 2011
 *      Author: gasst
 */


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
     class SegmentationUnaryClassifierBase: public itk::Object{
     protected:
         	  int m_nSegmentationLabels;
          public:
         	  virtual 	void setNSegmentationLabels(int n){
         		  m_nSegmentationLabels=n;
         	  }
              virtual ~SegmentationUnaryClassifierBase(){}

  }; 
  template<class ImageType>
     class SegmentationPairwiseClassifierBase: public itk::Object{
     protected	:
         	  int m_nSegmentationLabels;
          public:
         	  virtual 	void setNSegmentationLabels(int n){
         		  m_nSegmentationLabels=n;
         	  }
              virtual ~SegmentationPairwiseClassifierBase(){}

  };
  ///\brief Standard classifier that learns local posterior probabilities from intensities+class labels
  /// ...
     template<class ImageType>
    class SegmentationClassifierRandomForest: public SegmentationUnaryClassifierBase<ImageType>{
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
        typedef SegmentationClassifierRandomForest            Self;
        typedef SegmentationUnaryClassifierBase<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;

    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SegmentationClassifierRandomForest, Object);
        itkNewMacro(Self);

        SegmentationClassifierRandomForest(){
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
    
    template<class ImageType>
    class SegmentationClassifierGradient: public SegmentationClassifierRandomForest<ImageType> {
    public:
        typedef SegmentationClassifierGradient            Self;
        typedef SegmentationClassifierRandomForest<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
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
        itkTypeMacro(SegmentationClassifierGradient, Object);
        itkNewMacro(Self);

        SegmentationClassifierGradient(){
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
  

    
    template<class ImageType>
    class SegmentationGaussianClassifierGradient: public SegmentationClassifierGradient<ImageType> {
    public:
        typedef SegmentationGaussianClassifierGradient            Self;
        typedef SegmentationClassifierGradient<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
    protected:
        std::vector<double> m_meanIntens, m_meanGrad,m_varianceIntens,m_varianceGrad,m_covariance;
    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SegmentationGaussianClassifierGradient, Object);
        itkNewMacro(Self);

    
      
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

#if 0
        virtual void computeProbabilities(){
            this->m_probs= std::vector<float> (2*this->m_nIntensities*this->m_nIntensities,0);
            double min=9999;
            double max=-1;
            std::vector<double> mins(2,99999);
            std::vector<double> maxs(2,-99999);
            std::vector<double> det(2,0);
            for (int s=0;s<2;++s){
                det[s]=this->m_varianceIntens[s]*this->m_varianceGrad[s]-this->m_covariance[s]*this->m_covariance[s];
            }
        
            for (int i=0;i<this->m_nIntensities;++i){
                for (int j=0;j<this->m_nIntensities;++j){
                    double sum=0.0;
                    for (int s=0;s<2;++s){
                        double norm=1.0/sqrt(2*3.14*det[s]);
                        // p(s) = relative frequency
                        double p_s=1.0*this->m_counts[s] / ( this->m_counts[0] +  this->m_counts[1]);
                        //p(x) = multivariate gaussian
                        double bar_intens=i-this->m_meanIntens[s];
                        double bar_grad=j-this->m_meanGrad[s];
                        //-1/2(i-mu_i g-mu_g)^T E^-1 (i-mu_i g-mu_g)
                        double mahalanobis=-1.0/(2*det[s]) *(bar_intens*(bar_intens*this->m_varianceGrad[s]-bar_grad*this->m_covariance[s]) + bar_grad*(bar_grad*this->m_varianceIntens[s]-bar_intens*this->m_covariance[s]));
                        double p=norm*exp(mahalanobis);
                        sum+=p;
                        this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p;
                        LOGV(10)<<s<<" "<<i<<" "<<j<<" "<<p<<" "<<norm<<" "<<mahalanobis<<" "<<det[s]<<endl;
                        if (p<mins[0]) mins[0]=p;
                        if (p>maxs[0]) maxs[0]=p;
                    }
                    for (int s=0;s<2;++s){
                        double p=( this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]);
                        //double p_s=1.0*this->m_counts[s] / ( this->m_counts[0] +  this->m_counts[1]);
                        //p=p*p_s/sum;
                        this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p;
                    }

                }
            }
#if 0
            for (int i=0;i<this->m_nIntensities;++i){
                for (int j=0;j<this->m_nIntensities;++j){
                    for (int s=0;s<2;++s){
                        double p=( this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j] -mins[0])/(maxs[0]-mins[0]);
                        if (p==0.0) p= 0.0000000001;
                        //if (p<0.5) p=0.5;
                        this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p;
                    
                        //LOG<<i<<" "<<j<<" "<<s<<" "<<this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]<<endl;
                    }
                }
            }
#endif
        }
#else
        virtual void computeProbabilities(){
            this->m_probs= std::vector<float> (2*this->m_nIntensities*this->m_nIntensities,0);
            double min=9999;
            double max=-1;
            std::vector<double> mins(2,99999);
            std::vector<double> maxs(2,-99999);
            std::vector<double> det(2,0);
            for (int s=0;s<2;++s){
                det[s]=sqrt(this->m_varianceIntens[s]+this->m_varianceGrad[s]);
            }
        
            for (int i=0;i<this->m_nIntensities;++i){
                for (int j=0;j<this->m_nIntensities;++j){
                    double sum=0.0;
                    for (int s=0;s<2;++s){
                        double norm=1.0/sqrt(2*3.14*det[s]);
                        // p(s) = relative frequency
                        //p(x) = multivariate gaussian
                        double bar_intens=i-this->m_meanIntens[s];
                        double bar_grad=j-this->m_meanGrad[s];
                        //-1/2(i-mu_i g-mu_g)^T E^-1 (i-mu_i g-mu_g)
                        double mahalanobis=-1.0/(2*det[s]) *( bar_intens * bar_intens/(this->m_varianceIntens[s]) +
                                                              bar_grad*bar_grad/(this->m_varianceGrad[s]));
                        double p=norm*exp(mahalanobis);
                        sum+=p;
                        this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p;
                        //LOGV(10)<<s<<"  bla  "<<i<<" "<<j<<" "<<p<<" "<<norm<<" "<<mahalanobis<<" "<<det[s]<<endl;
                        if (p<mins[0]) mins[0]=p;
                        if (p>maxs[0]) maxs[0]=p;
                    }
                    for (int s=0;s<2;++s){
                        double p=( this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]);
                        //double p_s=1.0*this->m_counts[s] / ( this->m_counts[0] +  this->m_counts[1]);
                        //p=p*p_s/sum;
                        this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p;
                    }

                }
            }
        }
#endif

        virtual void loadProbs(string filename){

        }
        virtual void saveProbs(string filename){

        }

        virtual void train(){
            computeProbabilities();
        }


    };
  
    template<class ImageType>
    class HandcraftedBoneSegmentationClassifierGradient: public SegmentationClassifierGradient<ImageType> {
    public:
        typedef HandcraftedBoneSegmentationClassifierGradient            Self;
        typedef SegmentationClassifierGradient<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
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
        itkTypeMacro(HandcraftedBoneSegmentationClassifierGradient, Object);
        itkNewMacro(Self);

        HandcraftedBoneSegmentationClassifierGradient(){
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
    class HandcraftedBoneSegmentationClassifierMarcel: public SegmentationClassifierGradient<ImageType> {
    public:
        typedef HandcraftedBoneSegmentationClassifierMarcel            Self;
        typedef SegmentationClassifierGradient<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
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
    class SegmentationClassifierProbabilityImage: public SegmentationClassifierGradient<ImageType> {
    public:
        typedef SegmentationClassifierProbabilityImage            Self;
        typedef SegmentationClassifierGradient<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
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
  
    template<class ImageType>
    class SmoothnessClassifierGradient: public SegmentationClassifierRandomForest<ImageType> {
    public:
        typedef SmoothnessClassifierGradient            Self;
        typedef SegmentationClassifierRandomForest<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;    
        typedef typename FloatImageType::ConstPointer FloatImageConstPointerType;
    protected:
    
        FileData m_TrainData;
        Forest * m_Forest;
        std::vector<double> m_weights;
        std::vector<int> m_labelVector;
        matrix<float> m_data;
        matrix<float> m_conf;
        int m_nData, m_totalCount;
        std::vector<int> m_counts, m_intensCounts, m_jointCounts;
        int m_nIntensities;
        std::vector<float> m_probs;
        double m_meanIntens, m_meanGrad,m_varianceIntens,m_varianceGrad,m_covariance;
        double m_weight;
        std::vector<FloatImagePointerType> potentials;
    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SmoothnessClassifierGradient, Object);
        itkNewMacro(Self);
        
        SmoothnessClassifierGradient(){
            m_data=matrix<float>(1,3);
            m_conf=matrix<float>(1,2);
            m_labelVector=std::vector<int>(1);
            m_weight=1.0;
        };
        virtual void setNIntensities(int n){
            m_nIntensities=n;
            m_probs= std::vector<float> (2*m_nIntensities*m_nIntensities,0);
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
        };

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
#if 0

            //this code was used in conjunction with the old probability caching
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
            if(false){
                if (ImageType::ImageDimension ==2 ){
                    ImageUtils<ImageType>::writeImage("p0-rfGradient.nii",result0);
                    ImageUtils<ImageType>::writeImage("p1-rfGradient.nii",result1);
                }else{
                    ImageUtils<ImageType>::writeImage("p0-rfGradient.nii",result0);
                    ImageUtils<ImageType>::writeImage("p1-rfGradient.nii",result1);
     
                }}
#endif
        }
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
        };


    };
    template<class ImageType>
    class SmoothnessClassifierGradientContrast: public SmoothnessClassifierGradient<ImageType> {
    public:
        typedef SmoothnessClassifierGradientContrast            Self;
        typedef SmoothnessClassifierGradient<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
   

    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SmoothnessClassifierGradientContrast, Object);
        itkNewMacro(Self);

     
        virtual void freeMem(){
           
        }
      
        virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
        };

        virtual void computeProbabilities(){
       
        }
        
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
     
     

        virtual void train(bool t, string g=""){
          
        };


    };
    template<class ImageType>
    class SmoothnessClassifierUniform: public SmoothnessClassifierGradient<ImageType> {
    public:
        typedef SmoothnessClassifierUniform            Self;
        typedef SmoothnessClassifierGradient<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
   

    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SmoothnessClassifierGradientContrast, Object);
        itkNewMacro(Self);

     
        virtual void freeMem(){
           
        }
      
        virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
        };

        virtual void computeProbabilities(){
       
        }
        
        inline virtual int mapIntensity(float intensity){
            return intensity;
        }
        virtual double px_l(float intensityDiff,int label, int gradientDiff, int label2=0){
            label=label!=label2;
            return label;
        }
     
     

        virtual void train(){
          
        };


    };
    
    template<class ImageType>
    class SmoothnessClassifierSignedGradient: public SegmentationClassifierGradient<ImageType> {
    public:
        typedef SmoothnessClassifierSignedGradient            Self;
        typedef SegmentationClassifierGradient<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
    
    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SmoothnessClassifierSignedGradient, Object);
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
        };

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
            string confFile("/home/gasst/work/progs/rf/randomForest.conf");///home/gasst/work/progs/rf/src/randomForest.conf");
            HyperParameters hp;
            Config configFile;

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

    template<class ImageType>
    class SmoothnessClassifierFullMultilabelPosterior: public SegmentationClassifierGradient<ImageType> {
    public:
        typedef SmoothnessClassifierFullMultilabelPosterior            Self;
        typedef SegmentationClassifierGradient<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
    
    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SmoothnessClassifierFullMultilabelPosterior, Object);
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
        };

        virtual void computeProbabilities(){
#if 0
            this->m_probs= std::vector<float> (this->m_nSegmentationLabels*this->m_nSegmentationLabels*this->m_nIntensities*this->m_nIntensities*this->m_nIntensities*this->m_nIntensities,0);
            matrix<float> data(this->m_nIntensities*this->m_nIntensities*this->m_nIntensities*this->m_nIntensities,8);
            std::vector<int> labelVector(this->m_nIntensities*this->m_nIntensities*this->m_nIntensities*this->m_nIntensities,0);
            int c=0;
            for (int i=0;i<this->m_nIntensities;++i){
                for (int i2=0;i2<this->m_nIntensities;++i2){
                    for (int j1=0;j1<this->m_nIntensities;++j1){
                        for (int j2=0;j2<this->m_nIntensities;++j2,++c){
                            data(c,0)=(i-i2);
                            data(c,1)=(j1-j2);
                            data(c,2)=i;
                            data(c,3)=j1;
                            data(c,4)=i2;
                            data(c,5)=j2;
                            data(c,6)=fabs(i-i2);
                            data(c,7)=fabs(j1-j2);
                            labelVector[c]=0;
                        }
                    }
                }
            }
            LOG<<"evaluating forest "<<std::endl;
            this->m_Forest->eval(data,labelVector,false);
            matrix<float> conf = this->m_Forest->getConfidences();
            LOG<<conf.size1()<<" "<<conf.size2()<<std::endl;
            c=0;
            for (int i=0;i<this->m_nIntensities;++i){
                for (int i2=0;i2<this->m_nIntensities;++i2){
                    for (int j1=0;j1<this->m_nIntensities;++j1){
                        for (int j2=0;j2<this->m_nIntensities;++j2,++c){
                            for (int l=0;l<this->m_nSegmentationLabels*this->m_nSegmentationLabels;++l){
                                double p=conf(c,l) ;
                                p=p>0?p:0.0000001;
                                this->m_probs[c*this->m_nSegmentationLabels*this->m_nSegmentationLabels + l ] =p;
                            }
                        }
                    }
                }
            }
#endif
        }
        
      
        virtual double px_l(float intens1,float intens2, float grad1, float grad2,int label1,int label2){
            //            LOG<<intensityDiff<<" "<<label<<" "<<gradientDiff<<endl;
            intens1=this->mapIntensity(intens1);
            intens2=this->mapIntensity(intens2); 
            grad1=this->mapGradient(grad1);
            grad2=this->mapGradient(grad2);
#if 0
            double prob=this->m_probs[this->m_nSegmentationLabels*this->m_nSegmentationLabels*(
                                      pow(this->m_nIntensities,3)*intens1 
                                      +  pow(this->m_nIntensities,2)*intens2
                                      +  this->m_nIntensities*grad1
                                      +  grad2)
                                      + label1*this->m_nSegmentationLabels +label2];
#else
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
#endif
            return prob;
        }
#if 0
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
            if(false){
                if (ImageType::ImageDimension ==2 ){
                    ImageUtils<ImageType>::writeImage("p0-rfGradient.nii",result0);
                    ImageUtils<ImageType>::writeImage("p1-rfGradient.nii",result1);
                }else{
                    ImageUtils<ImageType>::writeImage("p0-rfGradient.nii",result0);
                    ImageUtils<ImageType>::writeImage("p1-rfGradient.nii",result1);
     
                }}
            
        }
#endif
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
            string confFile("/home/gasst/work/progs/rf/randomForest.conf");///home/gasst/work/progs/rf/src/randomForest.conf");
            HyperParameters hp;
            Config configFile;

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
            computeProbabilities();
        }
    };


    template<class ImageType>
    class SegmentationGenerativeClassifierGradient: public SegmentationClassifierGradient<ImageType> {
    public:
        typedef SegmentationGenerativeClassifierGradient            Self;
        typedef SegmentationClassifierGradient<ImageType> Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SegmentationGenerativeClassifierGradient, Object);
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
#if 0
                    this->m_meanIntens+=intens;
                    this->m_meanGrad+=grad;
                    this->m_varianceIntens+=intens*intens;
                    this->m_varianceGrad+=grad*grad;
                    this->m_covariance+=intens*grad;
                    //joint histogram, 10 bins
#endif
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
  

    };
  


}//namespace SRS
