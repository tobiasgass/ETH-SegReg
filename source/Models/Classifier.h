/*
 * Classifier.h
 *
 *  Created on: Feb 14, 2011
 *      Author: gasst
 */

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_
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
using namespace std;
using namespace boost::numeric::ublas;
using namespace libconfig;

namespace itk{
    template<class ImageType>
    class SegmentationClassifier: public itk::Object{
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
        typedef SegmentationClassifier            Self;
        typedef itk::Object Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ImageConstPointerType;
        typedef typename ImageType::PixelType PixelType;
        typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;

    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SegmentationClassifier, Object);
        itkNewMacro(Self);

        SegmentationClassifier(){
            m_data=matrix<float>(1,3);
            m_conf=matrix<float>(1,2);
            m_labelVector=std::vector<int>(1);
        };
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
            ImagePointerType result0=ImageUtils<ImageType>::createEmpty(im);
            ImagePointerType result1=ImageUtils<ImageType>::createEmpty(im);
            typename itk::ImageRegionConstIterator<ImageType> it(im,im->GetLargestPossibleRegion());
            for (it.GoToBegin();!it.IsAtEnd(); ++it){
                PixelType val=it.Get();
                double prob0=px_l(val,0);
                double prob1=px_l(val,1);
                result0->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob0);
                result1->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob1);
            }
            ImageUtils<ImageType>::writeImage("p0-rf.png",result0);
            ImageUtils<ImageType>::writeImage("p1-rf.png",result1);
            
        }
        virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient=NULL){
        
            int maxTrain=1000000;
            //maximal size
            long int nData=1;
            for (int d=0;d<ImageType::ImageDimension;++d)
                nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
            maxTrain=maxTrain>nData?nData:maxTrain;
            std::cout<<maxTrain<<" computed"<<std::endl;
            int nFeatures=1;
            matrix<float> data(maxTrain,nFeatures);
            std::cout<<maxTrain<<" matrix allocated"<<std::endl;
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
            cout<<"data mean :" <<m_mean<<" data variance: "<<m_variance<<std::endl;
            data.resize(i,nFeatures);
            std::vector<int> copy=labelVector;
            labelVector.resize(i);

            m_nData=i;
            std::vector<double> weights(labelVector.size());
            for (i=0;i<(int)labelVector.size();++i){
                weights[i]=1.0;
            }
            m_weights=weights;
            std::cout<<"done adding data. "<<std::endl;
            m_TrainData.setData(data);
            m_TrainData.setLabels(labelVector);
        };

        virtual void computeProbabilities(){
       
            matrix<float> data(m_nIntensities,1);
            std::vector<int> labelVector(m_nIntensities,0);
            for (int i=0;i<m_nIntensities;++i){
                double intens=1.0*i;
                data(i,0)=intens;
                labelVector[i]=0;
            }
            m_Forest->eval(data,labelVector,false);
            matrix<float> conf = m_Forest->getConfidences();
            std::cout<<conf.size1()<<" "<<conf.size2()<<std::endl;
            std::vector<float> p_S(2);
            p_S[0]=1.0*m_counts[0]/(m_counts[0]+m_counts[1]);
            p_S[1]=1.0*m_counts[1]/(m_counts[0]+m_counts[1]);

            std::cout<<p_S[0]<<" "<<p_S[1]<<std::endl;
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
                    //cout<<i<<" "<<s<<" "<<conf(i,s)<<" "<<p_S[s]<<" "<<p_x<<" "<<  m_probs[s*m_nIntensities+i] <<" "<<-log(m_probs[s*m_nIntensities+i])<<std::endl;
                }
            }
#if 1
            for (int i=0;i<m_nIntensities;++i){
                for (int s=0;s<2;++s){
                    
                    //m_probs[s*m_nIntensities+i]=(m_probs[s*m_nIntensities+i]-min)/(max-min);
                    m_probs[s*m_nIntensities+i]=(m_probs[s*m_nIntensities+i]);///(max);
                    //cout<<i<<" "<<s<<" "<< m_probs[s*m_nIntensities+i]<<endl;
                }
            }
#endif       
        }
        

        inline virtual int mapIntensity(float intensity){
            return intensity;
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
            std::cout<<"reading config"<<std::endl;
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

            std::cout<<"creating forest"<<std::endl;
            m_Forest= new Forest(hp);
            std::cout<<"training forest"<<std::endl;
            m_Forest->train(m_TrainData.getData(),m_TrainData.getLabels(),m_weights);//	,m_weights);
            std::cout<<"done"<<std::endl;
            computeProbabilities();
        };


    };
    
    template<class ImageType>
    class SegmentationClassifierGradient: public SegmentationClassifier<ImageType> {
    public:
        typedef SegmentationClassifierGradient            Self;
        typedef SegmentationClassifier<ImageType> Superclass;
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
        int m_nIntensities;
        std::vector<float> m_probs;
        double m_meanIntens, m_meanGrad,m_varianceIntens,m_varianceGrad,m_covariance;

    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SegmentationClassifierGradient, Object);
        itkNewMacro(Self);

        SegmentationClassifierGradient(){
            m_data=matrix<float>(1,3);
            m_conf=matrix<float>(1,2);
            m_labelVector=std::vector<int>(1);
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

        virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
       
            int maxTrain=3000000;
            //maximal size
            long int nData=1;
            for (int d=0;d<ImageType::ImageDimension;++d)
                nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
            
            maxTrain=maxTrain>nData?nData:maxTrain;
            std::cout<<maxTrain<<" computed"<<std::endl;
            int nFeatures=2;
            matrix<float> data(maxTrain,nFeatures);
            std::cout<<maxTrain<<" matrix allocated"<<std::endl;
            std::vector<int> labelVector(maxTrain);
            typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
            //		typedef typename itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;
            IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
            ImageIterator.SetNumberOfSamples(maxTrain);
            int i=0;
            ImageIterator.GoToBegin();
            this->m_counts= std::vector<int>(2,0);
            //this->m_intensCounts= std::vector<int>(this->m_nIntensities,0);
            this->m_mean=0.0;
            this->m_variance=0.0;
            m_meanIntens=0; m_meanGrad=0;m_varianceIntens=0;m_varianceGrad=0;m_covariance=0;
            m_jointCounts=std::vector<int>(26*26,0);
            for (;!ImageIterator.IsAtEnd() ;
                 ++ImageIterator)
                {
                    typename ImageType::IndexType idx=ImageIterator.GetIndex();
                    int grad=gradient->GetPixel(idx);
                    int label=labels->GetPixel(idx)>0;
                    int intens=mapIntensity(ImageIterator.Get());
                    data(i,0)=intens;
                    data(i,1)=grad;
                    labelVector[i]=label;
                    this->m_counts[label]++;
                    i++;
                    m_meanIntens+=intens;
                    m_meanGrad+=grad;
                    m_varianceIntens+=intens*intens;
                    m_varianceGrad+=grad*grad;
                    m_covariance+=intens*grad;
                    //joint histogram, 10 bins
                    m_jointCounts[floor(intens/10) + 26*floor(grad/10)]++;
                }
            m_totalCount=i;
            m_meanIntens/=i;
            m_meanGrad/=i;
            cout<<i<<" "<< m_varianceIntens<<" "<<  m_meanIntens<<" "<<m_varianceIntens/i- m_meanIntens<<endl;
            m_varianceIntens= m_varianceIntens/i- m_meanIntens;
            m_varianceGrad=m_varianceGrad/i-m_meanGrad;
            m_covariance= m_covariance/i-m_meanGrad*m_meanIntens;
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
            std::cout<<"done adding data. "<<std::endl;
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
                    
                    labelVector[c]=0;
                }
            }
            this->m_Forest->eval(data,labelVector,false);
            matrix<float> conf = this->m_Forest->getConfidences();
            std::cout<<conf.size1()<<" "<<conf.size2()<<std::endl;
            c=0;
            //double min=9999;
            //double max=-1;
            std::vector<double> mins(2,99999);
            std::vector<double> maxs(2,-99999);
            std::cout<<m_varianceIntens<<" "<<m_varianceGrad<<" "<<m_covariance<<" "<<this->m_counts[0] <<" "<<  this->m_counts[1]<<endl;
            double det=m_varianceIntens*m_varianceGrad-m_covariance*m_covariance;
            std::cout<<det<<" "<<this->m_counts[0] +  this->m_counts[1]<<endl;
            //double norm=1.0/sqrt(2*3.14*det);
            for (int i=0;i<this->m_nIntensities;++i){
                for (int j=0;j<this->m_nIntensities;++j,++c){
                    for (int s=0;s<2;++s){
                        // p(s) = relative frequency
                        //double p_s=1.0*this->m_counts[s] / ( this->m_counts[0] +  this->m_counts[1]);
                        //p(x) = multivariate gaussian
                        //double bar_intens=i-m_meanIntens;
                        //double bar_grad=j-m_meanGrad;
                        //-1/2(i-mu_i g-mu_g)^T E^-1 (i-mu_i g-mu_g)
                        //double mahalanobis=-1.0/(2*det) *(bar_intens*(bar_intens*m_varianceGrad-bar_grad*m_covariance) + bar_grad*(bar_grad*m_varianceIntens-bar_intens*m_covariance));
                        //double p_x=norm*exp(mahalanobis);
                        
                        //double p_x2= 1.0*m_jointCounts[floor(i/10) + 26*floor(j/10)]/m_totalCount;
                        //cout<<floor(i/10)<<" "<<26*floor(j/10)<<" "<<floor(i/10) + 26*floor(j/10)<<" "<<m_jointCounts[floor(i/10) + 26*floor(j/10)]<<" "<<p_x2<<endl;
                        double p=conf(c,s) ;/// p_s  * p_x2 ; 

#if 0
                        int bone=(300+1000)*255.0/2000;
                        int tissue=(-500+1000)*255.0/2000;
                        double segmentationProb=1;
                        if (s>0) {
                            if (i < tissue)
                                segmentationProb =fabs(i-tissue);
                            else if (i < bone) 
                                segmentationProb = 0.69; //log (0.5);
                            else
                                segmentationProb = 0.00000000001;
                        }else{
                            if ((i >  bone)  && j>128)
                                segmentationProb = fabs(i-bone);
                            else if (i >tissue)
                                segmentationProb =0.69 ;
                            else
                                segmentationProb = 0.00000000001;
                        }
                        segmentationProb=exp(-segmentationProb);
                        if (s){
                            p=segmentationProb>p?segmentationProb:p;
                        }else{
                            p=segmentationProb<p?segmentationProb:p;
                        }
                            
            

#endif



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
        virtual double px_l(float intensity,int label, int gradient){
            int intens=mapIntensity(intensity);
            return this->m_probs[(label>0)*this->m_nIntensities*this->m_nIntensities+intens*this->m_nIntensities+gradient];
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
                //                std::cout<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
                result0->SetPixel(it.GetIndex(),(PixelType)(multiplier*prob0));
                result1->SetPixel(it.GetIndex(),(PixelType)(multiplier*prob1));
            }
            if (false){
                if (ImageType::ImageDimension ==2 ){
                    ImageUtils<ImageType>::writeImage("p0-rfGradient.png",result0);
                    ImageUtils<ImageType>::writeImage("p1-rfGradient.png",result1);
                }else{
                    ImageUtils<ImageType>::writeImage("p0-rfGradient.nii",result0);
                    ImageUtils<ImageType>::writeImage("p1-rfGradient.nii",result1);
     
                }
            }
        }

        virtual void loadProbs(string filename){

        }
        virtual void saveProbs(string filename){

        }

        virtual void train(){
            std::cout<<"reading config"<<std::endl;
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

            std::cout<<"creating forest"<<std::endl;
            m_Forest= new Forest(hp);
            std::cout<<"training forest"<<std::endl;
            m_Forest->train(m_TrainData.getData(),m_TrainData.getLabels(),m_weights);
            std::cout<<"done"<<std::endl;
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
        std::vector<double> m_meanIntens, m_meanGrad,m_varianceIntens,m_varianceGrad,m_covariance;

    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(SegmentationGaussianClassifierGradient, Object);
        itkNewMacro(Self);

        SegmentationGaussianClassifierGradient(){
            m_data=matrix<float>(1,3);
            m_conf=matrix<float>(1,2);
            m_labelVector=std::vector<int>(1);
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

        virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
       
            int maxTrain=1000000;
            //maximal size
            long int nData=1;
            for (int d=0;d<ImageType::ImageDimension;++d)
                nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
            
            maxTrain=maxTrain>nData?nData:maxTrain;
            std::cout<<maxTrain<<" computed"<<std::endl;
            int nFeatures=2;
            typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
            //		typedef typename itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;
            IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
            ImageIterator.SetNumberOfSamples(maxTrain);
            int i=0;
            ImageIterator.GoToBegin();
            this->m_counts= std::vector<int>(2,0);
            //this->m_intensCounts= std::vector<int>(this->m_nIntensities,0);
            m_meanIntens=std::vector<double>(2,0.0); m_meanGrad=std::vector<double>(2,0.0);m_varianceIntens=std::vector<double>(2,0.0);m_varianceGrad=std::vector<double>(2,0.0);m_covariance=std::vector<double>(2,0.0);

            for (;!ImageIterator.IsAtEnd() ;
                 ++ImageIterator)
                {
                    typename ImageType::IndexType idx=ImageIterator.GetIndex();
                    int grad=gradient->GetPixel(idx);
                    int label=labels->GetPixel(idx)>0;
                    int intens=mapIntensity(ImageIterator.Get());
                    this->m_counts[label]++;
                    i++;
                    m_meanIntens[label]+=intens;
                    m_meanGrad[label]+=grad;
                    m_varianceIntens[label]+=intens*intens;
                    m_varianceGrad[label]+=grad*grad;
                    m_covariance[label]+=intens*grad;
                }

            for (int s=0;s<2;++s){
                m_meanIntens[s]/=i;
                m_meanGrad[s]/=i;
                m_varianceIntens[s]= m_varianceIntens[s]/i- m_meanIntens[s];
                m_varianceGrad[s]=m_varianceGrad[s]/i-m_meanGrad[s];
                m_covariance[s]= m_covariance[s]/i-m_meanGrad[s]*m_meanIntens[s];
            }
        };

        virtual void computeProbabilities(){

            double min=9999;
            double max=-1;
            std::vector<double> mins(2,99999);
            std::vector<double> maxs(2,-99999);
            std::vector<double> det(2,0);
            for (int s=0;s<2;++s){
                det[s]=m_varianceIntens[s]*m_varianceGrad[s]-m_covariance[s]*m_covariance[s];
            }
        
            for (int i=0;i<this->m_nIntensities;++i){
                for (int j=0;j<this->m_nIntensities;++j){
                    double sum=0.0;
                    for (int s=0;s<2;++s){
                        double norm=1.0/sqrt(2*3.14*det[s]);
                        // p(s) = relative frequency
                        double p_s=1.0*this->m_counts[s] / ( this->m_counts[0] +  this->m_counts[1]);
                        //p(x) = multivariate gaussian
                        double bar_intens=i-m_meanIntens[s];
                        double bar_grad=j-m_meanGrad[s];
                        //-1/2(i-mu_i g-mu_g)^T E^-1 (i-mu_i g-mu_g)
                        double mahalanobis=-1.0/(2*det[s]) *(bar_intens*(bar_intens*m_varianceGrad[s]-bar_grad*m_covariance[s]) + bar_grad*(bar_grad*m_varianceIntens[s]-bar_intens*m_covariance[s]));
                        double p=norm*exp(mahalanobis);
                        sum+=p;
                        this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]=p;
                        if (p<mins[0]) mins[0]=p;
                        if (p>maxs[0]) maxs[0]=p;
                    }
                    for (int s=0;s<2;++s){
                        double p=( this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]);
                        double p_s=1.0*this->m_counts[s] / ( this->m_counts[0] +  this->m_counts[1]);

                        p=p*p_s/sum;
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
                    
                        //cout<<i<<" "<<j<<" "<<s<<" "<<this->m_probs[s*this->m_nIntensities*this->m_nIntensities+i*this->m_nIntensities+j]<<endl;
                    }
                }
            }
#endif
        }
        
        inline virtual int mapIntensity(float intensity){
            return intensity;
        }
        virtual double px_l(float intensity,int label, int gradient){
            int intens=mapIntensity(intensity);
            return this->m_probs[(label>0)*this->m_nIntensities*this->m_nIntensities+intens*this->m_nIntensities+gradient];
        }
        virtual void evalImage(ImageConstPointerType im, ImageConstPointerType gradient){
            ImagePointerType result0=ImageUtils<ImageType>::createEmpty(im);
            ImagePointerType result1=ImageUtils<ImageType>::createEmpty(im);
            typename itk::ImageRegionConstIterator<ImageType> it(im,im->GetLargestPossibleRegion());
            typename itk::ImageRegionConstIterator<ImageType> itGrad(gradient,gradient->GetLargestPossibleRegion());
            for (it.GoToBegin();!it.IsAtEnd(); ++it,++itGrad){
                PixelType val=it.Get();
                PixelType grad=itGrad.Get();
                double prob0=px_l(val,grad,0);
                double prob1=px_l(val,grad,1);
                //                std::cout<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
                result0->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob0);
                result1->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob1);
            }
            if (false){
                if (ImageType::ImageDimension==2){
                    ImageUtils<ImageType>::writeImage("p0-gaussGradient.png",result0);
                    ImageUtils<ImageType>::writeImage("p1-gaussGradient.png",result1);
                }else{
                    ImageUtils<ImageType>::writeImage("p0-gaussGradient.nii",result0);
                    ImageUtils<ImageType>::writeImage("p1-gaussGradient.nii",result1);
                }
            }
        }

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
                //                std::cout<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
                result0->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob0);
                result1->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob1);
            }
            if (false){
                if (ImageType::ImageDimension==2){
                    ImageUtils<ImageType>::writeImage("p0-marcel.png",result0);
                    ImageUtils<ImageType>::writeImage("p1-marcel.png",result1);
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
                //                std::cout<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
                result0->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob0);
                result1->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob1);
            }
            if (false){
                if (ImageType::ImageDimension==2){
                    ImageUtils<ImageType>::writeImage("p0-marcel.png",result0);
                    ImageUtils<ImageType>::writeImage("p1-marcel.png",result1);
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
                //                std::cout<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
                result0->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob0);
                result1->SetPixel(it.GetIndex(),(PixelType)std::numeric_limits<PixelType>::max()*prob1);
            }
            if (false){
                if (ImageType::ImageDimension==2){
                    ImageUtils<ImageType>::writeImage("p0-marcel.png",result0);
                    ImageUtils<ImageType>::writeImage("p1-marcel.png",result1);
                }else{
                    ImageUtils<ImageType>::writeImage("p0-marcel.nii",result0);
                    ImageUtils<ImageType>::writeImage("p1-marcel.nii",result1);
                }
            
            }
        }
    };
  
        template<class ImageType>
        class SmoothnessClassifierGradient: public SegmentationClassifier<ImageType> {
        public:
            typedef SmoothnessClassifierGradient            Self;
            typedef SegmentationClassifier<ImageType> Superclass;
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
            int m_nIntensities;
            std::vector<float> m_probs;
            double m_meanIntens, m_meanGrad,m_varianceIntens,m_varianceGrad,m_covariance;
            double m_weight;
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
                std::cout<<maxTrain<<" computed"<<std::endl;
                int nFeatures=2;
                matrix<float> data(maxTrain,nFeatures);
                std::cout<<maxTrain<<" matrix allocated"<<std::endl;
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
                                //    cout<<d<<" "<<i<<" "<<idx+off<<" "<<intensities->GetLargestPossibleRegion().GetSize()<<endl;
                                int grad1=gradient->GetPixel(idx);
                                int label1=labels->GetPixel(idx)>0;
                                int intens1=mapIntensity(ImageIterator.Get());
                                int grad2=gradient->GetPixel(idx+off);
                                int label2=labels->GetPixel(idx+off)>0;
                                int intens2=mapIntensity(intensities->GetPixel(idx+off));
                                data(i,0)=fabs(intens1-intens2);
                                data(i,1)=fabs(grad1-grad2);
                                labelVector[i]=label1!=label2;
                                this->m_counts[label1!=label2]++;
                                i++;                
                            }
                        }
                    
                    }
                cout<<"finished adding data" <<endl;
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
                std::cout<<"done adding data. "<<std::endl;
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
                std::cout<<"evaluating forest "<<std::endl;
                this->m_Forest->eval(data,labelVector,false);
                matrix<float> conf = this->m_Forest->getConfidences();
                std::cout<<conf.size1()<<" "<<conf.size2()<<std::endl;
                c=0;
                for (int i=0;i<this->m_nIntensities;++i){
                    for (int j=0;j<this->m_nIntensities;++j,++c){
                        for (int s=0;s<2;++s){
                            // p(s) = relative frequency
                            //double p_s=1.0*this->m_counts[s] / ( this->m_counts[0] +  this->m_counts[1]);
                            double p=conf(c,s) ;/// p_s  * p_x2 ; 
                            //std::cout<<p<<std::endl;
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
                //            cout<<intensityDiff<<" "<<label<<" "<<gradientDiff<<endl;
                label=(label==label2);
                intensityDiff=fabs(intensityDiff);
                gradientDiff=fabs(gradientDiff);
                double prob=this->m_probs[(label>0)*this->m_nIntensities*this->m_nIntensities+intensityDiff*this->m_nIntensities+gradientDiff];
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
                    //                std::cout<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
                    result0->SetPixel(it.GetIndex(),(PixelType)(multiplier*prob0));
                    result1->SetPixel(it.GetIndex(),(PixelType)(multiplier*prob1));
                }
                if(false){
                    if (ImageType::ImageDimension ==2 ){
                        ImageUtils<ImageType>::writeImage("p0-rfGradient.png",result0);
                        ImageUtils<ImageType>::writeImage("p1-rfGradient.png",result1);
                    }else{
                        ImageUtils<ImageType>::writeImage("p0-rfGradient.nii",result0);
                        ImageUtils<ImageType>::writeImage("p1-rfGradient.nii",result1);
     
                    }}
            
            }

            virtual void loadProbs(string filename){
                
                ifstream myFile (filename.c_str(), ios::in | ios::binary);
                if (myFile){
                    myFile.read((char*)(&this->m_probs[0]),2*this->m_nIntensities*this->m_nIntensities*sizeof(float) );
                        std::cout<<" read m_segmentationPosteriorProbs from disk"<<std::endl;
                }else{
                        std::cout<<" error reading m_segmentationPosteriorProbs"<<std::endl;
                        exit(0);

                }

            }
            virtual void saveProbs(string filename){
                ofstream myFile (filename.c_str(), ios::out | ios::binary);
                myFile.write ((char*)(&this->m_probs[0]),2*this->m_nIntensities*this->m_nIntensities*sizeof(float) );
            }

            virtual void train(){
                std::cout<<"reading config"<<std::endl;
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

                std::cout<<"creating forest"<<std::endl;
                m_Forest= new Forest(hp);
                std::cout<<"training forest"<<std::endl;
                m_Forest->train(m_TrainData.getData(),m_TrainData.getLabels(),m_weights);
                std::cout<<"done"<<std::endl;
                computeProbabilities();
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
                //            cout<<intensityDiff<<" "<<label<<" "<<gradientDiff<<endl;
                //double prob=this->m_probs[(label>0)*this->m_nIntensities*this->m_nIntensities+intensityDiff*this->m_nIntensities+gradientDiff];
                intensityDiff=fabs(intensityDiff);
                gradientDiff=fabs(gradientDiff);
                double prob;
                if (gradientDiff<0){
                    //cout<<gradientDiff<<endl;
                    prob=0;
                }else{
                    //intensityDiff*=intensityDiff;
                    gradientDiff=(gradientDiff*gradientDiff);
                    prob=0.95*exp(-this->m_weight*0.00005*gradientDiff);
                    //cout<<gradientDiff<<" "<<prob<<endl;
                }
                if (label){
                    prob=1-prob;
                }
                return prob;
            }
     
     

            virtual void train(){
          
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
                this->m_probs= std::vector<float> (3*2*2*this->m_nIntensities*this->m_nIntensities,0);
            }
          
          
            virtual void setData(ImageConstPointerType intensities, ImageConstPointerType labels, ImageConstPointerType gradient){
       
                int maxTrain=3000000;
                //maximal size
                long int nData=1;
                for (int d=0;d<ImageType::ImageDimension;++d)
                    nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
                nData*=ImageType::ImageDimension;

                maxTrain=maxTrain>nData?nData:maxTrain;
                std::cout<<maxTrain<<" computed"<<std::endl;
                int nFeatures=2;
                matrix<float> data(maxTrain,nFeatures);
                std::cout<<maxTrain<<" matrix allocated"<<std::endl;
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
                                //    cout<<d<<" "<<i<<" "<<idx+off<<" "<<intensities->GetLargestPossibleRegion().GetSize()<<endl;
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
                cout<<"finished adding data" <<endl;
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
                std::cout<<"done adding data. "<<std::endl;
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
                std::cout<<"evaluating forest "<<std::endl;
                this->m_Forest->eval(data,labelVector,false);
                matrix<float> conf = this->m_Forest->getConfidences();
                std::cout<<conf.size1()<<" "<<conf.size2()<<std::endl;
                c=0;
                for (int i=0;i<2*this->m_nIntensities;++i){
                    for (int j=0;j<2*this->m_nIntensities;++j,++c){
                        for (int s=0;s<3;++s){
                            // p(s) = relative frequency
                            //double p_s=1.0*this->m_counts[s] / ( this->m_counts[0] +  this->m_counts[1]);
                            double p=conf(c,s) ;/// p_s  * p_x2 ; 
                            //std::cout<<p<<std::endl;
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
                //            cout<<intensityDiff<<" "<<label<<" "<<gradientDiff<<endl;
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
                    //                std::cout<<prob0<<" "<<prob1<<" "<<(PixelType)std::numeric_limits<PixelType>::max()*prob0<<std::endl;
                    result0->SetPixel(it.GetIndex(),(PixelType)(multiplier*prob0));
                    result1->SetPixel(it.GetIndex(),(PixelType)(multiplier*prob1));
                }
                if(false){
                    if (ImageType::ImageDimension ==2 ){
                        ImageUtils<ImageType>::writeImage("p0-rfGradient.png",result0);
                        ImageUtils<ImageType>::writeImage("p1-rfGradient.png",result1);
                    }else{
                        ImageUtils<ImageType>::writeImage("p0-rfGradient.nii",result0);
                        ImageUtils<ImageType>::writeImage("p1-rfGradient.nii",result1);
     
                    }}
            
            }

            virtual void loadProbs(string filename){
                
                ifstream myFile (filename.c_str(), ios::in | ios::binary);
                if (myFile){
                    myFile.read((char*)(&this->m_probs[0]),3*2*2*this->m_nIntensities*this->m_nIntensities*sizeof(float) );
                        std::cout<<" read m_segmentationPosteriorProbs from disk"<<std::endl;
                }else{
                        std::cout<<" error reading m_segmentationPosteriorProbs"<<std::endl;
                        exit(0);

                }

            }
            virtual void saveProbs(string filename){
                ofstream myFile (filename.c_str(), ios::out | ios::binary);
                myFile.write ((char*)(&this->m_probs[0]),3*2*2*this->m_nIntensities*this->m_nIntensities*sizeof(float) );
            }
            virtual void train(){
            std::cout<<"reading config"<<std::endl;
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

            std::cout<<"creating forest"<<std::endl;
            this->m_Forest= new Forest(hp);
            std::cout<<"training forest"<<std::endl;
            this->m_Forest->train(this->m_TrainData.getData(),this->m_TrainData.getLabels(),this->m_weights);
            std::cout<<"done"<<std::endl;
            computeProbabilities();
        };
        };
    }//namespace
#endif /* CLASSIFIER_H_ */
