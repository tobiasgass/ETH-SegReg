/*
 * Classifier.h
 *
 *  Created on: Feb 14, 2011
 *      Author: gasst
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
#include <sstream>

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "ImageUtils.h"
#include "unsupervised.h"

using namespace std;
using namespace boost::numeric::ublas;
using namespace libconfig;

namespace itk{
    template<class ImageType>
    class SegmentationRandomForestClassifier: public itk::Object{
    protected:
        FileData m_data;
        Forest * m_Forest;
        int m_nData;
        int m_nSegmentationLabels;
    public:
        typedef SegmentationRandomForestClassifier            Self;
        typedef itk::Object Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
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
        itkTypeMacro(SegmentationRandomForestClassifier, Object);
       itkNewMacro(Self);

        SegmentationRandomForestClassifier(){
            LOGV(5)<<"Initializing intensity based segmentation classifier" << endl;
          
         
        };
     
        virtual void setNSegmentationLabels(int n){
            m_nSegmentationLabels=n;
        }
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
                    //LOGV(10)<<i<<" "<<label<<" "<<nFeatures<<endl;
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
        };

        virtual std::vector<FloatImagePointerType> evalImage(std::vector<ImageConstPointerType> inputImage){
            LOGV(5)<<"Evaluating intensity based segmentation classifier" << endl;

            setData(inputImage);
            m_Forest->eval(m_data.getData(),m_data.getLabels(),false);
            matrix<float> conf = m_Forest->getConfidences();
            std::vector<FloatImagePointerType> result(m_nSegmentationLabels);
            for ( int s=0;s<m_nSegmentationLabels;++s){
                result[s]=FilterUtils<ImageType,FloatImageType>::createEmpty(inputImage[0]);
            }
           
          
            std::vector<FloatIteratorType> iterators;
            for ( int s=0;s<m_nSegmentationLabels;++s){
                iterators.push_back(FloatIteratorType(result[s],result[s]->GetLargestPossibleRegion()));
                iterators[s].GoToBegin();
            }
            
            for (int i=0;!iterators[0].IsAtEnd() ; ++i){
                 for ( int s=0;s<m_nSegmentationLabels;++s){
                     iterators[s].Set((conf(i,s)));
                     ++iterators[s];
                 }
            }
            std::string suff;
            if (ImageType::ImageDimension==2){
                suff=".png";
            }
            if (ImageType::ImageDimension==3){
                suff=".nii";
            }
            for ( int s=0;s<m_nSegmentationLabels;++s){
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
            hp.numClasses = m_nSegmentationLabels;//configFile.lookup("Data.numClasses");
            LOGV(9)<<VAR(hp.numLabeled)<<" "<<VAR(hp.numClasses)<<endl;
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
            LOGV(9)<<VAR(hp.numTrees)<<" "<<VAR(hp.maxTreeDepth)<<endl;

            LOG<<"creating forest"<<std::endl;
            m_Forest= new Forest(hp);
            LOG<<"training forest"<<std::endl;
            std::vector<double> weights(m_data.getLabels().size(),1.0);
            m_Forest->train(m_data.getData(),m_data.getLabels(),weights);
            LOG<<"done"<<std::endl;
        };



    };//class
     template<class ImageType>
    class SegmentationGMMClassifier: public itk::Object{
    protected:
         std::vector<NEWMAT::Matrix> m_observations;
         int m_nData;
         int m_nSegmentationLabels;
         std::vector<unsupervised> m_GMMs;

    public:
        typedef SegmentationGMMClassifier            Self;
        typedef itk::Object Superclass;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
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
        itkTypeMacro(SegmentationGMMClassifier, Object);
        itkNewMacro(Self);

        SegmentationGMMClassifier(){
            LOGV(5)<<"Initializing intensity based segmentation classifier" << endl;
        };
     
        virtual void setNSegmentationLabels(int n){
            m_nSegmentationLabels=n;
            m_GMMs=std::vector<unsupervised>(n);
        }
        virtual void freeMem(){
            m_observations=std::vector<NEWMAT::Matrix>();
        }
        virtual void save(string filename){
          
        }
        virtual void load(string filename){
           
        }
        virtual void setData(std::vector<ImageConstPointerType> inputImage, ImageConstPointerType labels=NULL){
            LOGV(5)<<"Setting up data for intensity based segmentation classifier" << endl;
            unsigned int nFeatures=inputImage.size();
            long int nData=1;
            for (int d=0;d<ImageType::ImageDimension;++d)
                nData*=inputImage[0]->GetLargestPossibleRegion().GetSize()[d];
            m_observations=std::vector<NEWMAT::Matrix>();
            long int maxTrain=100000;//std::numeric_limits<long int>::max();
            maxTrain=maxTrain>nData?nData:maxTrain;

            if (labels){
                std::vector<int> counts(m_nSegmentationLabels,0);
                ConstImageIteratorType lIt(labels,labels->GetLargestPossibleRegion());
                for (lIt.GoToBegin();!lIt.IsAtEnd();++lIt){counts[lIt.Get()>0]++;}
                for (int l=0;l<m_nSegmentationLabels;++l){
                    //if (counts[l]<maxTrain) maxTrain=counts[l];
                }
                for (int l=0;l<m_nSegmentationLabels;++l){
                    LOGV(7)<<VAR(l)<<" "<<VAR(counts[l])<<" "<<VAR(nFeatures)<<endl;
                    //m_observations.push_back(NEWMAT::Matrix(nFeatures,counts[l]));
                    m_observations.push_back(NEWMAT::Matrix(nFeatures,min((long int)counts[l],maxTrain)));
                }
            }else{
                m_observations.push_back(NEWMAT::Matrix(nFeatures,nData));
            }

            //maximal size
         
            LOGV(5)<<maxTrain<<" computed"<<std::endl;
            
            std::vector<ConstImageIteratorType> iterators;
            for (unsigned int s=0;s<nFeatures;++s){
                iterators.push_back(ConstImageIteratorType(inputImage[s],inputImage[s]->GetLargestPossibleRegion()));
                iterators[s].GoToBegin();
            }
            std::vector<int> counts(m_nSegmentationLabels,0);
            int i=0;
            for (;!iterators[0].IsAtEnd() ; ++i)
                {
                    int label=0;
                    if (labels)
                        label= (labels->GetPixel(iterators[0].GetIndex()) > 0 );
                    
                    // LOGV(10)<<i<<" "<<VAR(label)<<" "<<VAR(counts[label])<<" "<<nFeatures<<endl;
                    if ( counts[label] <maxTrain){
                        for (unsigned int f=0;f<nFeatures;++f){
                            int intens=(iterators[f].Get());
                            m_observations[label].element(f,counts[label])=intens;
                            ++iterators[f];
                        }
                        counts[label]++;
                    }else{
                        for (unsigned int f=0;f<nFeatures;++f){
                            ++iterators[f];
                        }
                     }
                }
            m_nData=i;
            LOG<<"done adding data. "<<std::endl;
            LOG<<"stored "<<m_nData<<" samples "<<std::endl;
            for ( int s=0;s<m_nSegmentationLabels;++s){
                LOG<<VAR(counts[s])<<endl;
            }

        };

   

        virtual void train(){
            for ( int s=0;s<m_nSegmentationLabels;++s){
                LOGV(1)<<"Training GMM for label :"<<s<<endl;
                m_GMMs[s].estimate(2,m_observations[s]);
                m_GMMs[s].display();
            }
        };

        virtual std::vector<FloatImagePointerType> evalImage(std::vector<ImageConstPointerType> inputImage){
            LOGV(5)<<"Evaluating intensity based segmentation classifier" << endl;

            std::vector<FloatImagePointerType> result(m_nSegmentationLabels);
            for ( int s=0;s<m_nSegmentationLabels;++s){
                result[s]=FilterUtils<ImageType,FloatImageType>::createEmpty(inputImage[0]);
            }
           
            unsigned int nFeatures=inputImage.size();
            std::vector<FloatIteratorType> resultIterators;
            for ( int s=0;s<m_nSegmentationLabels;++s){
                resultIterators.push_back(FloatIteratorType(result[s],result[s]->GetLargestPossibleRegion()));
                resultIterators[s].GoToBegin();
            }
            std::vector<ConstImageIteratorType> iterators;
            for (unsigned int f=0;f<nFeatures;++f){
                iterators.push_back(ConstImageIteratorType(inputImage[f],inputImage[f]->GetLargestPossibleRegion()));
                iterators[f].GoToBegin();
            }
            
            for (int i=0;!resultIterators[0].IsAtEnd() ; ++i){
                NEWMAT::ColumnVector c(nFeatures);
                for (unsigned int f=0;f<nFeatures;++f){
                    c.element(f)=iterators[f].Get();
                    ++iterators[f];
                }
                for ( int s=0;s<m_nSegmentationLabels;++s){
                    double p=m_GMMs[s].likelihood(c);
                    p=min(1.0,p);
                    resultIterators[s].Set(-log(p));
                    ++resultIterators[s];
                }
            }
            std::string suff;
            if (ImageType::ImageDimension==2){
                suff=".png";
                for ( int s=0;s<m_nSegmentationLabels;++s){
                    ostringstream probabilityfilename;
                    probabilityfilename<<"prob-gauss-c"<<s<<suff;
                    ImageUtils<ImageType>::writeImage(probabilityfilename.str().c_str(),FilterUtils<FloatImageType,ImageType>::normalize(result[s]));
                      //ImageUtils<ImageType>::writeImage(probabilityfilename.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(result[s],255.0*255.0)));
                }
            }
            if (ImageType::ImageDimension==3){
                suff=".nii";
                for ( int s=0;s<m_nSegmentationLabels;++s){
                    ostringstream probabilityfilename;
                    probabilityfilename<<"prob-gauss-c"<<s<<suff;
                    
                    ImageUtils<FloatImageType>::writeImage(probabilityfilename.str().c_str(),result[s]);
                }

            }
            return result;
        }

    };//class
  
}//namespace
