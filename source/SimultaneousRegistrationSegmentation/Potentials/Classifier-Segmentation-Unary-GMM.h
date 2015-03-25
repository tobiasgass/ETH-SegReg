/**
 * @file   Classifier-Segmentation-Unary-GMM.h
 * @author gasst <gasst@ETHSEGREG>
 * @date   Thu Mar  5 13:14:14 2015
 * 
 * @brief  Wrappers to use gaussian mixture model (GMM) estimators to calculate unary segmentation potentials
 * 
 * 
 */
#pragma once

#include "Log.h"
#include <vector>

#include "unsupervised.h"

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
#include "FilterUtils.hpp"

using namespace boost::numeric::ublas;
namespace SRS{

    template<class ImageType>
    class ClassifierSegmentationUnaryGMM: public itk::Object{
    protected:
        std::vector<NEWMAT::Matrix> m_observations;
        int m_nData;
        int m_nSegmentationLabels;
        std::vector<unsupervised> m_GMMs;
        std::vector<bool> m_trainedGMMs;
    public:
        typedef ClassifierSegmentationUnaryGMM            Self;
        typedef itk::Object Superclass;
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
        //typedef VariableLengthVector< unsigned char > RGBPixelType;
        //typedef RGBPixel< unsigned char > RGBPixelType;
        typedef itk::Vector< unsigned char, 3 > RGBPixelType;
        static const unsigned short D=ImageType::ImageDimension;
        typedef typename itk::Image<RGBPixelType,D > RGBImageType;
        typedef typename RGBImageType::Pointer RGBImagePointerType;
        typedef typename itk::ImageRegionIteratorWithIndex< RGBImageType > RGBIteratorType;

    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(ClassifierSegmentationUnaryGMM, Object);
        itkNewMacro(Self);

        ClassifierSegmentationUnaryGMM(){
            LOGV(5)<<"Initializing intensity based segmentation classifier" << endl;
        };
     
        virtual void setNSegmentationLabels(int n){
            m_nSegmentationLabels=n;
            m_GMMs=std::vector<unsupervised>(n);
            m_trainedGMMs=std::vector<bool>(n,false);
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
                    LOGV(7)<<VAR(l)<<" "<<VAR(counts[l])<<" "<<VAR(nFeatures)<<std::endl;
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
                    
                    // LOGV(10)<<i<<" "<<VAR(label)<<" "<<VAR(counts[label])<<" "<<nFeatures<<std::endl;
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
                LOG<<VAR(counts[s])<<std::endl;
            }

        };

        
   

        virtual void train(){
            for ( int s=0;s<m_nSegmentationLabels;++s){
                if (m_observations[s].size()>0){
                    LOGV(1)<<"Training GMM for label :"<<s<<std::endl;
                    m_GMMs[s].estimate(4,m_observations[s]);
                    LOGI(4,m_GMMs[s].display());
                    m_trainedGMMs[s]=true;
                }else{
                    LOGV(1)<<"No training data  for label :"<<s<<std::endl;
                }
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
                    
                    double p=0;
                    if (this->m_trainedGMMs[s]){
                        p=m_GMMs[s].likelihood(c);
                        p=min(1.0,p);
                        p=max(std::numeric_limits<double>::epsilon(),p);
                    }
                    //resultIterators[s].Set(-log(p));
                    resultIterators[s].Set((p));
                    ++resultIterators[s];
                }
            }
            std::string suff;
            if (true){
                if (ImageType::ImageDimension==2){
                    suff=".nii";
                    for ( int s=0;s<m_nSegmentationLabels;++s){
                        ostringstream probabilityfilename;
                        probabilityfilename<<"prob-gauss-c"<<s<<suff;
                        LOGI(10,ImageUtils<ImageType>::writeImage(probabilityfilename.str().c_str(),FilterUtils<FloatImageType,ImageType>::normalize(result[s])));
                        //ImageUtils<ImageType>::writeImage(probabilityfilename.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(result[s],255.0*255.0)));
                    }
                }
                if (ImageType::ImageDimension==3){
                    suff=".nii";
                    for ( int s=0;s<m_nSegmentationLabels;++s){
                        ostringstream probabilityfilename;
                        probabilityfilename<<"prob-gauss-c"<<s<<suff;
                    
                        LOGI(8,ImageUtils<FloatImageType>::writeImage(probabilityfilename.str().c_str(),result[s]));
                    }

                }
            }
            return result;
        }


        virtual std::vector<FloatImagePointerType> evalImage(RGBImagePointerType inputImage){
            LOGV(5)<<"Evaluating intensity based segmentation classifier" << endl;

             ImagePointerType comp1=ImageUtils<ImageType>::createEmpty(inputImage->GetRequestedRegion(),
                                                                  inputImage->GetOrigin(),
                                                                  inputImage->GetSpacing(),
                                                                  inputImage->GetDirection());

    
             typedef itk::ImageRegionIterator<RGBImageType> DeformationIteratorType;
             DeformationIteratorType defIt(inputImage,inputImage->GetLargestPossibleRegion());
             typedef itk::ImageRegionIterator<ImageType> FloatImageIteratorType;
             FloatImageIteratorType resultIt(comp1,comp1->GetLargestPossibleRegion());
             
             for (defIt.GoToBegin(),resultIt.GoToBegin();!defIt.IsAtEnd();++defIt,++resultIt){
                 resultIt.Set(defIt.Get()[0]);
             }
             
             ImageUtils<ImageType>::writeImage("comp1.nii",comp1);
             for (defIt.GoToBegin(),resultIt.GoToBegin();!defIt.IsAtEnd();++defIt,++resultIt){
                 resultIt.Set(defIt.Get()[1]);
             }
             
             ImageUtils<ImageType>::writeImage("comp2.nii",comp1);
             for (defIt.GoToBegin(),resultIt.GoToBegin();!defIt.IsAtEnd();++defIt,++resultIt){
                 resultIt.Set(defIt.Get()[2]);
             }
             
             ImageUtils<ImageType>::writeImage("comp3.nii",comp1);
             
            std::vector<FloatImagePointerType> result(m_nSegmentationLabels);
            for ( int s=0;s<m_nSegmentationLabels;++s){
                result[s]=ImageUtils<FloatImageType>::createEmpty(inputImage->GetRequestedRegion(),
                                                                  inputImage->GetOrigin(),
                                                                  inputImage->GetSpacing(),
                                                                  inputImage->GetDirection());
            }
            typename ImageType::IndexType idx;idx.Fill(0);
            unsigned int nFeatures=3;//inputImage->GetPixel(idx).GetSize();
            
            std::vector<FloatIteratorType> resultIterators;
            for ( int s=0;s<m_nSegmentationLabels;++s){
                resultIterators.push_back(FloatIteratorType(result[s],result[s]->GetLargestPossibleRegion()));
                resultIterators[s].GoToBegin();
            }
           
            RGBIteratorType iterator(inputImage,inputImage->GetLargestPossibleRegion());
            iterator.GoToBegin();
            
            for (int i=0;!resultIterators[0].IsAtEnd() ; ++i,++iterator){
                NEWMAT::ColumnVector c(nFeatures);
                RGBPixelType px=iterator.Get();

                for (unsigned int f=0;f<nFeatures;++f){
                    c.element(f)=px[f];
                }
                for ( int s=0;s<m_nSegmentationLabels;++s){
                    
                    double p=0;
                    if (this->m_trainedGMMs[s]){
                        p=m_GMMs[s].likelihood(c);
                        p=min(1.0,p);
                        p=max(std::numeric_limits<double>::epsilon(),p);
                    }
                    //resultIterators[s].Set(-log(p));
                    resultIterators[s].Set((p));
                    ++resultIterators[s];
                }
            }
            std::string suff;
            if (true){
                if (false && ImageType::ImageDimension==2){
                    suff=".nii";
                    for ( int s=0;s<m_nSegmentationLabels;++s){
                        ostringstream probabilityfilename;
                        probabilityfilename<<"prob-gauss-c"<<s<<suff;
                        LOGI(10,ImageUtils<ImageType>::writeImage(probabilityfilename.str().c_str(),FilterUtils<FloatImageType,ImageType>::normalize(result[s])));
                        //ImageUtils<ImageType>::writeImage(probabilityfilename.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(result[s],255.0*255.0)));
                    }
                }else{ //                if (ImageType::ImageDimension==3){
                    suff=".nii";
                    for ( int s=0;s<m_nSegmentationLabels;++s){
                        ostringstream probabilityfilename;
                        probabilityfilename<<"prob-gauss-c"<<s<<suff;
                    
                        LOGI(8,ImageUtils<FloatImageType>::writeImage(probabilityfilename.str().c_str(),result[s]));
                    }

                }
            }
            return result;
        }

    };//class

  ///\brief Segmentation unary classifier with support for more labels (>2) 
    template<class ImageType>
    class ClassifierSegmentationUnaryGMMMultilabel: public ClassifierSegmentationUnaryGMM<ImageType>{
    public:
        typedef ClassifierSegmentationUnaryGMMMultilabel            Self;
        typedef ClassifierSegmentationUnaryGMM<ImageType> Superclass;
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

        typedef typename Superclass::RGBPixelType RGBPixelType;
        static const unsigned short D=ImageType::ImageDimension;
        typedef typename itk::Image<RGBPixelType,D > RGBImageType;
        typedef typename RGBImageType::Pointer RGBImagePointerType;
        typedef typename itk::ImageRegionIteratorWithIndex< RGBImageType > RGBIteratorType;
        
    public:
        /** Standard part of every itk Object. */
        itkTypeMacro(ClassifierSegmentationUnaryGMMMultilabel, Object);
        itkNewMacro(Self);
        virtual void setData(RGBImagePointerType inputImage, ImageConstPointerType labels=NULL){
            LOGV(5)<<"Setting up data for intensity based segmentation classifier" << endl;
            typename ImageType::IndexType idx;idx.Fill(0);
            unsigned int nFeatures=3;//inputImage->GetPixel(idx).GetSize();
            long int nData=1;
            for (int d=0;d<ImageType::ImageDimension;++d)
                nData*=inputImage->GetLargestPossibleRegion().GetSize()[d];
            this->m_observations=std::vector<NEWMAT::Matrix>();
            long int maxTrain=100000;//std::numeric_limits<long int>::max();
            maxTrain=maxTrain>nData?nData:maxTrain;

            if (labels){
                std::vector<int> counts(this->m_nSegmentationLabels,0);
                ConstImageIteratorType lIt(labels,labels->GetLargestPossibleRegion());
                for (lIt.GoToBegin();!lIt.IsAtEnd();++lIt){counts[lIt.Get()]++;}
                for (int l=0;l<this->m_nSegmentationLabels;++l){
                    //if (counts[l]<maxTrain) maxTrain=counts[l];
                }
                for (int l=0;l<this->m_nSegmentationLabels;++l){
                    LOGV(7)<<VAR(l)<<" "<<VAR(counts[l])<<" "<<VAR(nFeatures)<<std::endl;
                    //this->m_observations.push_back(NEWMAT::Matrix(nFeatures,counts[l]));
                    this->m_observations.push_back(NEWMAT::Matrix(nFeatures,min((long int)counts[l],maxTrain)));
                }
            }else{
                this->m_observations.push_back(NEWMAT::Matrix(nFeatures,nData));
            }

            //maximal size
         
            LOGV(5)<<maxTrain<<" computed"<<std::endl;
            std::vector<int> counts(this->m_nSegmentationLabels,0);
            int i=0;
            typename itk::ImageRandomNonRepeatingConstIteratorWithIndex<ImageType> randomIt(labels,labels->GetLargestPossibleRegion());
            randomIt.SetNumberOfSamples(nData);
            
            for (randomIt.GoToBegin();!randomIt.IsAtEnd();++randomIt){
                int label=randomIt.Get();
                if ( counts[label] <maxTrain){
                    RGBPixelType intens=inputImage->GetPixel(randomIt.GetIndex());
                    for (int f=0;f<nFeatures;++f){
                        this->m_observations[label].element(f,counts[label])=intens[f];
                    }
                    ++counts[label];
                    ++i;
                }
            }
            

            Superclass::m_nData=i;
            LOG<<"done adding data. "<<std::endl;
            LOG<<"stored "<<this->m_nData<<" samples "<<std::endl;
            for ( int s=0;s<this->m_nSegmentationLabels;++s){
                LOG<<VAR(counts[s])<<std::endl;
            }

        }


        virtual void setData(std::vector<ImageConstPointerType> inputImage, ImageConstPointerType labels=NULL){
            LOGV(5)<<"Setting up data for intensity based segmentation classifier" << endl;
            unsigned int nFeatures=inputImage.size();
            long int nData=1;
            for (int d=0;d<ImageType::ImageDimension;++d)
                nData*=inputImage[0]->GetLargestPossibleRegion().GetSize()[d];
            this->m_observations=std::vector<NEWMAT::Matrix>();
            long int maxTrain=100000;//std::numeric_limits<long int>::max();
            maxTrain=maxTrain>nData?nData:maxTrain;

            if (labels){
                std::vector<int> counts(this->m_nSegmentationLabels,0);
                ConstImageIteratorType lIt(labels,labels->GetLargestPossibleRegion());
                for (lIt.GoToBegin();!lIt.IsAtEnd();++lIt){counts[lIt.Get()]++;}
                for (int l=0;l<this->m_nSegmentationLabels;++l){
                    //if (counts[l]<maxTrain) maxTrain=counts[l];
                }
                for (int l=0;l<this->m_nSegmentationLabels;++l){
                    LOGV(7)<<VAR(l)<<" "<<VAR(counts[l])<<" "<<VAR(nFeatures)<<std::endl;
                    //this->m_observations.push_back(NEWMAT::Matrix(nFeatures,counts[l]));
                    this->m_observations.push_back(NEWMAT::Matrix(nFeatures,min((long int)counts[l],maxTrain)));
                }
            }else{
                this->m_observations.push_back(NEWMAT::Matrix(nFeatures,nData));
            }

            //maximal size
         
            LOGV(5)<<maxTrain<<" computed"<<std::endl;
            std::vector<int> counts(this->m_nSegmentationLabels,0);
            int i=0;

            if (nFeatures>1){
                std::vector<ConstImageIteratorType> iterators;
                for (unsigned int s=0;s<nFeatures;++s){
                    iterators.push_back(ConstImageIteratorType(inputImage[s],inputImage[s]->GetLargestPossibleRegion()));
                    iterators[s].GoToBegin();
                }
                for (;!iterators[0].IsAtEnd() ; ++i)
                    {
                        int label=0;
                        if (labels)
                            label= (labels->GetPixel(iterators[0].GetIndex())  );
                    
                        // LOGV(10)<<i<<" "<<VAR(label)<<" "<<VAR(counts[label])<<" "<<nFeatures<<std::endl;
                        if ( counts[label] <maxTrain){
                            for (unsigned int f=0;f<nFeatures;++f){
                                int intens=(iterators[f].Get());
                                this->m_observations[label].element(f,counts[label])=intens;
                                ++iterators[f];
                            }
                            counts[label]++;
                        }else{
                            for (unsigned int f=0;f<nFeatures;++f){
                                ++iterators[f];
                            }
                        }
                    }
            }else{

                typename itk::ImageRandomNonRepeatingConstIteratorWithIndex<ImageType> randomIt(labels,labels->GetLargestPossibleRegion());
                randomIt.SetNumberOfSamples(nData);

                for (randomIt.GoToBegin();!randomIt.IsAtEnd();++randomIt){
                    int label=randomIt.Get();
                    if ( counts[label] <maxTrain){
                        double intens=inputImage[0]->GetPixel(randomIt.GetIndex());
                        this->m_observations[label].element(0,counts[label])=intens;
                        ++counts[label];
                        ++i;
                    }
                }
                    

            }
            Superclass::m_nData=i;
            LOG<<"done adding data. "<<std::endl;
            LOG<<"stored "<<this->m_nData<<" samples "<<std::endl;
            for ( int s=0;s<this->m_nSegmentationLabels;++s){
                LOG<<VAR(counts[s])<<std::endl;
            }

        }

        double getProbability(int label, double i1, double i2=0){
            if (this->m_trainedGMMs[label]){
                NEWMAT::ColumnVector c(1);
                c.element(0)=i1;
                //c.element(1)=i2;
                
                return this->m_GMMs[label].likelihood(c);
            }else
                return std::numeric_limits<float>::epsilon()*100;
        }
    };//class
}//namespace
