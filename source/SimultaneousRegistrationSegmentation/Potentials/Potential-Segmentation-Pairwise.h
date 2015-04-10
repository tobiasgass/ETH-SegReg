#include "Log.h"
/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _SEGMENTATIONPAIRWISEPOTENTIALS_H_
#define _SEGMENTATIONPAIRWISEPOTENTIALS_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include <itkStatisticsImageFilter.h>

namespace SRS{

    template<class TImage>
    class PairwisePotentialSegmentation: public itk::Object{
    public:
        //itk declarations
        typedef PairwisePotentialSegmentation            Self;
        typedef itk::Object Superclass;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        static const int D=ImageType::ImageDimension;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        SizeType m_targetSize;
        typedef typename itk::StatisticsImageFilter< ImageType > StatisticsFilterType;
    protected:
        ConstImagePointerType m_targetImage, m_gradientImage;
        ConstImagePointerType m_scaledTargetImage, m_scaledTargetGradient;

        SpacingType m_spacingFactor;
        //LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
        double m_gradientSigma, m_Sigma;
        double m_gradientScaling;
        ConstImagePointerType m_atlasSegmentation, m_atlasGradient, m_atlasImage;
        int m_nSegmentationLabels;
        double m_alpha;
        double m_theta;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialSegmentation, Object);

        PairwisePotentialSegmentation(){
            this->m_haveLabelMap=false;
        }
        virtual void freeMemory(){
        }
        virtual void SetTargetImage(string filename){
            if (filename!=""){
                LOG<<"warning, trying to load RGB iamge in unsuitable pairwise segmentation function!"<<endl;
            }
        }

        virtual void SetAtlasImage(string filename){
             if (filename!=""){
                LOG<<"warning, trying to load RGB iamge in unsuitable pairwise segmentation function!"<<endl;
            }
        }
        virtual void SetRGBTargetImage(string filename){ if (filename!="") LOG<<"Warning, call does nothing!"<<endl;}
        virtual void SetRGBAtlasImage(string filename){if (filename!="") LOG<<"Warning, call does nothing!"<<endl;}

        void SetAlpha(double a){m_alpha=a;}
        void SetTheta(double a){m_theta=a;}
        void SetNSegmentationLabels(int n){m_nSegmentationLabels=2;}
        virtual void Init(){
            assert(this->m_targetImage);
            assert(this->m_gradientImage);
	  
            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            filter->SetInput(this->m_gradientImage);
            filter->Update();
            this->m_gradientSigma=filter->GetSigma();
            this->m_gradientSigma*=this->m_gradientSigma;

            LOGV(5)<<"Target image gradient variance: "<<m_gradientSigma<<std::endl;
            filter->SetInput(this->m_targetImage);
            filter->Update();
            this->m_Sigma=filter->GetSigma();
            this->m_Sigma*=this->m_Sigma;	  
            LOGV(5)<<"Target image  variance: "<<m_Sigma<<std::endl;
            m_scaledTargetImage=m_targetImage;
            m_scaledTargetGradient=m_gradientImage;
           
            //compute weighting factor for cases where the grid is not isotropically spaced
            //we give a linearly lower weight for neighborhood relations with spacing>minSpacing
            //eg minSpacing=1; spacing=2, weight=0.5;
            double minSpacing=10000000.0;
            for (int d=0;d<D;++d){
                double space=this->m_targetImage->GetSpacing()[d];
                if (space<minSpacing) minSpacing = space;
            }
            for (int d=0;d<D;++d){
                m_spacingFactor[d] = 1.0*minSpacing/this->m_targetImage->GetSpacing()[d];
            }
            LOGV(4)<<VAR(minSpacing)<<" "<<VAR(m_spacingFactor)<<" "<<VAR(this->m_targetImage->GetSpacing())<<endl;
            
        }
        virtual void ResamplePotentials(double segmentationScalingFactor){
            if (m_targetImage.IsNull()){
                LOG<<"Target image not allocated, aborting"<<endl;
                exit(0);
            }
            if (m_gradientImage.IsNull()){
                LOG<<"Target gradient image not allocated, aborting"<<endl;
                exit(0);
            }
            if (segmentationScalingFactor<1.0){
                //only use gaussian smoothing if down scaling
                m_scaledTargetImage=FilterUtils<ImageType>::LinearResample(m_targetImage,segmentationScalingFactor,false,true);
                m_scaledTargetGradient=FilterUtils<ImageType>::LinearResample(m_gradientImage,segmentationScalingFactor,false,true);
            }else{
                m_scaledTargetImage=FilterUtils<ImageType>::LinearResample(m_targetImage,segmentationScalingFactor,false,false);
                m_scaledTargetGradient=FilterUtils<ImageType>::LinearResample(m_gradientImage,segmentationScalingFactor,false,false);
        
            }
            if (m_scaledTargetImage.IsNull()){
                LOG<<"Target image rescaling failed, aborting"<<endl;
                exit(0);
            }
            if (m_scaledTargetGradient.IsNull()){
                LOG<<"Target gradient image rescaling failed, aborting"<<endl;
                exit(0);
            }
            
            
        }
        void SetGradientScaling(double s){m_gradientScaling=s;}
        virtual void SetTargetImage(ConstImagePointerType targetImage){
            this->m_targetImage=targetImage;
            this->m_targetSize=this->m_targetImage->GetLargestPossibleRegion().GetSize();

        }
        virtual void SetTargetGradient(ConstImagePointerType gradientImage){
            this->m_gradientImage=gradientImage;
        }
        virtual void SetAtlasSegmentation(ConstImagePointerType im){
            m_atlasSegmentation=im;
        }
        virtual void SetAtlasGradient(ConstImagePointerType im){
            m_atlasGradient=im;
        }
        virtual void SetAtlasImage(ConstImagePointerType im){
            m_atlasImage=im;
        }
        virtual double getPotential(IndexType idx1, IndexType idx2, int label1, int label2){
            if (label1!=label2){  
                int s1=this->m_gradientImage->GetPixel(idx1);
                int s2=this->m_gradientImage->GetPixel(idx2);
                double edgeWeight=abs(s1-s2);
                edgeWeight*=edgeWeight;
                //int i1=this->m_targetImage->GetPixel(idx1);
                //int i2=this->m_targetImage->GetPixel(idx2);
                //double intensityDiff=(i1-i2)*(i1-i2);
                edgeWeight=(s1 < s2) ? 1.0 : exp( - 40* (edgeWeight/this->m_gradientSigma) );
                return edgeWeight;
            }else{
                return 0;
            }
        }
        virtual void evalImage(ConstImagePointerType im,ConstImagePointerType grad){  
            typedef typename itk::ImageRegionConstIterator< ImageType > IteratorType;
            typedef itk::Image<float, ImageType::ImageDimension> FloatImageType;
            typedef typename FloatImageType::Pointer FloatImagePointerType;
            typedef typename FloatImageType::ConstPointer FloatImageConstPointerType;
            typedef typename itk::ImageRegionIterator< FloatImageType > NewIteratorType;
            IteratorType iterator(im, im->GetLargestPossibleRegion());
            IteratorType iterator2(grad, grad->GetLargestPossibleRegion());
            FloatImagePointerType horiz=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
            FloatImagePointerType vert=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
            FloatImagePointerType sum=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
            NewIteratorType horIt(horiz, im->GetLargestPossibleRegion());
            NewIteratorType verIt(vert, im->GetLargestPossibleRegion());
            NewIteratorType sumIt(sum, im->GetLargestPossibleRegion());
            sumIt.GoToBegin();
            horIt.GoToBegin();
            verIt.GoToBegin();
            typedef typename ImageType::OffsetType OffsetType;
            for (iterator.GoToBegin(),iterator2.GoToBegin();!iterator.IsAtEnd();++iterator,++iterator2,++horIt,++verIt,++sumIt){
                IndexType idx1=iterator.GetIndex();
                OffsetType off;
                off.Fill(0);
                if (idx1[0]<(int)im->GetLargestPossibleRegion().GetSize()[0]-1){
                    off[0]+=1;
                    IndexType idx2=idx1+off;
                    horIt.Set(getPotential(idx1,idx2,0,1));
                    sumIt.Set(horIt.Get()*horIt.Get());
                }
                else sumIt.Set(0);
                off.Fill(0);
                if (idx1[1]<(int)im->GetLargestPossibleRegion().GetSize()[1]-1){
                    off[1]+=1;
                    IndexType idx2=idx1+off;
                    verIt.Set(getPotential(idx1,idx2,0,1));
                    sumIt.Set(sumIt.Get()+verIt.Get()*verIt.Get());
                    //LOG<<getPotential(idx1,idx2,0,1)<<" iterator:"<<verIt.Get()<<" "<<verIt.GetIndex()<<" "<<vert->GetPixel(verIt.GetIndex())<<endl;
                }
                if (ImageType::ImageDimension == 3){
                    off.Fill(0);
                    if (idx1[2]<(int)im->GetLargestPossibleRegion().GetSize()[2]-1){
                        off[2]+=1;
                        IndexType idx2=idx1+off;
                        sumIt.Set(sumIt.Get()+getPotential(idx1,idx2,0,1)*getPotential(idx1,idx2,0,1));
                    }
                    //LOG<<getPotential(idx1,idx2,0,1)<<" iterator:"<<verIt.Get()<<" "<<verIt.GetIndex()<<" "<<vert->GetPixel(verIt.GetIndex())<<endl;
                }
            }
          
                LOGI(8,ImageUtils<FloatImageType>::writeImage("smooth-horizontal.nii",(FloatImageConstPointerType)horiz));
                LOGI(8,ImageUtils<FloatImageType>::writeImage("smooth-vertical.nii",(FloatImageConstPointerType)vert));
                LOGI(8,ImageUtils<FloatImageType>::writeImage("smooth-sum.nii",(FloatImageConstPointerType)sum) );
         
        }

    };//class

    template<class TImage, class TSmoothnessClassifier>
    class PairwisePotentialSegmentationClassifier: public PairwisePotentialSegmentation<TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialSegmentationClassifier            Self;
        typedef PairwisePotentialSegmentation<TImage> Superclass;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef TSmoothnessClassifier ClassifierType;
        typedef typename ClassifierType::Pointer ClassifierPointerType;
    private:
        ClassifierPointerType m_classifier;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialSegmentationClassifier, Object);
        virtual void Init(string filename, bool train){
            assert(this->m_targetImage);
            assert(this->m_gradientImage);
            assert(this->m_atlasSegmentation);
            assert(this->m_atlasGradient);
            assert(this->m_atlasImage);
            m_classifier=ClassifierType::New();
            m_classifier->setNIntensities(3000);
            m_classifier->setNSegmentationLabels(this->m_nSegmentationLabels);
            this->m_classifier->setData( this->m_atlasImage,(ConstImagePointerType)this->m_atlasSegmentation,(ConstImagePointerType)this->m_atlasGradient);
#if 1
            m_classifier->train( train,filename);
            //m_classifier->saveProbs("segmentationPairwise.probs");
#else
            //m_classifier->loadProbs("segmentationPairwise.probs");
#endif
        }
        virtual void Init(string filename){
            assert(false);
            assert(this->m_targetImage);
            assert(this->m_gradientImage);
            m_classifier=ClassifierType::New();
            
            //m_classifier->LoadProbs(filename);
        }
        virtual void SetClassifier(ClassifierPointerType c){ m_classifier=c;}
#if 1
        virtual void evalImage(ConstImagePointerType im,ConstImagePointerType grad){
            assert(ImageType::ImageDimension==2);
            typedef typename itk::ImageRegionConstIterator< ImageType > IteratorType;
            typedef itk::Image<float, ImageType::ImageDimension> FloatImageType;
            typedef typename FloatImageType::Pointer FloatImagePointerType;
            typedef typename FloatImageType::ConstPointer FloatImageConstPointerType;
            typedef typename itk::ImageRegionIterator< FloatImageType > NewIteratorType;
            IteratorType iterator(im, im->GetLargestPossibleRegion());
            IteratorType iterator2(grad, grad->GetLargestPossibleRegion());
            FloatImagePointerType horiz=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
            FloatImagePointerType vert=FilterUtils<ImageType,FloatImageType>::createEmpty(im);
            NewIteratorType horIt(horiz, im->GetLargestPossibleRegion());
            NewIteratorType verIt(vert, im->GetLargestPossibleRegion());
            horIt.GoToBegin();
            verIt.GoToBegin();
            typedef typename ImageType::OffsetType OffsetType;
            for (iterator.GoToBegin(),iterator2.GoToBegin();!iterator.IsAtEnd();++iterator,++iterator2,++horIt,++verIt){
                IndexType idx1=iterator.GetIndex();
              
                OffsetType off;
                off.Fill(0);
                if (idx1[0]<(int)im->GetLargestPossibleRegion().GetSize()[0]-1){
                    off[0]+=1;
                    IndexType idx2=idx1+off;
                    horIt.Set(getPotential(idx1,idx2,0,1));
                }
                off.Fill(0);
                if (idx1[1]<(int)im->GetLargestPossibleRegion().GetSize()[1]-1){
                    off[1]+=1;
                    IndexType idx2=idx1+off;
                    verIt.Set(getPotential(idx1,idx2,0,1));
                    //LOG<<getPotential(idx1,idx2,0,1)<<" iterator:"<<verIt.Get()<<" "<<verIt.GetIndex()<<" "<<vert->GetPixel(verIt.GetIndex())<<endl;
                }
            }
            if (ImageType::ImageDimension == 2){
                typedef itk::RescaleIntensityImageFilter<FloatImageType,ImageType> CasterType;
                //typedef itk::CastImageFilter<ImageType,ImageType> CasterType;
                typename CasterType::Pointer caster=CasterType::New();
                caster->SetOutputMinimum( numeric_limits<typename ImageType::PixelType>::min() );
                caster->SetOutputMaximum( numeric_limits<typename ImageType::PixelType>::max() );
                caster->SetInput(horiz);
                caster->Update();
                //ImageUtils<ImageType>::writeImage("smooth-horizontal.nii",(ConstImagePointerType)caster->GetOutput());
                caster->SetInput(vert);
                caster->Update();
                //ImageUtils<ImageType>::writeImage("smooth-vertical.nii",(ConstImagePointerType)caster->GetOutput());
            }else{
                if (false){
                    //ImageUtils<FloatImageType>::writeImage("smooth-horizontal.nii",(FloatImageConstPointerType)horiz);
                    //ImageUtils<FloatImageType>::writeImage("smooth-vertical.nii",(FloatImageConstPointerType)vert);
                }
            }
        }
#endif
        ClassifierPointerType GetClassifier(){return m_classifier;}
        virtual double getPotential(IndexType idx1, IndexType idx2, int label1, int label2){
            if (label1==label2) return 0;
#if 1
            if (!label1 && label2){

            }else{
                IndexType tmp=idx1;idx1=idx2;idx2=tmp;
                int tmpL=label1;label1=label2;label2=tmpL;
            }
#endif       
            int s1=this->m_gradientImage->GetPixel(idx1);
            int s2=this->m_gradientImage->GetPixel(idx2);
            //       if (s1<s2) return 100;
            double gradientDiff=(s1-s2);
            //   if (s1<s2 && label1!=label2) return 100;
            int i1=this->m_targetImage->GetPixel(idx1);
            int i2=this->m_targetImage->GetPixel(idx2);
            double intensityDiff=(i1-i2);
            //double prob=m_classifier->px_l(intensityDiff,label1!=label2,gradientDiff);
            double prob=m_classifier->px_l(intensityDiff,label1,gradientDiff,label2);
            //double prob=m_classifier->px_l(i1,i2,s1,s2,label1,label2);
            if (prob<=0.000000001) prob=0.00000000001;
            //LOG<<"Pairwise: "<<(label1!=label2)<<" "<<gradientDiff<<" "<<intensityDiff<<" "<<prob<<" "<<-log(prob)<<endl;
            //return 1+100*(-log(prob));
            if (prob>1){
                LOG<<VAR(prob)<<endl;
            }
                
            return (-log(prob));
            //return 1.0-prob;
        }
      
    };//class


    template<class TImage>
    class PairwisePotentialSegmentationMarcel: public PairwisePotentialSegmentation<TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialSegmentationMarcel            Self;
        typedef PairwisePotentialSegmentation<TImage> Superclass;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;
        static const int D=TImage::ImageDimension;
        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::PointType PointType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef typename ImageType::IndexType IndexType;
  
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialSegmentationMarcel, Object);

        virtual double getPotential(IndexType idx1, IndexType idx2, int label1, int label2){
            //equal labels don't have costs
            if (label1==label2) return 0;
            //always penalize secondary-to-primary label
            double gradientCost;
            double factor=1.0;

            if (  ((label1==2 &&label2 ) || (label2 == 2 && label1)) ) {
                factor=0.5;
            }
                
            {
#if 1
                if (!label1 && label2){
                    
                }else{
                    IndexType tmp=idx1;idx1=idx2;idx2=tmp;
                    int tmpL=label1;label1=label2;label2=tmpL;
                }
#endif       


                double s1=1.0*this->m_scaledTargetGradient->GetPixel(idx1);
                double s2=1.0*this->m_scaledTargetGradient->GetPixel(idx2);
                double gradientDiff=fabs(s1-s2)/100;
                                
                gradientCost=(s1>s2)?1:exp(-5*gradientDiff);
                //LOGV(30)<<s1<<" "<<s2<<" "<<" "<<gradientDiff<<" "<<gradientCost<<std::endl;
            }

            //get neighborhood direction
            double directionalFactor=1.0;
          
            for (int d=0;d<D;++d){
                if (abs(idx1[d]-idx2[d])){
                    directionalFactor=this->m_spacingFactor[d];
                    break;
                }
            }
            //return 1.0+1000.0*factor*gradientCost;
            return factor*gradientCost*directionalFactor;
        }
    };//class

    template<class TImage>
    class PairwisePotentialSegmentationContrastWithGradient: public PairwisePotentialSegmentation<TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialSegmentationContrastWithGradient            Self;
        typedef PairwisePotentialSegmentation<TImage> Superclass;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef typename ImageType::IndexType IndexType;
        static const int D=ImageType::ImageDimension;
    private:
        std::vector<std::vector<double> > m_labelProbs;
         
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialSegmentationContrastWithGradient, Object);
        void SetNSegmentationLabels(int n){this->m_nSegmentationLabels=n;}

        virtual void Init(){
            assert(this->m_targetImage);
            assert(this->m_gradientImage);
            typedef typename itk::StatisticsImageFilter< ImageType > StatisticsFilterType;

            typename StatisticsFilterType::Pointer filter=StatisticsFilterType::New();
            
        


            filter->SetInput(FilterUtils<ImageType>::gradient(this->m_gradientImage));
            filter->Update();
            this->m_gradientSigma=filter->GetSigma();
            this->m_gradientSigma*=this->m_gradientSigma;
            LOGV(3)<<VAR( this->m_gradientSigma )<<" "<<VAR(this->m_alpha)<<endl;
            if (this->m_alpha>0){
                this->m_gradientSigma*=this->m_alpha;
            }
            LOGV(5)<<"Target image gradient variance: "<<this->m_gradientSigma<<std::endl;
            typename StatisticsFilterType::Pointer filter2=StatisticsFilterType::New();

            filter2->SetInput(FilterUtils<ImageType>::gradient(this->m_targetImage));
            filter2->Update();
            this->m_Sigma=filter2->GetSigma();
            this->m_Sigma*=this->m_Sigma;	  
            LOGV(5)<<"Target image  variance: "<<this->m_Sigma<<std::endl;
            this->m_scaledTargetImage=this->m_targetImage;
            this->m_scaledTargetGradient=this->m_gradientImage;

            typedef typename itk::ImageRegionConstIteratorWithIndex< ImageType > IteratorType;
            typedef typename ImageType::OffsetType OffsetType;
            IteratorType iterator(this->m_atlasSegmentation, this->m_atlasSegmentation->GetLargestPossibleRegion());
            std::vector<int> totalCounts(this->m_nSegmentationLabels,0);
            m_labelProbs= std::vector<std::vector<double> > (this->m_nSegmentationLabels,std::vector<double>(this->m_nSegmentationLabels,0));
            LOGV(3)<<"Computing pairwise segmentation label probabilities..."<<endl;
            for (iterator.GoToBegin();!iterator.IsAtEnd();++iterator){
                IndexType idx1=iterator.GetIndex();
                int label1=iterator.Get();
                for (int d=0;d<D;++d){
                    OffsetType off;
                    off.Fill(0);
                    off[d]+=1;
                    IndexType idx2=idx1+off;
                    if (idx2[d]<this->m_atlasSegmentation->GetLargestPossibleRegion().GetSize()[d]){
                        int label2=this->m_atlasSegmentation->GetPixel(idx2);
                        if (label1!=label2){
                            //exclude selg neighborhood
                            if (label2<label1){
                                m_labelProbs[label2][label1]+=1;
                            }else{
                                m_labelProbs[label1][label2]+=1;
                            }
                      
                            ++totalCounts[label1];
                            ++totalCounts[label2];
                        }
                    }
                    
                }
            }
            for (int l1=0;l1<this->m_nSegmentationLabels;++l1){
                for (int l2=l1;l2<this->m_nSegmentationLabels;++l2){
                    m_labelProbs[l1][l2]/=0.5*(totalCounts[l1]+totalCounts[l2]);
                    LOGV(5)<<VAR(l1)<<" "<<VAR(l2)<<" "<<VAR( m_labelProbs[l1][l2])<<std::endl;
                    if (m_labelProbs[l1][l2]<=0.0){
                        m_labelProbs[l1][l2]=std::numeric_limits<float>::epsilon()*100;
                    }
                }
            }
            //make symmetric
            for (int l1=0;l1<this->m_nSegmentationLabels;++l1){
                for (int l2=l1;l2<this->m_nSegmentationLabels;++l2){
                    m_labelProbs[l2][l1]=m_labelProbs[l1][l2];
                }
            }


            
        }


        virtual double getPotential(IndexType idx1, IndexType idx2, int label1, int label2){
            //equal labels don't have costs
            if (label1==label2) return 0;

            double gradientCost;
            //LOGV(10)<<VAR(this->m_gradientSigma)<<" "<<VAR(-log(m_labelProbs[label1][label2]))<<endl;
            double factor=1.0;//-log(m_labelProbs[label1][label2]);

            {
                double s1=1.0*this->m_scaledTargetGradient->GetPixel(idx1);
                double s2=1.0*this->m_scaledTargetGradient->GetPixel(idx2);
                double gradientDiff=(s1-s2)*(s1-s2)/this->m_gradientSigma;
                gradientCost=exp(-0.5*fabs(gradientDiff));
                
                //LOGV(30)<<s1<<" "<<s2<<" "<<" "<<gradientDiff<<" "<<gradientCost<<std::endl;
            }
            //return 1.0+1000.0*factor*gradientCost;
            return factor*gradientCost;
        }
    };//class
    template<class TImage>
    class PairwisePotentialSegmentationRGBContrast: public PairwisePotentialSegmentation<TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialSegmentationRGBContrast            Self;
        typedef PairwisePotentialSegmentation<TImage> Superclass;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::OffsetType OffsetType;
        static const int D=ImageType::ImageDimension;
        typedef  itk::RGBPixel<unsigned char> RGBPixelType;
        typedef typename itk::Image<RGBPixelType,D > RGBImageType;
        typedef typename RGBImageType::Pointer RGBImagePointerType;
    private:
        RGBImagePointerType m_targetRGBImage,m_atlasRGBImage,m_scaledTargetRGBImage;
        std::vector<std::vector<double> > m_labelProbs;

    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialSegmentationContrastWithGradient, Object);
        void SetNSegmentationLabels(int n){this->m_nSegmentationLabels=n;}
        void SetRGBTargetImage(string filename){
            m_targetRGBImage=ImageUtils<RGBImageType>::readImage(filename);
            this->m_targetSize=this->m_targetRGBImage->GetLargestPossibleRegion().GetSize();
            if (this->m_targetImage.IsNotNull()){
#if 1
                LOGV(3)<<"resampling RGB image from "<<VAR(m_targetRGBImage)<<" to "<<VAR( this->m_targetImage)<<endl;
                //smoothing for RGB images seems not to work using the FilterUtils built-in function :(
                m_targetRGBImage=FilterUtils<RGBImageType>::LinearResample( m_targetRGBImage,
                                                                            this->m_targetImage->GetRequestedRegion().GetSize(),
                                                                            this->m_targetImage->GetOrigin(),
                                                                            this->m_targetImage->GetSpacing(),
                                                                            this->m_targetImage->GetDirection(),
                                                                            false);
                            
#endif
            }
            this->m_targetImage=NULL;
            
        }

        void SetRGBAtlasImage(string filename){
            m_atlasRGBImage=ImageUtils<RGBImageType>::readImage(filename);
            this->m_atlasImage=NULL;
        }
        virtual void Init(){
           
            this->m_gradientSigma=1.0;
            typename itk::ImageRegionIterator<RGBImageType> iterator(m_targetRGBImage,m_targetRGBImage->GetRequestedRegion());
            iterator.GoToBegin();                   
            double meanDiff=0.0;
            int count=0;
            for (;!iterator.IsAtEnd();++iterator){
                IndexType idx=iterator.GetIndex();
                OffsetType off;
                for (int d=0;d<D;++d){
                    off.Fill(0);
                    off[d]=1;
                    IndexType idx2=idx+off;
                    if ( m_targetRGBImage->GetRequestedRegion().IsInside(idx2)){
                        double pot=getPotential(idx,idx2,0,1,true);
                        //LOGV(10)<<VAR(pot)<<endl;
                        meanDiff+=(pot);
                        ++count;
                    }
                }

            }
            this->m_scaledTargetRGBImage=this->m_targetRGBImage;
            LOGV(3)<<            VAR(this->m_scaledTargetRGBImage)<<" "<<VAR(this->m_targetRGBImage)<<endl;;
            this->m_gradientSigma=meanDiff/count*this->m_alpha;
            LOGV(3)<<VAR(this->m_gradientSigma)<<endl;

            typedef typename itk::ImageRegionConstIteratorWithIndex< ImageType > IteratorType;
            typedef typename ImageType::OffsetType OffsetType;
            IteratorType iterator2(this->m_atlasSegmentation, this->m_atlasSegmentation->GetLargestPossibleRegion());
            std::vector<int> totalCounts(this->m_nSegmentationLabels,0);
            m_labelProbs= std::vector<std::vector<double> > (this->m_nSegmentationLabels,std::vector<double>(this->m_nSegmentationLabels,0));
            LOGV(3)<<"Computing pairwise segmentation label probabilities..."<<endl;
            for (iterator2.GoToBegin();!iterator2.IsAtEnd();++iterator2){
                IndexType idx1=iterator2.GetIndex();
                int label1=iterator2.Get();
                for (int d=0;d<D;++d){
                    OffsetType off;
                    off.Fill(0);
                    off[d]+=1;
                    IndexType idx2=idx1+off;
                    if (idx2[d]<this->m_atlasSegmentation->GetLargestPossibleRegion().GetSize()[d]){
                        int label2=this->m_atlasSegmentation->GetPixel(idx2);
                        if (label1!=label2){
                            //exclude selg neighborhood
                            if (label2<label1){
                                m_labelProbs[label2][label1]+=1;
                            }else{
                                m_labelProbs[label1][label2]+=1;
                            }
                      
                            ++totalCounts[label1];
                            ++totalCounts[label2];
                        }
                    }
                    
                }
            }
            for (int l1=0;l1<this->m_nSegmentationLabels;++l1){
                for (int l2=l1;l2<this->m_nSegmentationLabels;++l2){
                    m_labelProbs[l1][l2]/=0.5*(totalCounts[l1]+totalCounts[l2]);
                    LOGV(5)<<VAR(l1)<<" "<<VAR(l2)<<" "<<VAR( m_labelProbs[l1][l2])<<std::endl;
                    if (m_labelProbs[l1][l2]<=0.0){
                        m_labelProbs[l1][l2]=std::numeric_limits<float>::epsilon();
                    }
                }
            }
            //make symmetric
            for (int l1=0;l1<this->m_nSegmentationLabels;++l1){
                for (int l2=l1;l2<this->m_nSegmentationLabels;++l2){
                    m_labelProbs[l2][l1]=m_labelProbs[l1][l2];
                }
            }

        }
        virtual void ResamplePotentials(double segmentationScalingFactor){
            if (segmentationScalingFactor!=1.0){
                LOGV(3)<<"Resampling RGB image for segmentation pairwise"<<endl;
                this->m_scaledTargetRGBImage=FilterUtils<RGBImageType>::LinearResample(this->m_targetRGBImage,segmentationScalingFactor,true);
            }else
                this->m_scaledTargetRGBImage=this->m_targetRGBImage;
            
        }
        virtual double getPotential(IndexType idx1, IndexType idx2, int label1, int label2){
            return getPotential(idx1,idx2,label1,label2,false);
        }

        virtual double getPotential(IndexType idx1, IndexType idx2, int label1, int label2,bool nonExp){
            //equal labels don't have costs
            if (label1==label2) return 0;

            double gradientCost;
            //LOGV(10)<<VAR(this->m_gradientSigma)<<" "<<VAR(-log(m_labelProbs[label1][label2]))<<endl;
            double factor=1.0;
            if (false && ! nonExp)
                factor=-log(m_labelProbs[label1][label2]);

            {
#if 0                
                RGBPixelType s1=this->m_scaledTargetRGBImage->GetPixel(idx1);
                RGBPixelType s2=this->m_scaledTargetRGBImage->GetPixel(idx2);
#else
                RGBPixelType s1=this->m_targetRGBImage->GetPixel(idx1);
                RGBPixelType s2=this->m_targetRGBImage->GetPixel(idx2);
#endif
                double gradientDiff=0.0;
                for (unsigned int c=0;c<3;++c){
                    double diff = 1.0*s1[c] - s2[c];
                    gradientDiff+=diff*diff;
                }
                
                if (nonExp){
                    gradientCost=gradientDiff;
                }
                else{
                    gradientCost=this->m_theta+exp(-0.5*fabs(gradientDiff/this->m_gradientSigma));
                }
                //LOGV(30)<<s1<<" "<<s2<<" "<<" "<<gradientDiff<<" "<<gradientCost<<std::endl;
            }
            //return 1.0+1000.0*factor*gradientCost;
            return factor*gradientCost;
        }
    };//class


    template<class TImage>
    class PairwisePotentialSegmentationUniform: public PairwisePotentialSegmentation<TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialSegmentationUniform            Self;
        typedef PairwisePotentialSegmentation<TImage> Superclass;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef typename ImageType::IndexType IndexType;
  
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(PairwisePotentialSegmentationUniform, Object);

        virtual double getPotential(IndexType idx1, IndexType idx2, int label1, int label2){
            //equal labels don't have costs
            if (label1==label2) return 0;
            else return 1;
        }
    };//class


    template<class TImage, class TSmoothnessClassifier>
    class CachingPairwisePotentialSegmentationClassifier: public PairwisePotentialSegmentationClassifier<TImage,TSmoothnessClassifier >{
    public:
        //itk declarations
        typedef CachingPairwisePotentialSegmentationClassifier            Self;
        typedef PairwisePotentialSegmentationClassifier<TImage,TSmoothnessClassifier> Superclass;
        typedef itk::SmartPointer<Self>        Pointer;
        typedef itk::SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;

        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef TSmoothnessClassifier ClassifierType;
        typedef typename ClassifierType::Pointer ClassifierPointerType;
        typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;

    private:
        ClassifierPointerType m_classifier;
        bool m_trainOnTargetROI;
        std::vector<FloatImagePointerType> m_probabilityImages,m_resampledProbImages;

    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(CachingPairwisePotentialSegmentationClassifier, Object);
        virtual void Init(string filename, bool train){
            assert(this->m_targetImage);
            assert(this->m_gradientImage);
            assert(this->m_atlasSegmentation);
            assert(this->m_atlasGradient);
            assert(this->m_atlasImage);
            m_classifier=ClassifierType::New();
            m_classifier->setNIntensities(3000);
            m_classifier->setNSegmentationLabels(this->m_nSegmentationLabels);
            
            this->m_classifier->setData( this->m_atlasImage,(ConstImagePointerType)this->m_atlasSegmentation,(ConstImagePointerType)this->m_atlasGradient);
            m_classifier->train(train,filename);
            // m_classifier->cachePotentials(this->m_targetImage,this->m_gradientImage);
            
            m_probabilityImages= m_classifier->getProbabilities(this->m_targetImage,this->m_gradientImage);
            
        }
        virtual  void Init(){
            m_trainOnTargetROI=true;

            assert(this->m_targetImage);
            assert(this->m_gradientImage);
            assert(this->m_atlasSegmentation);
            assert(this->m_atlasGradient);
            assert(this->m_atlasImage);
            if (m_trainOnTargetROI){
                this->m_atlasImage=FilterUtils<ImageType>::NNResample(this->m_atlasImage,this->m_targetImage,false);
                this->m_atlasSegmentation=FilterUtils<ImageType>::NNResample(this->m_atlasSegmentation,this->m_targetImage,false);
                this->m_atlasGradient=FilterUtils<ImageType>::NNResample(this->m_atlasGradient,this->m_targetImage,false);
            }
            m_classifier=ClassifierType::New();
            m_classifier->setNIntensities(3000);
            m_classifier->setNSegmentationLabels(this->m_nSegmentationLabels);
            this->m_classifier->setData( this->m_atlasImage,(ConstImagePointerType)this->m_atlasSegmentation,(ConstImagePointerType)this->m_atlasGradient);
            m_classifier->train(true);
            //            m_classifier->cachePotentials(this->m_targetImage,this->m_gradientImage);
            m_probabilityImages=     m_classifier->getProbabilities(this->m_targetImage,this->m_gradientImage);

        }
        virtual void Init(string filename){
            assert(false);
            assert(this->m_targetImage);
            assert(this->m_gradientImage);
            if (m_classifier) delete m_classifier;
            m_classifier=ClassifierType::New();
        }
        virtual void SetClassifier(ClassifierPointerType c){ m_classifier=c;}
        virtual void evalImage(ConstImagePointerType im,ConstImagePointerType grad){
            
        }
        void ResamplePotentials(double scale){
            m_resampledProbImages= std::vector<FloatImagePointerType>(m_probabilityImages.size());
            for (int i=0;i<m_probabilityImages.size();++i){
                m_resampledProbImages[i]=FilterUtils<FloatImageType>::LinearResample(m_probabilityImages[i],scale,true);
            }
        }
        ClassifierPointerType GetClassifier(){return m_classifier;}
        virtual double getPotential(IndexType idx1, IndexType idx2, int label1, int label2){
         
            if (label1==label2){
                return 0;
            }
            double probabilityEqualLabel;
#if 1
            //get probability from own cache
            for (unsigned int d=0;d<ImageType::ImageDimension;++d){
                int diff= idx1[d]-idx2[d];
                if (diff>0){
                    probabilityEqualLabel=1-m_resampledProbImages[d]->GetPixel(idx1);
                    break;
                }else if (diff<0){
                    probabilityEqualLabel=1-m_resampledProbImages[d]->GetPixel(idx2);
                    break;
                }
            }
#else
            //use cache of classifier, doesnt work with resampling
            probabilityEqualLabel=m_classifier->getCachedPotential(idx1,idx2);
#endif
          
            if (probabilityEqualLabel<=std::numeric_limits<double>::epsilon()){
                probabilityEqualLabel=std::numeric_limits<double>::epsilon();
            }
            return -log(probabilityEqualLabel);
            //return 1.0-probabilityEqualLabel;
        }
      
    };//class

}//namespace
#endif /* POTENTIALS_H_ */
