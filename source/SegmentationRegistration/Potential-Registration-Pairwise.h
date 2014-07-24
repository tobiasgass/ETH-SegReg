#include "Log.h"
/*
 * Potentials.h
 *
 *  Created on: Nov 24, 2010
 *      Author: gasst
 */

#ifndef _REGISTRATIONPAIRWISEPOTENTIAL_H_
#define _REGISTRATIONPAIRWISEPOTENTIAL_H_
#include "itkObject.h"
#include "itkObjectFactory.h"
#include <utility>
#include "itkVector.h"
#include "itkLinearInterpolateImageFunction.h"

namespace itk{

    template<class TLabelMapper,class TImage>
    class PairwisePotentialRegistration : public itk::Object{
    public:
        //itk declarations
        typedef PairwisePotentialRegistration            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::PointType PointType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        static const unsigned int D=ImageType::ImageDimension;
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename itk::Vector<double,ImageType::ImageDimension> SpacingType;
        typedef LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
    protected:
        SizeType m_targetSize,m_atlasSize;
        ConstImagePointerType m_targetImage, m_atlasImage;
        LabelImagePointerType m_baseLabelMap;
        bool m_haveLabelMap;
        SpacingType m_gridSpacing;
        double m_maxDist;
        double m_threshold;
        bool m_fullRegPairwise;
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationPairwisePotential, Object);

        PairwisePotentialRegistration(){
            m_haveLabelMap=false;
            m_threshold=std::numeric_limits<double>::max();
            m_fullRegPairwise=false;
        }
        virtual void freeMemory(){
        }
        virtual void setThreshold(double t){m_threshold=t;}
        void SetBaseLabelMap(LabelImagePointerType blm){m_baseLabelMap=blm;m_haveLabelMap=true;}
        LabelImagePointerType GetBaseLabelMap(LabelImagePointerType blm){return m_baseLabelMap;}
        void SetTargetImage(ConstImagePointerType targetImage){
            m_targetImage=targetImage;
            m_targetSize=m_targetImage->GetLargestPossibleRegion().GetSize();
        }
        void SetSpacing(SpacingType  sp) {
            m_gridSpacing=sp;
            m_maxDist=0.0;
            for (unsigned int d=0;d<D;++d){
                m_maxDist+=sp[d]*sp[d];
            }
            //m_maxDist=sqrt(m_maxDist);
        }
        virtual void setFullRegularization(bool b){ m_fullRegPairwise = b; }
        virtual inline double getPotential(PointType pt1, PointType pt2,LabelType displacement1, LabelType displacement2){
            assert(m_haveLabelMap);
            double result=0;
            IndexType targetIndex1, targetIndex2;
            //LOGV(50)<<VAR(targetIndex1)<<endl;            
            m_baseLabelMap->TransformPhysicalPointToIndex(pt1,targetIndex1);
            //LOGV(50)<<VAR(targetIndex1)<<endl;            
            m_baseLabelMap->TransformPhysicalPointToIndex(pt2,targetIndex2);
            //LOG<<VAR(targetIndex1)<<" "<<m_baseLabelMap->GetLargestPossibleRegion().GetSize()<<endl;
			LabelType oldl1=m_baseLabelMap->GetPixel((targetIndex1));
			LabelType oldl2=m_baseLabelMap->GetPixel((targetIndex2));
			double d1,d2;
			//double delta;
            //LOGV(50)<<VAR(displacement1)<<" "<<VAR(oldl1)<<endl;
            //LOGV(50)<<VAR(displacement2)<<" "<<VAR(oldl2)<<endl;
            if (m_fullRegPairwise){
                displacement1+=oldl1;
                displacement2+=oldl2;
            }
#if 1
            
            LabelType diff=displacement1-displacement2;
            //result=diff.GetSquaredNorm();
            result=diff.GetNorm();
          
            
            LOGV(13)<<VAR(result)<<" "<<VAR(displacement1)<<" "<<VAR(displacement2)<<endl;
            //if (result > 200) result = 20000;

            

            LOGV(50)<<VAR(displacement1)<<" "<<VAR(displacement2)<<endl;
#else
            double axisPositionDifference;
			for (unsigned int d=0;d<D;++d){

				d1=displacement1[d];
				d2=displacement2[d];
				//delta=(pt2[d]-pt1[d]);
                axisPositionDifference=1.0*(d2-d1);//(m_gridSpacing[d]);
                //result+=(axisPositionDifference)*(axisPositionDifference);
                result+=fabs(axisPositionDifference);
			}
#endif
			//			if (false){
            //LOGV(50)<<VAR(result)<<" "<<VAR(displacement1)<<" "<<VAR(displacement2)<<" "<<VAR(targetIndex1)<<" "<<VAR(targetIndex1)<<" "<<VAR(oldl1)<<" "<<VAR(oldl2)<<endl;
            if (m_threshold<numeric_limits<double>::max()){
                result=min(m_maxDist*m_threshold,(result));
            }
            return (result);
        }
            
        virtual inline double getPotential(IndexType targetIndex1, IndexType targetIndex2,LabelType displacement1, LabelType displacement2){
            LOG<<"DEPRECATED, do not call!"<<endl;
            exit(0);
            assert(m_haveLabelMap);
            double result=0;
            PointType pt1,pt2;
            LOGV(50)<<VAR(targetIndex1)<<endl;            
            //m_coarseGraphImage->TransformIndexToPhysicalPoint(targetIndex1,pt1);
            m_baseLabelMap->TransformPhysicalPointToIndex(pt1,targetIndex1);
            LOGV(50)<<VAR(targetIndex1)<<endl;            
            //m_coarseGraphImage->TransformIndexToPhysicalPoint(targetIndex2,pt2);
            m_baseLabelMap->TransformPhysicalPointToIndex(pt2,targetIndex2);
            //LOG<<VAR(targetIndex1)<<" "<<m_baseLabelMap->GetLargestPossibleRegion().GetSize()<<endl;
			LabelType oldl1=m_baseLabelMap->GetPixel((targetIndex1));
			LabelType oldl2=m_baseLabelMap->GetPixel((targetIndex2));
			double d1,d2;
			double delta;
            LOGV(50)<<VAR(displacement1)<<" "<<VAR(oldl1)<<endl;
            LOGV(50)<<VAR(displacement2)<<" "<<VAR(oldl2)<<endl;

            displacement1+=oldl1;
			displacement2+=oldl2;

            LOGV(50)<<VAR(displacement1)<<" "<<VAR(displacement2)<<endl;

			for (unsigned int d=0;d<D;++d){

				d1=displacement1[d];
				d2=displacement2[d];
                //	delta=(pt2[d]-pt1[d]);
				double axisPositionDifference=1.0*(d2-d1);//(m_gridSpacing[d]);
              
				result+=(axisPositionDifference)*(axisPositionDifference);
			}

			//			if (false){
            LOGV(50)<<VAR(result)<<" "<<VAR(displacement1)<<" "<<VAR(displacement2)<<" "<<VAR(targetIndex1)<<" "<<VAR(targetIndex1)<<" "<<VAR(oldl1)<<" "<<VAR(oldl2)<<endl;
            if (m_threshold<numeric_limits<double>::max()){
                result=min(m_maxDist*m_threshold,(result));
            }
            return result;
        }
    };//class
    template<class TLabelMapper,class TImage>
    class PairwisePotentialRegistrationSigmoid : public PairwisePotentialRegistration<TLabelMapper,TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialRegistrationSigmoid            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        static const unsigned int D=ImageType::ImageDimension;
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
  
        
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationPairwisePotentialSigmoid, Object);

   
        
        virtual double getPotential(IndexType targetIndex1, IndexType targetIndex2,LabelType displacement1, LabelType displacement2){
            assert(this->m_haveLabelMap);
            double result=0;
            

			LabelType oldl1=this->m_baseLabelMap->GetPixel((targetIndex1));
			LabelType oldl2=this->m_baseLabelMap->GetPixel((targetIndex2));
			double d1,d2;
			double delta;
			displacement1+=oldl1;
			displacement2+=oldl2;
			for (unsigned int d=0;d<D;++d){

				d1=displacement1[d];
				d2=displacement2[d];
				delta=(targetIndex2[d]-targetIndex1[d]);
                if (delta>0){
                    double axisPositionDifference=((1.0*fabs(d2-d1)-delta/3)/(delta/5));//(this->m_spacing[d]);
                    result+=(axisPositionDifference);
                }
			}

			//			if (false){
            //LOG<<displacement1-displacement2<<" "<<1.0/(1+exp(-result))<<endl;
            return 1.0/(1+exp(-result));
        }
    };//class
 template<class TLabelMapper,class TImage>
    class PairwisePotentialRegistrationL1 : public PairwisePotentialRegistration<TLabelMapper,TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialRegistrationL1            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        static const unsigned int D=ImageType::ImageDimension;
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
  
        
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationPairwisePotentialSigmoid, Object);

   
        
     virtual inline double getPotential(PointType pt1, PointType pt2,LabelType displacement1, LabelType displacement2){

            double result=0;
            IndexType targetIndex1, targetIndex2;
            //LOGV(50)<<VAR(targetIndex1)<<endl;            
            this->m_baseLabelMap->TransformPhysicalPointToIndex(pt1,targetIndex1);
            //LOGV(50)<<VAR(targetIndex1)<<endl;            
            this->m_baseLabelMap->TransformPhysicalPointToIndex(pt2,targetIndex2);
            //LOG<<VAR(targetIndex1)<<" "<<this->m_baseLabelMap->GetLargestPossibleRegion().GetSize()<<endl;
			LabelType oldl1=this->m_baseLabelMap->GetPixel((targetIndex1));
			LabelType oldl2=this->m_baseLabelMap->GetPixel((targetIndex2));
			double d1,d2;
			//double delta;
            //LOGV(50)<<VAR(displacement1)<<" "<<VAR(oldl1)<<endl;
            //LOGV(50)<<VAR(displacement2)<<" "<<VAR(oldl2)<<endl;
            if (this->m_fullRegPairwise){
                displacement1+=oldl1;
                displacement2+=oldl2;
            }
            
            LabelType diff=displacement1-displacement2;

            result=0.0;
            for (unsigned int d=0;d<D;++d){
                result+=fabs(diff[d]);
            }
          
            if (this->m_threshold<numeric_limits<double>::max()){
                result=min(this->m_maxDist*this->m_threshold,(result));
            }
            return (result);
        }
    };//class

    template<class TLabelMapper,class TImage>
    class PairwisePotentialRegistrationACP : public PairwisePotentialRegistration<TLabelMapper,TImage>{
    public:
        //itk declarations
        typedef PairwisePotentialRegistrationACP            Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;

        typedef	TImage ImageType;
        typedef typename ImageType::Pointer ImagePointerType;
        typedef typename ImageType::ConstPointer ConstImagePointerType;
        static const unsigned int D=ImageType::ImageDimension;
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType LabelType;
        typedef typename ImageType::IndexType IndexType;
        typedef typename ImageType::PointType PointType;
        typedef typename ImageType::SizeType SizeType;
        typedef typename ImageType::SpacingType SpacingType;
        typedef LinearInterpolateImageFunction<ImageType> InterpolatorType;
        typedef typename InterpolatorType::Pointer InterpolatorPointerType;
        typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
        typedef typename LabelMapperType::LabelImagePointerType LabelImagePointerType;
  
        
    public:
        /** Method for creation through the object factory. */
        itkNewMacro(Self);
        /** Standard part of every itk Object. */
        itkTypeMacro(RegistrationPairwisePotentialSigmoid, Object);

   
        
        virtual inline double getPotential(PointType pt1, PointType pt2,LabelType displacement1, LabelType displacement2){
            assert(this->m_haveLabelMap);
            double leftCost=0, rightCost=0;
            double controlPointDistance=0.0;
            double delta;
            IndexType targetIndex1, targetIndex2;
            this->m_baseLabelMap->TransformPhysicalPointToIndex(pt1,targetIndex1);
            this->m_baseLabelMap->TransformPhysicalPointToIndex(pt2,targetIndex2);
            
            //get neighboring axis, and make sure Index1<index2
            IndexType rightNeighbor=targetIndex2, leftNeighbor=targetIndex1;
            int neighbAxis=-1;
            bool leftNeighb=false,rightNeighb=false;
			for (unsigned int d=0;d<D;++d){
                delta=(pt1[d]-pt2[d]);
                if (delta<0) {
                    //swap
                    IndexType tmp=targetIndex1;
                    targetIndex1=targetIndex2;
                    targetIndex2=targetIndex1;
                    rightNeighbor=targetIndex2;
                    leftNeighbor=targetIndex1;
                    PointType f=pt1;
                    pt1=pt2;
                    pt2=f;
                }
                if (delta!=0.0) {
                    neighbAxis=d;
                    rightNeighbor[d]+=this->m_gridSpacing[d]; 
                    if (rightNeighbor[d]<this->m_targetSize[d]){
                        rightNeighb=true;
                    }
                    leftNeighbor[d]-=this->m_gridSpacing[d]; 
                    if (leftNeighbor[d]>=0){
                        leftNeighb=true;
                    }
                    
                }
                controlPointDistance+=delta*delta;
            }

            
            if (! (rightNeighb || leftNeighb) ){
                LOG<<"ERROR, no pair has left or right neighbors. 2x2 grids dont work!"<<endl;
                LOG<<targetIndex1<<" "<<targetIndex2<<" "<<this->m_gridSpacing<<" "<<this->m_targetSize<<endl;
            }

            LabelType oldl1=this->m_baseLabelMap->GetPixel((targetIndex1));
			LabelType oldl2=this->m_baseLabelMap->GetPixel((targetIndex2));
            LabelType leftDisp, rightDisp;
            if (rightNeighb){
                rightDisp=this->m_baseLabelMap->GetPixel(rightNeighbor);
            }
            if (rightNeighb){
                leftDisp=this->m_baseLabelMap->GetPixel(leftNeighbor);
            }
            itk::Vector<LabelType,D> neighborLabels;
			double d1,d2,d0,d3;
			displacement1+=oldl1;
			displacement2+=oldl2;
            #if 0
            LOG<<targetIndex1<<" "<<targetIndex2<<" "<<leftNeighbor<<" "<<rightNeighbor<<endl;;
            LOG<<displacement1<<" "<<leftDisp<<" "<<displacement2<<" "<<rightDisp<<endl;;
            #endif
            for (unsigned int d=0;d<D;++d){
				d1=displacement1[d];
				d2=displacement2[d];
                d3=rightDisp[d];
                d0=leftDisp[d];
                double tmpDist;
                tmpDist=d2-d1;
                if (rightNeighb){
                    tmpDist+=d2-d3;
                }
                tmpDist*=tmpDist;
                rightCost+=tmpDist;
                tmpDist=d1-d2;
                if (leftNeighb){
                    tmpDist+=d1-d0;
                }
                tmpDist*=tmpDist;
                leftCost+=tmpDist;

            }
            double normaliser=1.0/2;//(int(rightNeighb)+int(leftNeighb));
			//			if (false){
            //LOG<<displacement1-displacement2<<" "<<1.0/(1+exp(-result))<<endl;
            //double result=normaliser*(leftCost+rightCost)/controlPointDistance;
            double result=normaliser*(rightNeighb*leftCost+leftNeighb*rightCost)/controlPointDistance;

            return result;
        }
    };//class
}//namespace
#endif /* POTENTIALS_H_ */
