#pragma once

#include "Potential-Registration-Unary.h"
 
//Abstract class interface for caching unary registration potential functions
//chaching here means that either all potentials are precomputed for all labels and nodes, or that potentials are cached for all nodes with the same label (default)

template<class TImage>
class CachingUnaryPotentialRegistration: public UnaryPotentialRegistration<TLabelMapper,TImage> {
public:
    //itk declarations
    typedef CachingUnaryPotentialRegistration            Self;
    typedef SmartPointer<Self>        Pointer;
    typedef SmartPointer<const Self>  ConstPointer;
    typedef UnaryPotentialRegistration<TLabelMapper,TImage> Superclass;

    typedef	TImage ImageType;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::ConstPointer ConstImagePointerType;
    static const int D=ImageType::ImageDimension;

    typedef typename TransfUtils<ImageType>::DisplacementType LabelType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::PointType PointType;
    typedef typename ImageType::PixelType PixelType;

    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageType::SpacingType SpacingType;
    typedef LinearInterpolateImageFunction<ImageType> InterpolatorType;
    typedef typename InterpolatorType::Pointer InterpolatorPointerType;
    typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;

    typedef typenameTransfUtils<ImageType>::DeformationFieldType LabelImageType;
    typedef typename LabelImageType::Pointer LabelImagePointerType;
    typedef typename itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
    typedef typename ImageNeighborhoodIteratorType::RadiusType RadiusType;
        
    typedef itk::TranslationTransform<double,ImageType::ImageDimension> TranslationTransformType;
    typedef typename TranslationTransformType::Pointer TranslationTransformPointerType;
    typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
        
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename FloatImageType::Pointer FloatImagePointerType;
    typedef typename itk::ImageRegionIteratorWithIndex<FloatImageType> FloatImageIteratorType;
    typedef typename itk::ImageRegionIteratorWithIndex<ImageType> ImageIteratorType;
        
    typedef  typename itk::PointSet< PixelType, D >   PointSetType;
    typedef typename  PointSetType::PointsContainer PointsContainerType;
    typedef typename  PointSetType::PointsContainerPointer PointsContainerPointer;
    typedef typename  PointSetType::PointsContainerIterator PointsContainerIterator;
    typedef itk::PointsLocator<PointsContainerType> PointsLocatorType;
    typedef typename PointsLocatorType::Pointer PointsLocatorPointerType;


protected:
    ImageNeighborhoodIteratorType m_atlasNeighborhoodIterator,m_maskNeighborhoodIterator;
    std::vector<LabelType> m_displacements;
    std::vector<FloatImagePointerType> m_potentials;
    LabelType m_currentActiveDisplacement;
    FloatImagePointerType m_currentCachedPotentials;
    ImagePointerType m_coarseImage,m_deformedAtlasImage,m_deformedMask;
    double m_averageFixedPotential,m_oldAveragePotential;
    double m_normalizationFactor;
    bool m_normalize;
    PointsContainerPointer m_atlasLandmarks,m_targetLandmarks;
    FloatImagePointerType m_unaryPotentialWeights;
public:
    itkNewMacro(Self);
    /** Standard part of every itk Object. */
    itkTypeMacro(CachingRegistrationUnaryPotential, Object);
    void SetPotentialWeights(FloatImagePointerType img){m_unaryPotentialWeights=img;}
    void SetAtlasLandmarks(PointsContainerPointer p){m_atlasLandmarks=p;}
    void SetTargetLandmarks(PointsContainerPointer p){m_targetLandmarks=p;}
    void SetAtlasLandmarksFile(string f){
        SetAtlasLandmarks(readLandmarks(f));
    }
    void SetTargetLandmarksFile(string f){
        SetTargetLandmarks(readLandmarks(f));
    }
    PointsContainerPointer readLandmarks(string f);
    void setNormalize(bool b){m_normalize=b;}
    void resetNormalize(){
        m_normalize=false;
        m_normalizationFactor=1.0;
    }
    
    void setDisplacements(std::vector<LabelType> displacements){
        m_displacements=displacements;
    }
    void setCoarseImage(ImagePointerType img){m_coarseImage=img;}
    
    virtual double getPotential(IndexType coarseIndex, unsigned int displacementLabel){
        //LOG<<"DEPRECATED BEHAVIOUR!"<<endl;
        return m_potentials[displacementLabel]->GetPixel(coarseIndex);
    }
    virtual double getPotential(IndexType coarseIndex){
        //LOG<<"NEW BEHAVIOUR!"<<endl;
        LOGV(90)<<" "<<VAR(m_currentCachedPotentials->GetLargestPossibleRegion().GetSize())<<endl;
        return  m_normalizationFactor*m_currentCachedPotentials->GetPixel(coarseIndex);
    }
    virtual double getPotential(IndexType coarseIndex, LabelType l){
        LOG<<"ERROR NEVER CALL THIS"<<endl;
        exit(0);
    }
    
    


    //#define PREDEF
    //#define LOCALSIMS
    virtual void initCaching(){};
    void cachePotentials(LabelType displacement)=0;
    virtual double getLocalPotential(IndexType targetIndex)=0;
    
    virtual FloatImagePointerType localPotentials(ImagePointerType i1, ImagePointerType i2){
        return localPotentials((ConstImagePointerType)i1,(ConstImagePointerType)i2);
    }
    virtual FloatImagePointerType localPotentials(ConstImagePointerType i1, ConstImagePointerType i2)=0;
};
