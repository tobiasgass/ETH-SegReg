#pragma once


#include "itkImage.h"
#include "Log.h"
#include "ImageUtils.h"
#include "itkAffineTransform.h"
#include <iostream>
#include "FilterUtils.hpp"
#include "itkTransformFileReader.h"
#include "itkTransformFactoryBase.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkContinuousIndex.h"
#include <itkWarpImageFilter.h>
#include "itkVectorLinearInterpolateImageFunction.h"
#include <itkVectorNearestNeighborInterpolateImageFunction.h>
#include <itkVectorResampleImageFilter.h>
#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkDisplacementFieldCompositionFilter.h"
#include <utility>
#include <itkWarpVectorImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include "itkFixedPointInverseDeformationFieldImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkCenteredTransformInitializer.h"
#include <itkVectorResampleImageFilter.h>
#include <itkDisplacementFieldTransform.h>
#include "itkTransformFactoryBase.h"
#include "itkTransformFactory.h"
#include "itkMatrixOffsetTransformBase.h"
 
using namespace std;

template<class ImageType, class CDisplacementPrecision=float, class COutputPrecision=double>
class DeformationCache {

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename itk::AffineTransform<double,ImageType::ImageDimension> AffineTransformType;
    //typedef typename itk::Transform<double,ImageType::ImageDimension> AffineTransformType;
    typedef typename AffineTransformType::Pointer AffineTransformPointerType;
    static const int D=ImageType::ImageDimension;

    typedef  CDisplacementPrecision DisplacementPrecision;
    //typedef double DisplacementPrecision;
    typedef itk::Vector<DisplacementPrecision,D> DisplacementType;
    typedef itk::Image<DisplacementType,D> DeformationFieldType;
    typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename DeformationFieldType::ConstPointer DeformationFieldConstPointerType;

    typedef itk::Vector<COutputPrecision,D> OutputDisplacementType;
    typedef itk::Image<OutputDisplacementType,D> OutputDeformationFieldType;
    typedef typename OutputDeformationFieldType::Pointer OutputDeformationFieldPointerType;
    typedef typename OutputDeformationFieldType::ConstPointer OutputDeformationFieldConstPointerType;
    typedef map< string, map <string, DeformationFieldPointerType> > DeformationCacheType;
    typedef map< string, map <string, string> > DeformationFilenameCacheType;
private:
    bool m_cacheDeformations;
    DeformationCacheType m_deformationCache;
    DeformationFilenameCacheType m_deformationFilenameCacheType;
    
public:

    DeformationCache(){
        m_cacheDeformations=false;
    }
    
    void setCaching(bool b){
        m_cacheDeformations=b;
    }    
    
    void add(string id1, string id2, string filename){
        if (m_cacheDeformations){
            m_deformationCache[id1][id2]=ImageUtils<DeformationFieldType>::readImage(filename);
        }else{
            m_deformationFilenameCache[id1][id2]=filename;
        }
    }
    
    void add(string id1, string id2, DeformationFieldPointerType def){
        if (!m_cacheDeformations){
            LOG<<"adding deformation field to cache, when cache is set to no-caching mode!"<<endl;
        }
        m_deformationCache[id1][id2]=def;
    }
    
    bool get(string id1,string id2, DeformationFieldPointerType & def){
        if (!find(id1,id2)){
            def=NULL;
            return false;
        }
        if (m_cacheDeformations){
            def=m_deformationCache[id1][id2];
        }else{
            def=ImageUtils<DeformationFieldType>::readImage(m_deformationFilenameCache[id1][id2]);
        }
        return true;
    }

    bool find(string id1,string id2){
        if (m_cacheDeformations){
            if ( m_deformationCache.find(id1)!= m_deformationCache.end() 
                 && m_deformationCache[id1].find(id2)!=m_deformationCache[id1].end() 
                 &&  m_deformationCache[id1][id2].IsNotNull()){
                return true;
            }else{
                return false;
            }
        }else{
            if ( m_deformationFilenameCache.find(id1)!= m_deformationFilenameCache.end() 
                 && m_deformationFilenameCache[id1].find(id2)!=m_deformationFilenameCache[id1].end() 
                 &&  m_deformationFilenameCache[id1][id2]!=""){
                return true;
            }else{
                return false;
            }
        }
    }
};
