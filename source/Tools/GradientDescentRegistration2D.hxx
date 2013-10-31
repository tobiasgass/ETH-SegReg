#include <iostream>
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "itkCastImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include <map>

#include <map>
#include "argstream.h"
#include <limits>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <sstream>
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkCorrelationImageToImageMetricv4.h"
#include "itkRegistrationParameterScalesFromJacobian.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"


using namespace std;

const unsigned int D=2;
typedef unsigned char Label;
typedef itk::Image< Label, D >  ImageType;
typedef  ImageType::Pointer ImageTypePointerType;
typedef itk::Image< float, D >  FloatImageType;
typedef FloatImageType::Pointer FloatImagePointerType;
typedef itk::Vector<float,D> DeformationType;
typedef itk::Image<DeformationType,D> DeformationFieldType;
typedef DeformationFieldType::Pointer DeformationFieldPointerType;


int main(int argc, char * argv [])
{


    argstream as(argc, argv);
	string target,movingationFilename,outputFilename="",outputDefFilename;
    double sigma=1;
    std::string metric="lncc";
	as >> parameter ("t", target, "image 1", true);
	as >> parameter ("m", movingationFilename, "image2", true);
	as >> parameter ("s", sigma, "lncc kernel width", true);
	as >> parameter ("o", outputFilename, "output image (file name)", false);
	as >> parameter ("T", outputDefFilename, "output Def image (file name)", false);
	as >> parameter ("metric", metric, "metric (lncc,itklncc,lsad,lssd)", false);
  
	as >> help();
	as.defaultErrorHandling();

 
    ImageType::Pointer targetImg =
        (ImageUtils<ImageType>::readImage(target));
    ImageType::Pointer movingedImg =
        (ImageUtils<ImageType>::readImage(movingationFilename));

    DeformationFieldPointerType deformation=TransfUtils<ImageType>::createEmpty(FilterUtils<ImageType>::LinearResample(targetImg,0.125,false));

    typedef itk::CorrelationImageToImageMetricv4<FloatImageType,FloatImageType> MetricType;
    //typedef itk::MeanSquaresImageToImageMetricv4<FloatImageType,FloatImageType> MetricType;
    typedef MetricType::Pointer MetricPointer;
    typedef MetricType::DerivativeType MetricDerivativeType;

    //typedef itk::RegistrationParameterScalesFromJacobian<MetricType> ScalesEstimatorType;
    typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType> ScalesEstimatorType;
    typedef ScalesEstimatorType::Pointer ScalesEstimatorPointer;

    double m_resolutionFactor=0.5;


    for (int iter=0;iter<10;++iter){

        FloatImagePointerType fimg1=FilterUtils<ImageType,FloatImageType>::LinearResample(targetImg,m_resolutionFactor,true);
        FloatImagePointerType fimg2=FilterUtils<ImageType,FloatImageType>::LinearResample(movingedImg,m_resolutionFactor,true);
        TransfUtils<ImageType,float,double>::OutputDeformationFieldPointerType dblDef=TransfUtils<ImageType,float,double>::cast(def);
        
#if 0
        typedef  itk::DisplacementFieldTransform<double, D> DisplacementFieldTransformType;
        typedef DisplacementFieldTransformType::Pointer DisplacementFieldTransformPointer;
        DisplacementFieldTransformPointer defTransf=DisplacementFieldTransformType::New();
        defTransf->SetDisplacementField(dblDef);
        //    LOG<<defTransf<<endl;
#else
        
        typedef  itk::BSplineDeformableTransform<double, D,3> BSplineDeformableTransformType;
        typedef BSplineDeformableTransformType::Pointer BSplineDeformableTransformPointer;
        BSplineDeformableTransformType::ImagePointer paramImages[D];
         
        BSplineDeformableTransformPointer defTransf=BSplineDeformableTransformType::New();
        for (int d=0;d<D;++d){
            paramImages[d]=TransfUtils<ImageType,double>::getComponent(dblDef,d);
        }
        defTransf->SetCoefficientImages(paramImages);
#endif
        //LOG<<VAR(defTransf->GetNumberOfParameters())<<endl;
        
        
        MetricPointer metric=MetricType::New();
        metric->SetFixedImage(fimg1);
        metric->SetMovingImage(fimg2);
        metric->SetTransform(defTransf);
        
        metric->Initialize();
        MetricDerivativeType derivative;
        metric->GetValueAndDerivative(value, derivative);
        
        ScalesEstimatorPointer scalesEstimator=ScalesEstimatorType::New();
        scalesEstimator->SetMetric(metric);
        ScalesEstimatorType::ScalesType scales,localScales;
        scalesEstimator->EstimateScales(scales);
        //scalesEstimator->EstimateLocalScales(derivative,localScales);
        float learningRate;
        float stepScale;
        float maxStepSize;
        //modify gradient by scales
        for (int i=0;i<derivative.size();++i){
            LOGV(4)<<derivative[i]<<"  "<<i<<" "<<i%scales.size()<<" "<<scales[i%scales.size()]<<endl;
            derivative[i]/=scales[i%scales.size()];
        }
        stepScale=scalesEstimator->EstimateStepScale(derivative);
        //estimate learning rate
        maxStepSize=scalesEstimator->EstimateMaximumStepSize();
        learningRate=maxStepSize/stepScale;
        
        LOGV(2)<<VAR(value)<<endl;
        
        
        //deriv=TransfUtils<ImageType>::createEmpty(def);
        int numberOfPixels=deriv->GetBufferedRegion().GetNumberOfPixels();
        typedef itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
        DeformationIteratorType defIt(deformation,deformation->GetLargestPossibleRegion());
        int p=0;
        for (defIt.GoToBegin();!defIt.IsAtEnd();++defIt,++p){
            DeformationType disp;
            for (int d=0;d<D;++d){
                LOGV(4)<<VAR(p+d*numberOfPixels)<<" "<<VAR(derivative[p+d*numberOfPixels])<<" "<<VAR(scales[p+d*numberOfPixels])<<" "<<VAR(stepScale)<<" "<<VAR(maxStepSize)<<" "<<VAR(learningRate)<<endl;
                disp[d]=derivative[p+d*numberOfPixels]*learningRate;///scales[p+d*numberOfPixels];
            }
            defIt.Set(disp);
            
        }
        ImageUtils<DeformationFieldType>::writeImage("derivative.mha",deriv);
    }
}
