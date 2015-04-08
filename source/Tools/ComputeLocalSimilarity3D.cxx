#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "TransformationUtils.h"
#include <itkWarpImageFilter.h>
#include <fstream>
#include "Metrics.h"

#include <itkInverseDisplacementFieldImageFilter.h>
#include "itkVectorLinearInterpolateImageFunction.h"

using namespace std;
using namespace itk;



int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  short PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;
    typedef   ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef   FloatImageType::Pointer FloatImagePointerType;
  typedef   TransfUtils<ImageType>::DisplacementType DisplacementType;
    typedef   TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef   DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef   itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;

    typedef   ImageNeighborhoodIteratorType::RadiusType RadiusType;

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Vector<float,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string inFile,inFile2,defFile="", outFile;
    int radius=3;
    string metricName="NCC";
    enum MetricType {NONE,MAD,NCC,MI,NMI,MSD};
    double m_gamma;
    
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("in2", inFile2, " filename...", true);
    as->parameter ("def", defFile, " filename...", false);
    as->parameter ("out", outFile, " filename...", true);
    as->parameter ("metric", metricName,"metric to be used for global or local weighting, valid: NONE,SAD,MSD,NCC,MI,NMI",false);
    as->parameter ("gamma", m_gamma,"scaling",false);
    as->parameter ("radius", radius,"patch radius for local metrics",false);


    as->parse();
    
    RadiusType m_patchRadius;
    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) m_patchRadius[i] = radius;

    MetricType metric;
    if (metricName=="NONE")
        metric=NONE;
    else if (metricName=="MSD")
        metric=MSD;
        else if (metricName=="MAD")
            metric=MAD;
        else if (metricName=="NCC")
            metric=NCC;
        else if (metricName=="NMI")
            metric=NMI;
        else if (metricName=="MI")
            metric=MI;
        else{
            LOG<<"don't understand "<<metricName<<", defaulting to NONE"<<endl;
            metric=NONE;
        }
    ImagePointerType img1 = ImageUtils<ImageType>::readImage(inFile);
    ImagePointerType img2 = ImageUtils<ImageType>::readImage(inFile2);

    if (defFile!=""){
        img2=TransfUtils<ImageType>::warpImage(img2,ImageUtils<DeformationFieldType>::readImage(defFile));
    }
    
    FloatImagePointerType metricImage;
    switch(metric){
    case NCC:
        metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(img2,img1,radius,m_gamma);
        break;
    case MSD:
        metricImage=Metrics<ImageType,FloatImageType>::LSSDNorm(img2,img1,radius,m_gamma);
        break;
    case MAD:
        metricImage=Metrics<ImageType,FloatImageType>::LSADNorm(img2,img1,radius,m_gamma);
        break;
    default:
        metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(img2,img1,radius,m_gamma);
    }

    ImageUtils<FloatImageType>::writeImage(outFile,metricImage);

	return 1;
}
