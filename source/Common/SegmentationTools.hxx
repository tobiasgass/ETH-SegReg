#pragma once

#include "itkImage.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include "itkLabelOverlapMeasuresImageFilter.h"


template <class ImageType>
class SegmentationTools{
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::SizeType SizeType;
    static const unsigned int D=ImageType::ImageDimension;

private:
    
    static ImagePointerType selectLabel(ImagePointerType img, int l){
        ImagePointerType result=ImageUtils<ImageType>::createEmpty(img);
        typedef typename itk::ImageRegionIterator<ImageType> IteratorType;
        IteratorType it1(img,img->GetLargestPossibleRegion());
        IteratorType it2(result,img->GetLargestPossibleRegion());
        for (it1.GoToBegin(),it2.GoToBegin();!it1.IsAtEnd();++it1,++it2){
            it2.Set(it1.Get()==l);
        }
        return result;
    }
public:
    static void computeOverlap(ImagePointerType groundTruthImg, ImagePointerType segmentedImg, double & dice, double &mean, double & hd, int evalLabel=-1, bool evalDistance=true,bool connectedComponent=false){
        
        groundTruthImg=FilterUtils<ImageType>::NNResample(groundTruthImg,segmentedImg,false);

        if (evalLabel>-1){
            groundTruthImg=selectLabel(groundTruthImg,evalLabel);   
            segmentedImg=selectLabel(segmentedImg,evalLabel);
        }
      
        typedef typename ImageType::ConstPointer ConstType;
        if (connectedComponent){  
            ImagePointerType testImage = FilterUtils<ImageType>::relabelComponents(segmentedImg);
            //ImageUtils<ImageType>::writeImage("relabel.png", FilterUtils<ImageType>::normalize(testImage));

            typedef typename itk::MinimumMaximumImageCalculator <ImageType>
                ImageCalculatorFilterType;
            typedef typename itk::ConnectedComponentImageFilter<ImageType,ImageType>  ConnectedComponentImageFilterType;
            typename ConnectedComponentImageFilterType::Pointer filter =
                ConnectedComponentImageFilterType::New();
            filter->SetInput(segmentedImg);
            filter->Update();
    
            typedef typename itk::LabelShapeKeepNObjectsImageFilter< ImageType > LabelShapeKeepNObjectsImageFilterType;
            typename LabelShapeKeepNObjectsImageFilterType::Pointer labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
            labelShapeKeepNObjectsImageFilter->SetInput( filter->GetOutput() );
            labelShapeKeepNObjectsImageFilter->SetBackgroundValue( 0 );
            labelShapeKeepNObjectsImageFilter->SetNumberOfObjects( 1);
            labelShapeKeepNObjectsImageFilter->SetAttribute( LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
            labelShapeKeepNObjectsImageFilter->Update();
            //segmentedImg =  labelShapeKeepNObjectsImageFilter->GetOutput();     //;//f->GetOutput();// FilterUtils<ImageType>::binaryThresholding(labelShapeKeepNObjectsImageFilter->GetOutput(),1,10000);//filter->GetOutput(),1,1);
            segmentedImg =  FilterUtils<ImageType>::binaryThresholdingLow(labelShapeKeepNObjectsImageFilter->GetOutput(), 1);     //;//f->GetOutput();// FilterUtils<ImageType>::binaryThresholding(labelShapeKeepNObjectsImageFilter->GetOutput(),1,10000);//filter->GetOutput(),1,1);
          
        }
   
   
        if (evalDistance){
            typedef typename itk::HausdorffDistanceImageFilter<ImageType, ImageType> HausdorffDistanceFilterType;
            typedef typename HausdorffDistanceFilterType::Pointer HDPointerType;
            HDPointerType hdFilter=HausdorffDistanceFilterType::New();
            if (FilterUtils<ImageType>::getMax(segmentedImg)==0){
                //segmentedImg->FillBuffer(1.0);
                typename ImageType::IndexType idx;
                idx.Fill(0);
                segmentedImg->SetPixel(idx,1);
            }
            hdFilter->SetInput1(groundTruthImg);
            hdFilter->SetInput2(segmentedImg);
            hdFilter->SetUseImageSpacing(true);
            hdFilter->Update();
            mean=hdFilter->GetAverageHausdorffDistance();
            hd=hdFilter->GetHausdorffDistance();
        }

        typedef typename itk::LabelOverlapMeasuresImageFilter<ImageType> OverlapMeasureFilterType;
        typename OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
        filter->SetSourceImage(groundTruthImg);
        filter->SetTargetImage(segmentedImg);
        filter->Update();
        dice=filter->GetDiceCoefficient();
    }


};
