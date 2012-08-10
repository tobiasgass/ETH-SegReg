#pragma once

#include <itkImageToImageFilter.h>
#include <itkWarpVectorImageFilter.h>
#include <itkAddImageFilter.h>



template <class TInputImage, class TOutputImage>
class  DisplacementFieldCompositionFilter :
{
public:
    /** Standard class typedefs. */
    typedef DisplacementFieldCompositionFilter             Self;
    typedef ImageToImageFilter<TInputImage,TOutputImage>   Superclass;
    typedef SmartPointer<Self>                             Pointer;
    typedef SmartPointer<const Self>                       ConstPointer;

    /** InputImage type. */
    typedef TInputImage         DisplacementFieldType;
    typedef TInputImage::Pointer      DisplacementFieldPointer;
    typedef TInputImage::ConstPointer DisplacementFieldConstPointer;
    
    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro( DisplacementFieldImageFilter );

    /** Warper type. */
    typedef WarpVectorImageFilter<DisplacementFieldType,
                                  DisplacementFieldType,DisplacementFieldType>     VectorWarperType;
    typedef typename VectorWarperType::Pointer
    VectorWarperPointer;
  
 


protected:
    DisplacementFieldCompositionFilter();
    ~DisplacementFieldCompositionFilter() {};
  
    /** Adder type. */
    typedef AddImageFilter<DisplacementFieldType,DisplacementFieldType,
                           DisplacementFieldType>                           AdderType;
    typedef typename AdderType::Pointer                  AdderPointer;

    static DisplacementFieldPointer compose(DisplacementFieldPointer leftField, DisplacementFieldPointer rightField){
        VectorWarperPointer        m_Warper;
        AdderPointer               m_Adder;
        m_Warper = VectorWarperType::New();
        m_Adder = AdderType::New();
        
        // Setup the default interpolator
        typedef VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<
            DisplacementFieldType,double> DefaultFieldInterpolatorType;
        m_Warper->SetInterpolator( DefaultFieldInterpolatorType::New() );
        
        // Setup the adder to not be inplace
        m_Adder->InPlaceOff();
        // Set up mini-pipeline
        m_Warper->SetInput( leftField );
        m_Warper->SetDisplacementField( rightField );
        m_Warper->SetOutputOrigin( rightField->GetOrigin() );
        m_Warper->SetOutputSpacing( rightField->GetSpacing() );
        m_Warper->SetOutputDirection( rightField->GetDirection() );
        
        m_Adder->SetInput1( m_Warper->GetOutput() );
        m_Adder->SetInput2( rightField );
        
        //m_Adder->GetOutput()->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );
        m_Adder->Update();

        return m_Adder->GetOutput();
    }

};
