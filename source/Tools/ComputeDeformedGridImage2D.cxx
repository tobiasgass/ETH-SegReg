#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include <itkWarpImageFilter.h>
#include <itkGridForwardWarpImageFilter.h>
#include "TransformationUtils.h"
#include "ArgumentParser.h"


using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{
    ArgumentParser as(argc, argv);
    string def,outputFilename="";
    double gridResolutionFactor=0.125;
    int verbose=0;
    bool backwards=false;
    as.parameter ("def", def, "input deformation", true);
    as.parameter ("s", gridResolutionFactor, "resolutionfactor for grid creation", false);
    as.parameter ("o", outputFilename, "output image (file name)", false);
    as.option ("b", backwards, "compute inverse of deformation");
    as.parameter ("v", verbose, "verbosity level", false);
	as.parse();
	
    logSetVerbosity(verbose);
	feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Vector<double,D> DisplacementType;
    typedef Image<DisplacementType,D> DeformationFieldType;
    typedef DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef ImageType::IndexType IndexType;
    
    DeformationFieldPointerType deformation1 = ImageUtils<DeformationFieldType>::readImage(def);
    if (backwards)
        deformation1=TransfUtils<ImageType,double>::invert(deformation1);
    

#if 1

  typedef itk::Image< PixelType, D > GridImageType;

  typedef itk::GridForwardWarpImageFilter<DeformationFieldType, GridImageType> GridForwardWarperType;
    GridForwardWarperType::Pointer fwWarper = GridForwardWarperType::New();
    fwWarper->SetInput(deformation1);
    fwWarper->SetForegroundValue( itk::NumericTraits<PixelType>::max() );
    fwWarper->SetGridPixSpacing(int(gridResolutionFactor));
    fwWarper->Update();
    GridImageType::Pointer result=fwWarper->GetOutput();

#else
    typedef itk::GridForwardWarpImageFilter<DeformationFieldType,ImageType> GridForwardImageFilter;
    GridForwardImageFilter::Pointer gridWarpFilter=GridForwardImageFilter::New();
    
    typedef itk::Image< PixelType, D > GridImageType;
    GridImageType::Pointer gridImage = GridImageType::New();
    gridImage->SetRegions( deformation1->GetRequestedRegion() );
    gridImage->SetOrigin( deformation1->GetOrigin() );
    gridImage->SetSpacing( deformation1->GetSpacing() );
    gridImage->Allocate();
    gridImage->FillBuffer(0);
    
    typedef itk::ImageRegionIteratorWithIndex<GridImageType> GridImageIteratorWithIndex;
    GridImageIteratorWithIndex itergrid = GridImageIteratorWithIndex(
                                                                     gridImage, gridImage->GetRequestedRegion() );
    
    const int gridspacing(8);
    for (itergrid.GoToBegin(); !itergrid.IsAtEnd(); ++itergrid)
        {
            itk::Index<D> index = itergrid.GetIndex();
            
            if (D==2 || D==3)
                {
                    // produce an xy grid for all z
                    if ( (index[0]%gridspacing) == 0 ||
                         (index[1]%gridspacing) == 0 )
                        {
                            itergrid.Set( itk::NumericTraits<PixelType>::max() );
                        }
                }
            else
                {
                    unsigned int numGridIntersect = 0;
                    for( unsigned int dim = 0; dim < D; dim++ )
                        {
                            numGridIntersect += ( (index[dim]%gridspacing) == 0 );
                        }
                    if (numGridIntersect >= (D-1))
                        {
                            itergrid.Set( itk::NumericTraits<PixelType>::max() );
                        }
                }
        }

    typedef itk::WarpImageFilter
        < GridImageType, GridImageType, DeformationFieldType >  GridWarperType;
    GridWarperType::Pointer gridwarper = GridWarperType::New();
    gridwarper->SetInput( gridImage );
    gridwarper->SetOutputSpacing( deformation1->GetSpacing() );
    gridwarper->SetOutputOrigin( deformation1->GetOrigin() );
    gridwarper->SetOutputDirection( deformation1->GetDirection() );
    gridwarper->SetDeformationField( deformation1 );
    gridwarper->Update();
    GridImageType::Pointer result=gridwarper->GetOutput();
#endif
    ImageUtils<GridImageType>::writeImage(outputFilename,result);

    LOG<<"deformed image "<<argv[1]<<endl;
	return 1;
}
