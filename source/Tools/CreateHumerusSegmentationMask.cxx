#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <fstream>
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include "itkImageSliceIteratorWithIndex.h"
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

    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
 
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string inFile, outFile;
    double thresh=0.0;
    as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);
    as->parameter ("thresh", thresh, "fraction of pixels for inside", true);

    as->parse();
    

    ImagePointerType image = ImageUtils<ImageType>::readImage(inFile);
    
    
    ImageSliceIteratorWithIndex<ImageType> it( image, image->GetRequestedRegion() );
    
    it.SetFirstDirection(1);
    it.SetSecondDirection(0);
    
    it.GoToBegin();
    int c=0;
    std::vector<int> sliceSums(image->GetRequestedRegion().GetSize()[2],0);
    PixelType maxVal=std::numeric_limits<PixelType>::min();
    while( !it.IsAtEnd() )
        {
          
            while( !it.IsAtEndOfSlice() )
                {
                    while( !it.IsAtEndOfLine() )
                        {
                            PixelType value = it.Get();  
                            ++it;
                            maxVal=value>maxVal?value:maxVal;
                            sliceSums[c]+=value;
                        }
                    it.NextLine();
                }
            it.NextSlice();
            if (sliceSums[c] > maxVal){
                maxVal=sliceSums[c] ;
            }
            ++c;
        }
    it.GoToBegin();
    c=0;
    while( !it.IsAtEnd() )
        {
            int b=1.0*sliceSums[c]/maxVal>thresh;
            LOG<<VAR(c)<<" "<<VAR(1.0*sliceSums[c]/maxVal)<<" "<<VAR(b)<<endl;
             while( !it.IsAtEndOfSlice() )
                {
                    while( !it.IsAtEndOfLine() )
                        {
                            it.Set(b);
                            ++it;
                            
                        }
                    it.NextLine();
                }
            
            it.NextSlice();
            ++c;
        }
    
    image=FilterUtils<ImageType>::dilation(FilterUtils<ImageType>::erosion(image,2),10);

    ImageUtils<ImageType>::writeImage(outFile,image);

	return 1;
}
