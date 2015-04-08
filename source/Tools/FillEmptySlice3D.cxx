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
     as->parameter ("in", inFile, " filename...", true);
    as->parameter ("out", outFile, " filename...", true);

    as->parse();
    

    ImagePointerType image = ImageUtils<ImageType>::readImage(inFile);
    
    
    ImageSliceIteratorWithIndex<ImageType> it( image, image->GetRequestedRegion() );
    
    it.SetFirstDirection(1);
    it.SetSecondDirection(0);
    
    it.GoToBegin();
    int c=0;
    while( !it.IsAtEnd() )
        {
            PixelType minVal=std::numeric_limits<PixelType>::max();
            PixelType maxVal=std::numeric_limits<PixelType>::min();
            ImageSliceIteratorWithIndex<ImageType> tmpIt(it);
            while( !it.IsAtEndOfSlice() )
                {
                    while( !it.IsAtEndOfLine() )
                        {
                            PixelType value = it.Get();  
                            ++it;
                            minVal=value<minVal?value:minVal;
                            maxVal=value>maxVal?value:maxVal;
                        }
                    it.NextLine();
                }
            
            if (minVal==maxVal){
                //empy slice!
                std::cout<<"Replacing empty slice nr"<<c<<"; which has min/max value of : "<<minVal<<endl;
                ImageSliceIteratorWithIndex<ImageType> prevIt(tmpIt);
                ImageSliceIteratorWithIndex<ImageType> nextIt(tmpIt);
                prevIt.PreviousSlice();
                nextIt.NextSlice();
                while( !tmpIt.IsAtEndOfSlice() )
                    {
                        while( !tmpIt.IsAtEndOfLine() )
                            {
                                //std::cout<<tmpIt.Get()<<" "<<prevIt.Get()<<" "<<nextIt.Get()<<endl;
                                tmpIt.Set(0.5*(prevIt.Get()+nextIt.Get() ) );
                                ++tmpIt; ++prevIt ; ++nextIt;
                            }
                        tmpIt.NextLine();
                        prevIt.NextLine();
                        nextIt.NextLine();
                    }
                

            }
            it.NextSlice();
            ++c;
        }
    
  

    ImageUtils<ImageType>::writeImage(outFile,image);

	return 1;
}
