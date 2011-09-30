#ifndef __chamfer_distance_h
#define __chamfer_distance_h

#include <vector>
#include "itkImageRegionIteratorWithIndex.h"
#include "ImageUtils.h"

/*

Implementation of Two-Sweep Chamfer Distance Transform as described in:

 [1] 3D Distance Fields: A Survey of Techniques and Applications,
     Mark W.Jones, J.Andreas Berentzen, Milos Sramek,
     IEEE Transactions on Visualization, 2006

*/
template<class LabelImage, class DistanceImage, class PropagationImage = DistanceImage>
class ChamferDistanceTransform {

public:

    enum DistanceType {
        MANHATTEN, // 3 positions
        CHESSBOARD, // 9 positions
        QUASI_EUCLIDEAN, // 9 positions
        COMPLETE_EUCLIDEAN // 13 positions
    };


private:

    typedef typename DistanceImage::Pointer DistanceImagePointer;
    typedef typename LabelImage::Pointer LabelImagePointer;
    typedef typename LabelImage::ConstPointer LabelImageConstPointer;

    typedef typename PropagationImage::Pointer PropagationImagePointer;

    typedef typename LabelImage::PixelType Label;
    typedef typename DistanceImage::PixelType Distance;
    typedef typename PropagationImage::PixelType PropagationPixel;

    typedef typename DistanceImage::IndexType ImageIndex;
    typedef typename DistanceImage::SizeType ImageRegionSize;
    typedef typename DistanceImage::RegionType ImageRegion;
    typedef typename DistanceImage::OffsetType ImageOffset;
    typedef typename DistanceImage::SpacingType ImageSpacing;


    ImageRegion _largestRegion;
    ImageRegionSize _largestRegionSize;
    float _infinityDistance;
    PropagationImagePointer _propagationImage;

    struct TemplateElement {
        ImageOffset offset;
        float weight;

        TemplateElement(int x, int y, int z, float w) {
            weight = w;
            offset[0] = x;
            offset[1] = y;
            offset[2] = z;
        }
    };

    typedef std::vector<TemplateElement> ChamferTemplate;


    void addToTemplateIfPositiveWeight(
        ChamferTemplate & templ, int x, int y, int z, float weight) {

        // do nothing for zero weight
        if (weight < 0000.1 )
            return;

        templ.push_back(TemplateElement(x,y,z, weight));
    }


    // calculate vector length
    float length(float x, float y, float z = 0) {
        return sqrt(x*x + y*y + z*z);
    }

    /*
    Pre-compude chamfer forward distance template
    */
    ChamferTemplate getForwardTemplate(
        DistanceType type, ImageSpacing spacing, bool useImageSpacing
    ) {

        ChamferTemplate templ;

        float a,b,c;

        if (!useImageSpacing) {

            // set weight according to the distance type
            switch (type) {
                case MANHATTEN:
                    a = 1.0; b = 0.0; c = 0.0;
                    break;
                case CHESSBOARD:
                    a = 1.0; b = 1.0; c = 0.0;
                    break;
                case QUASI_EUCLIDEAN:
                    a = 1.0; b = sqrt(2); c = 0.0;
                    break;
                case COMPLETE_EUCLIDEAN:
                    a = 1.0; b = sqrt(2); c = sqrt(3);
                    break;
                default:
                    assert(false);
            }

            addToTemplateIfPositiveWeight(templ, -1,  0,  0, a);
            addToTemplateIfPositiveWeight(templ,  0, -1,  0, a);
            addToTemplateIfPositiveWeight(templ, -1, -1,  0, b);
            addToTemplateIfPositiveWeight(templ, -1, +1,  0, b);

            addToTemplateIfPositiveWeight(templ,  0,  0, -1, a);

            addToTemplateIfPositiveWeight(templ, -1,  0, -1, b);
            addToTemplateIfPositiveWeight(templ, +1,  0, -1, b);
            addToTemplateIfPositiveWeight(templ,  0, -1, -1, b);
            addToTemplateIfPositiveWeight(templ,  0, +1, -1, b);

            addToTemplateIfPositiveWeight(templ, -1, -1, -1, c);
            addToTemplateIfPositiveWeight(templ, -1, +1, -1, c);
            addToTemplateIfPositiveWeight(templ, +1, -1, -1, c);
            addToTemplateIfPositiveWeight(templ, +1, +1, -1, c);
        }


        else {
            // use image spacing

           float sx = spacing[0];
           float sy = spacing[1];
           float sz = spacing[2];

           switch (type) {

                // !!! missing break statements are mandantory !!!
                case COMPLETE_EUCLIDEAN:
                    addToTemplateIfPositiveWeight(templ, -1, -1, -1, length(sx,sy,sz));
                    addToTemplateIfPositiveWeight(templ, -1, +1, -1, length(sx,sy,sz));
                    addToTemplateIfPositiveWeight(templ, +1, -1, -1, length(sx,sy,sz));
                    addToTemplateIfPositiveWeight(templ, +1, +1, -1, length(sx,sy,sz));

                case QUASI_EUCLIDEAN:
                    addToTemplateIfPositiveWeight(templ, -1, -1,  0, length(sx,sy));
                    addToTemplateIfPositiveWeight(templ, -1, +1,  0, length(sx,sy));
                    addToTemplateIfPositiveWeight(templ, -1,  0, -1, length(sx,sz));
                    addToTemplateIfPositiveWeight(templ, +1,  0, -1, length(sx,sz));
                    addToTemplateIfPositiveWeight(templ,  0, -1, -1, length(sy,sz));
                    addToTemplateIfPositiveWeight(templ,  0, +1, -1, length(sy,sz));

                case MANHATTEN:
                case CHESSBOARD:
                    addToTemplateIfPositiveWeight(templ, -1,  0,  0, sx);
                    addToTemplateIfPositiveWeight(templ,  0, -1,  0, sy);
                    addToTemplateIfPositiveWeight(templ,  0,  0, -1, sz);
            }
        }

        return templ;

    }


    /*
                           /  infty       for labelImg[idx] = 0,
    InitialDistance[idx] =
                           \  0           otherwise,

    where idx denotes index of a pixel.
    */
    DistanceImagePointer initializeDistanceTransform(LabelImageConstPointer labelImg) {

        DistanceImagePointer distanceImg =
            ImageUtils<DistanceImage>::createEmpty(_largestRegionSize);

        distanceImg->SetOrigin(labelImg->GetOrigin());
        distanceImg->SetSpacing(labelImg->GetSpacing());
        distanceImg->SetDirection(labelImg->GetDirection());


        itk::ImageRegionConstIteratorWithIndex<LabelImage> it(labelImg, _largestRegion);
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

            distanceImg->SetPixel(
                it.GetIndex(),
                (it.Get() == 0) ? _infinityDistance : 0
            );
        }

        return distanceImg;
    }






    void updatePixel(
        DistanceImagePointer distMap,
        ImageIndex centerPixelIndex,
        const ChamferTemplate &templ
    ) {

        float minDistance = distMap->GetPixel(centerPixelIndex);
        ImageIndex minIndex = centerPixelIndex;

        for (unsigned i = 0; i < templ.size(); ++i) {

            const TemplateElement &elem = templ[i];

            ImageIndex idx = centerPixelIndex + elem.offset;
            if (_largestRegion.IsInside(idx)) {

                float d = elem.weight + distMap->GetPixel(idx);
                if (d < minDistance) {
                    minDistance = d;
                    minIndex = idx;
                }
            }
        }

        if (!_propagationImage.IsNull()) {
            _propagationImage->SetPixel(
                centerPixelIndex,
                _propagationImage->GetPixel(minIndex)
            );
        }
        distMap->SetPixel(centerPixelIndex, minDistance);
    }





    void forwardSweep(DistanceImagePointer distMap,ChamferTemplate templ) {

        // perform forward sweep
        itk::ImageRegionIteratorWithIndex<DistanceImage> it(distMap, _largestRegion);
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            updatePixel(distMap, it.GetIndex(), templ);
        }
    }




    void backwardSweep(DistanceImagePointer distMap,ChamferTemplate templ) {

        // reverse chamfer template
        for (unsigned elemIndex = 0; elemIndex < templ.size(); ++elemIndex) {
           for (unsigned dim=0; dim < ImageOffset::GetOffsetDimension(); ++ dim) {
               templ[elemIndex].offset[dim] *= -1;
           }
        }

        // perform backward sweep
        itk::ImageRegionIteratorWithIndex<DistanceImage> it(distMap, _largestRegion);
        for (it.GoToReverseBegin(); !it.IsAtReverseEnd(); --it) {
            updatePixel(distMap,it.GetIndex(), templ);
        }

    }



public:

    // constructor
    ChamferDistanceTransform() : _propagationImage(NULL)
    {}

    void setPropagationImage(PropagationImagePointer p) {
        _propagationImage = p;
    }

    PropagationImagePointer getPropagationImage() {
        return _propagationImage;
    }
    /*
    Input binary image,
    Output chamfer distance as a floating point image
    */
    DistanceImagePointer compute(
        LabelImageConstPointer labelImg, DistanceType type, bool useImageSpacing
    ) {

        // initialize variables used all over the computation
        _largestRegion = labelImg->GetLargestPossibleRegion();
        _largestRegionSize = _largestRegion.GetSize();
        _infinityDistance = _largestRegionSize[0] + _largestRegionSize[1]
            + _largestRegionSize[2] + 1;

        DistanceImagePointer distanceMap = initializeDistanceTransform(labelImg);

        // compute the template for the given distance transfortm type
        ChamferTemplate chamferTemplate = getForwardTemplate(
            type, labelImg->GetSpacing(),useImageSpacing);

        //std::cerr << "Chamfer Distance: forward sweep";
        forwardSweep(distanceMap, chamferTemplate);

        //        std::cerr << ", backward sweep";
        backwardSweep(distanceMap, chamferTemplate);

        // done :)
        //        std::cerr << ", done\n";
        return distanceMap;
    }


};

#endif
