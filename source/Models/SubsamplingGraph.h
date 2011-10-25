/*
 * Grid.h
 *
 *  Created on: Nov 25, 2010
 *      Author: gasst
 */

#ifndef SSGRAPH_H
#define SSGRAPH_H
 
#include <vector>
#include <assert.h>
#include "itkConstNeighborhoodIterator.h"
#include <limits>
#include "Graph.h"
#include <vnl/vnl_bignum.h>
using namespace std;
/*
 * Isotropic Graph
 * Returns current/next position in a grid based on size and resolution
 */
namespace itk{
    template<class TImage, 
             class TUnaryRegistrationFunction, 
             class TPairwiseRegistrationFunction, 
             class TUnarySegmentationFunction, 
             class TPairwiseSegmentationFunction,
             class TPairwiseSegmentationRegistrationFunction,
             class TLabelMapper>
    class SubsamplingGraphModel: 
        public GraphModel<TImage,
                          TUnaryRegistrationFunction,
                          TPairwiseRegistrationFunction,
                          TUnarySegmentationFunction, 
                          TPairwiseSegmentationFunction,
                          TPairwiseSegmentationRegistrationFunction,
                          TLabelMapper>
    {
    public:
        typedef SubsamplingGraphModel Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef GraphModel<TImage,
                           TUnaryRegistrationFunction,
                           TPairwiseRegistrationFunction,
                           TUnarySegmentationFunction, 
                           TPairwiseSegmentationFunction,
                           TPairwiseSegmentationRegistrationFunction,
                           TLabelMapper> Superclass;
        itkNewMacro(Self);

        //    typedef  itk::ImageToimageFilter<TImage,TImage> Superclass;
        typedef TImage ImageType;
        typedef typename TImage::IndexType IndexType;
        typedef typename TImage::PixelType PixelType;
        typedef typename TImage::OffsetType OffsetType;
        typedef typename TImage::PointType PointType;
        typedef typename TImage::SizeType SizeType;
        typedef typename TImage::SpacingType SpacingType;
        typedef typename TImage::Pointer ImagePointerType;
        typedef typename TImage::ConstPointer ConstImagePointerType;

        typedef typename itk::ConstNeighborhoodIterator<ImageType> ConstImageNeighborhoodIteratorType;

        typedef TUnaryRegistrationFunction UnaryRegistrationFunctionType;
        typedef typename UnaryRegistrationFunctionType::Pointer UnaryRegistrationFunctionPointerType;
        typedef TPairwiseRegistrationFunction PairwiseRegistrationFunctionType;
        typedef typename PairwiseRegistrationFunctionType::Pointer PairwiseRegistrationFunctionPointerType;
        typedef TUnarySegmentationFunction UnarySegmentationFunctionType;
        typedef typename UnarySegmentationFunctionType::Pointer UnarySegmentationFunctionPointerType;
        typedef TPairwiseSegmentationFunction PairwiseSegmentationFunctionType;
        typedef typename PairwiseSegmentationFunctionType::Pointer PairwiseSegmentationFunctionPointerType;
        typedef TPairwiseSegmentationRegistrationFunction PairwiseSegmentationRegistrationFunctionType;
        typedef typename PairwiseSegmentationRegistrationFunctionType::Pointer PairwiseSegmentationRegistrationFunctionPointerType;
    
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType RegistrationLabelType;
        typedef typename itk::Image<RegistrationLabelType,ImageType::ImageDimension> RegistrationLabelImageType;
        typedef typename RegistrationLabelImageType::Pointer RegistrationLabelImagePointerType;

        typedef int SegmentationLabelType;
        typedef typename itk::Image<SegmentationLabelType,ImageType::ImageDimension> SegmentationLabelImageType;
        typedef typename SegmentationLabelImageType::Pointer SegmentationLabelImagePointerType;
    
        static const int Dimension=ImageType::ImageDimension;
    protected:
        //here we store the necessary information per node
        struct nodeInformation {
            //mapping
            std::vector<RegistrationLabelType> indexToSubsampledDisplacementMapping;
            //costs
            std::vector<double> subsampledNodeCosts;
        };
        //std::pair< std::vector<RegistrationLabelType>, std::vector<double> >
        //information for all nodes
        std::vector< nodeInformation > m_nodeMappingInfo;
        vnl_matrix_fixed< double, 6, 9 > constA;
        int gridSize;
        int nRegistrationLabels;
    public:

        void Init(){
            gridSize=LabelMapperType::nDisplacementSamples*2+1;
            const double A[9][6] = {  // constant matrix for 2D paraboloid fitting
#if 0           
                {1,-1,-1, 1, 1, 1},
                {1,-1, 0, 0, 1, 0},
                {1,-1, 1,-1, 1, 1},
                {1, 0,-1, 0, 0, 1},
                {1, 0, 0, 0, 0, 0},
                {1, 0, 1, 0, 0, 1},
                {1, 1,-1,-1, 1, 1},
                {1, 1, 0, 0, 1, 0},
                {1, 1, 1, 1, 1, 1}};
#else
            {1,-1,-1, 1, 1, 1},
                {1, 0,-1, 0, 0, 1},
                    {1, 1,-1,-1, 1, 1},
                        {1,-1, 0, 0, 1, 0},
                            {1, 0, 0, 0, 0, 0},
                                {1, 1, 0, 0, 1, 0},
                                    {1,-1, 1,-1, 1, 1},
                                        {1, 0, 1, 0, 0, 1},
                                            {1, 1, 1, 1, 1, 1}};
#endif
        vnl_matrix_fixed< double, 9, 6 > tempM;
        for(int ii=0; ii<9; ii++){
            for(int jj=0; jj<6; jj++){
                tempM.put(ii, jj, A[ii][jj] );
            }
        }
            
        // constA = inv(A'*A)*A'    // according to Eq(13) in Sub-sample Disp..., Zahiri, Goksel, Salcudean: TMI 2010
        constA = vnl_matrix_inverse<double>( vnl_transpose(tempM) * tempM ) * vnl_transpose(tempM);
                                                                        
        int nNewSamples=(gridSize-2)*(gridSize-2);
        cout<<nNewSamples<<endl;
        nRegistrationLabels=nNewSamples;
        m_nodeMappingInfo= std::vector< nodeInformation >(0);
        //computing new subsampled labels and label costs
        for (int r=0;r<this->m_nRegistrationNodes;++r){
            vnl_matrix< double> originalRegistrationCosts(gridSize,gridSize);
            for (int l=0;l<this->m_nDisplacementLabels;++l){
                double tmp=Superclass::getUnaryRegistrationPotential(r,l);
                RegistrationLabelType label=LabelMapperType::getLabel(l);
                originalRegistrationCosts.put(label[0]+(gridSize-1)/2,label[1]+(gridSize-1)/2,tmp);
                //if (r==0) std::cout<<r<<" "<<l<<" "<<label<<" "<< tmp<<endl;
            }
                
            // now compute novel mapping and costs and store in m_nodeMappingInfo
            //for the beginning it might be safe to leave m_nRegistrationLabels at its old value. otherwise we'd need to do some copying/overwritng to be safe
            m_nodeMappingInfo.push_back(Subsample(originalRegistrationCosts,nNewSamples));

        }
    }
        virtual int nRegLabels(){
            return nRegistrationLabels;
        }

    nodeInformation Subsample( vnl_matrix< double> & gridCosts, int nNewSamples)
    {
        assert(Dimension==2);
        //	int gridSize=gridCosts.columns();
        nodeInformation nodeInfo;
        // Find all requested labels
        bool someMin=false;
        for(int labelNum=0; labelNum<nNewSamples; labelNum++)
            {
                // center of subsample grid (local minima)
                int locMin[Dimension];
                //??? locMin = gridCosts.arg_min();
                // Why doesn't this work, old vnl version?  Do the long way below...
                double *g = gridCosts.data_block();
                double minVal = vnl_bignum("+Infinity");   int ind=0;
                for(int ii = 0; ii<gridCosts.size(); ii++){
                    if (g[ii]<minVal)
                        {
                            ind = ii;
                            minVal = g[ii];
                        }
                }
                locMin[0] = ind / gridSize;
                locMin[1] = ind - gridSize*locMin[0]; 
                locMin[0] = max(1, min(gridSize-2,locMin[0]) );  // bracket the subGrid center inside the grid
                locMin[1] = max(1, min(gridSize-2,locMin[1]) );

                // vector for subsample grid ( center + 8 neighbour costs )
                vnl_matrix_fixed< double, 9, 1> subGridCosts;
                // copy the subsample grid (minus 1's are in order to get the corner of 3x3 window)
                gridCosts.extract(3,3,locMin[0]-1,locMin[1]-1).copy_out( subGridCosts.data_block() );

                //check whether we found a local minimum
                g = subGridCosts.data_block();
                minVal = vnl_bignum("+Infinity");    ind=0;
                for(int ii = 0; ii<subGridCosts.size(); ii++){
                    if (g[ii]<minVal)
                        {
                            ind = ii;
                            minVal = g[ii];
                        }
                }
                if (1 || ind == subGridCosts.size()/2){
 
                    vnl_matrix_fixed<double,6,1> aMat = constA * subGridCosts;
                    double *a = aMat.data_block();

                    // There is a closed-form for the below inversion operation [keeping this for now for easier templating for 3D]
                    vnl_matrix_fixed<double,2,2> temp2x2;
                    temp2x2.put(0, 0, 2*a[4] );
                    temp2x2.put(0, 1,   a[3] );
                    temp2x2.put(1, 0,   a[3] );
                    temp2x2.put(1, 1, 2*a[5] );
                    vnl_matrix_fixed<double,2,1> dispMat = -(vnl_matrix_inverse<double>( temp2x2 ) * aMat.extract(2,1,1,0));
                    double *d = dispMat.data_block();

                    // bracket the displacement inside the subGrid [-1,1]
                    d[0] = max(-1.0, min(1.0,d[0]) );
                    d[1] = max(-1.0, min(1.0,d[1]) );

                    // Be careful: I make certain assumption on the current vector contents below with no bounds checking
                    double gridCenterOffset = (gridSize-1)/2;
                    RegistrationLabelType label;
                    label[0] = d[0] + locMin[0] - gridCenterOffset;
                    label[1] = d[1] + locMin[1] - gridCenterOffset;
                    nodeInfo.indexToSubsampledDisplacementMapping.push_back( label );
                    nodeInfo.subsampledNodeCosts.push_back( a[0] + a[1]*d[0] + a[2]*d[1] + a[3]*d[0]*d[1] + a[4]*d[0]*d[0] + a[5]*d[1]*d[1] );
                }
                else{
                    RegistrationLabelType label;
                    label.Fill(0);
                    nodeInfo.indexToSubsampledDisplacementMapping.push_back( label );
                    nodeInfo.subsampledNodeCosts.push_back( 100000 );

                }
                // Hack :) to remove this local minimum in order to get other possible labels :: Fix Later!
                // !!! Modifying pass by reference !!! assuming the cost grid will not be needed outside this function
                g[ind] = gridCosts.max_value();
            }
        return nodeInfo;
    }
#if 0
    nodeInformation Subsample( vnl_matrix< double> & gridCosts, int nNewSamples)
    {
        assert(Dimension==2);
        double gridCenterOffset = (gridSize-1)/2;

        nodeInformation nodeInfo;
        // Find all requested labels
        bool someMin=false;
        for(int labelNum=0; labelNum<nNewSamples; labelNum++)
            {
                // center of subsample grid (local minima)
                int locMin[Dimension];
                //??? locMin = gridCosts.arg_min();
                // Why doesn't this work, old vnl version?  Do the long way below...
                double *g = gridCosts.data_block();
                double minVal = vnl_bignum("+Infinity");   int ind=0;
                for(int ii = 0; ii<gridCosts.size(); ii++){
                    if (g[ii]<minVal)
                        {
                            ind = ii;
                            minVal = g[ii];
                        }
                }
                locMin[0] = max(1.0,min(1.0*ind / gridSize,gridCenterOffset));
                locMin[1] = max(1.0,min(1.0*ind - gridSize*locMin[0],gridCenterOffset)); 
                    
                // vector for subsample grid ( center + 8 neighbour costs )
                vnl_matrix_fixed< double, 9, 1> subGridCosts;
                // copy the subsample grid (minus 1's are in order to get the corner of 3x3 window)
                gridCosts.extract(3,3,locMin[0]-1,locMin[1]-1).copy_out( subGridCosts.data_block() );

                vnl_matrix_fixed<double,6,1> aMat = constA * subGridCosts;
                double *a = aMat.data_block();

                // There is a closed-form for the below inversion operation [keeping this for now for easier templating for 3D]
                vnl_matrix_fixed<double,2,2> temp2x2;
                temp2x2.put(0, 0, 2*a[4] );
                temp2x2.put(0, 1,   a[3] );
                temp2x2.put(1, 0,   a[3] );
                temp2x2.put(1, 1, 2*a[5] );
                vnl_matrix_fixed<double,2,1> dispMat = -(vnl_matrix_inverse<double>( temp2x2 ) * aMat.extract(2,1,1,0));
                double *d = dispMat.data_block();
                d[0]=max(-1.0,min(d[0],1.0));
                d[1]=max(-1.0,min(d[1],1.0));
                // Be careful: I make certain assumption on the current vector contents below with no bounds checking
                RegistrationLabelType label;
                label[0]=d[0] + locMin[0] - gridCenterOffset;
                label[1]=d[1] + locMin[1] - gridCenterOffset;
                nodeInfo.indexToSubsampledDisplacementMapping.push_back(label );
                if (label[0]<-gridCenterOffset or label[0]>gridCenterOffset or label[1]<-gridCenterOffset or label[1]>gridCenterOffset){
                    nodeInfo.subsampledNodeCosts.push_back(1000000);
                }
                else{
                    nodeInfo.subsampledNodeCosts.push_back( a[0] + a[1]*d[0] + a[2]*d[1] + a[3]*d[0]*d[1] + a[4]*d[0]*d[0] + a[5]*d[1]*d[1] );
                    someMin=true;
                }
                // Hack :) to remove this local minimum in order to get other possible labels :: Fix Later!
                // !!! Modifying pass by reference !!! assuming the cost grid will not be needed outside this function
                g[ind] = gridCosts.max_value();
                    
            }
        assert(someMin);
        return nodeInfo;
    }
#endif
      
    //this is now all interfacing stuff i needed to rewrite because of the optimizer, don't bother :)
    double getUnaryRegistrationPotential(int nodeIndex,int labelIndex){
        double result=m_nodeMappingInfo[nodeIndex].subsampledNodeCosts[labelIndex];
        cout<<nodeIndex<<" "<<labelIndex<<" "<<result<<endl;
        return result;
    }
     
    double getPairwiseRegistrationPotential(int nodeIndex1, int nodeIndex2, int labelIndex1, int labelIndex2){
        IndexType graphIndex1=this->getImageIndexFromCoarseGraphIndex(nodeIndex1);
        RegistrationLabelType l1=m_nodeMappingInfo[nodeIndex1].indexToSubsampledDisplacementMapping[labelIndex1];
        l1=LabelMapperType::scaleDisplacement(l1,this->getDisplacementFactor());
        IndexType graphIndex2=this->getImageIndexFromCoarseGraphIndex(nodeIndex2);
        RegistrationLabelType l2=m_nodeMappingInfo[nodeIndex2].indexToSubsampledDisplacementMapping[labelIndex2];
        l2=LabelMapperType::scaleDisplacement(l2,this->getDisplacementFactor());
        return this->m_pairwiseRegFunction->getPotential(graphIndex1, graphIndex2, l1,l2)/this->m_nRegEdges;
    };
      
    //#define MULTISEGREGNEIGHBORS
    inline double getPairwiseRegSegPotential(int nodeIndex1, int nodeIndex2, int labelIndex1, int segmentationLabel){
    
        IndexType imageIndex=this->getImageIndex(nodeIndex2);
        //compute distance between center index and patch index
        double weight=1.0;
        IndexType graphIndex=this->getImageIndexFromCoarseGraphIndex(nodeIndex1);
        double dist=1;
        for (unsigned int d=0;d<this->m_dim;++d){
            dist*=1.0-fabs((1.0*graphIndex[d]-imageIndex[d])/(this->m_gridPixelSpacing[d]));
        }
        weight=dist;
        RegistrationLabelType registrationLabel=m_nodeMappingInfo[nodeIndex1].indexToSubsampledDisplacementMapping[labelIndex1];
        registrationLabel=LabelMapperType::scaleDisplacement(registrationLabel,this->getDisplacementFactor());
        return weight*this->m_pairwiseSegRegFunction->getPotential(imageIndex,imageIndex,registrationLabel,segmentationLabel)/this->m_nSegRegEdges;
    }

        
    virtual RegistrationLabelImagePointerType getDeformationImage(std::vector<int>  labels){
        RegistrationLabelImagePointerType result=RegistrationLabelImageType::New();
        typename RegistrationLabelImageType::RegionType region;
        region.SetSize(this->m_gridSize);
        result->SetRegions(region);
        result->SetSpacing(this->m_gridSpacing);
        result->SetDirection(this->m_fixedImage->GetDirection());
        result->SetOrigin(this->m_origin);
        result->Allocate();
        typename itk::ImageRegionIterator<RegistrationLabelImageType> it(result,region);
        unsigned int i=0;
        for (it.GoToBegin();!it.IsAtEnd();++it,++i){
            assert(i<labels.size());
            RegistrationLabelType l=m_nodeMappingInfo[i].indexToSubsampledDisplacementMapping[labels[i]];
            std::cout<<"FINAL :"<<i<<" "<<labels[i]<<" "<<l<<" "<<m_nodeMappingInfo[i].subsampledNodeCosts[labels[i]]<<endl;
            l=LabelMapperType::scaleDisplacement(l,this->getDisplacementFactor());
            it.Set(l);
        }
        assert(i==(labels.size()));
        return result;
    }
   
}; //GraphModel

}//namespace

#endif /* GRIm_dim_H_ */
