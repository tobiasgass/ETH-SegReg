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
#include <vnl/vnl_random.h>

// Method of center fitting
#define ISFITCENTER      0
#define CENTERFITHACK    0
#define SCALECENTER      10

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
        struct NodeInformation {
            //mapping
            std::vector<RegistrationLabelType> indexToSubsampledDisplacementMapping;
            //costs
            std::vector<double> subsampledNodeCosts;
        };
        struct GridInformation {
            int dim[3];  // dimZ=0 (or undefined) for 2D
            std::vector<double> costs;
        };
        
        //information for all nodes
        std::vector< NodeInformation > m_nodeMappingInfo;
        vnl_matrix_fixed< double, 4*Dimension-2, 3*3*(Dimension==3?3:1) > constA;  //template for Dimension=[2,3]
        //vnl_matrix_fixed< double, 6, 9 > constA;
        int gridSize;
        int nRegistrationLabels;
    public:

        virtual void Init(){
            gridSize=LabelMapperType::nDisplacementSamples*2+1;
            double A2[9][6] = {0};    // constant matrix for 2D paraboloid fitting
            A2[0][0] = 1.0;	  A2[0][1] = -1.0;	  A2[0][2] = -1.0;	  A2[0][3] = 1.0;	  A2[0][4] = 1.0;	  A2[0][5] = 1.0;
            A2[1][0] = 1.0;	  A2[1][2] = -1.0;	  A2[1][5] = 1.0;
            A2[2][0] = 1.0;	  A2[2][1] = 1.0;	  A2[2][2] = -1.0;	  A2[2][3] = -1.0;	  A2[2][4] = 1.0;	  A2[2][5] = 1.0;
            A2[3][0] = 1.0;	  A2[3][1] = -1.0;	  A2[3][4] = 1.0;
            A2[4][0] = 1.0;
            A2[5][0] = 1.0;	  A2[5][1] = 1.0;	  A2[5][4] = 1.0;
            A2[6][0] = 1.0;	  A2[6][1] = -1.0;	  A2[6][2] = 1.0;	  A2[6][3] = -1.0;	  A2[6][4] = 1.0;	  A2[6][5] = 1.0;
            A2[7][0] = 1.0;	  A2[7][2] = 1.0;	  A2[7][5] = 1.0;
            A2[8][0] = 1.0;	  A2[8][1] = 1.0;	  A2[8][2] = 1.0;	  A2[8][3] = 1.0;	  A2[8][4] = 1.0;	  A2[8][5] = 1.0;

            // syms x y z real;  f = inline([1,x,y,z,x*y,x*z,y*z,x*x,y*y,z*z],'x','y','z');
            // [x,y,z] = ndgrid(-1:1,-1:1,-1:1);  A=[];
            // for i=1:numel(x), A(i,:)=f(x(i),y(i),z(i)); end; A3=sym(A),  regexprep(ccode(A3),'\n','\t')
            double A3[27][10] = {0};    // constant matrix for 2D paraboloid fitting
            A3[0][0] = 1.0;	  A3[0][1] = -1.0;	  A3[0][2] = -1.0;	  A3[0][3] = -1.0;	  A3[0][4] = 1.0;	  A3[0][5] = 1.0;	  A3[0][6] = 1.0;	  A3[0][7] = 1.0;	  A3[0][8] = 1.0;	  A3[0][9] = 1.0;
            A3[1][0] = 1.0;	  A3[1][2] = -1.0;	  A3[1][3] = -1.0;	  A3[1][6] = 1.0;	  A3[1][8] = 1.0;	  A3[1][9] = 1.0;
            A3[2][0] = 1.0;	  A3[2][1] = 1.0;	  A3[2][2] = -1.0;	  A3[2][3] = -1.0;	  A3[2][4] = -1.0;	  A3[2][5] = -1.0;	  A3[2][6] = 1.0;	  A3[2][7] = 1.0;	  A3[2][8] = 1.0;	  A3[2][9] = 1.0;
            A3[3][0] = 1.0;	  A3[3][1] = -1.0;	  A3[3][3] = -1.0;	  A3[3][5] = 1.0;	  A3[3][7] = 1.0;	  A3[3][9] = 1.0;	  A3[4][0] = 1.0;	  A3[4][3] = -1.0;	  A3[4][9] = 1.0;
            A3[5][0] = 1.0;	  A3[5][1] = 1.0;	  A3[5][3] = -1.0;	  A3[5][5] = -1.0;	  A3[5][7] = 1.0;	  A3[5][9] = 1.0;
            A3[6][0] = 1.0;	  A3[6][1] = -1.0;	  A3[6][2] = 1.0;	  A3[6][3] = -1.0;	  A3[6][4] = -1.0;	  A3[6][5] = 1.0;	  A3[6][6] = -1.0;	  A3[6][7] = 1.0;	  A3[6][8] = 1.0;	  A3[6][9] = 1.0;
            A3[7][0] = 1.0;	  A3[7][2] = 1.0;	  A3[7][3] = -1.0;	  A3[7][6] = -1.0;	  A3[7][8] = 1.0;	  A3[7][9] = 1.0;
            A3[8][0] = 1.0;	  A3[8][1] = 1.0;	  A3[8][2] = 1.0;	  A3[8][3] = -1.0;	  A3[8][4] = 1.0;	  A3[8][5] = -1.0;	  A3[8][6] = -1.0;	  A3[8][7] = 1.0;	  A3[8][8] = 1.0;	  A3[8][9] = 1.0;
            A3[9][0] = 1.0;	  A3[9][1] = -1.0;	  A3[9][2] = -1.0;	  A3[9][4] = 1.0;	  A3[9][7] = 1.0;	  A3[9][8] = 1.0;
            A3[10][0] = 1.0;	  A3[10][2] = -1.0;	  A3[10][8] = 1.0;
            A3[11][0] = 1.0;	  A3[11][1] = 1.0;	  A3[11][2] = -1.0;	  A3[11][4] = -1.0;	  A3[11][7] = 1.0;	  A3[11][8] = 1.0;
            A3[12][0] = 1.0;	  A3[12][1] = -1.0;	  A3[12][7] = 1.0;
            A3[13][0] = 1.0;
            A3[14][0] = 1.0;	  A3[14][1] = 1.0;	  A3[14][7] = 1.0;
            A3[15][0] = 1.0;	  A3[15][1] = -1.0;	  A3[15][2] = 1.0;	  A3[15][4] = -1.0;	  A3[15][7] = 1.0;	  A3[15][8] = 1.0;
            A3[16][0] = 1.0;	  A3[16][2] = 1.0;	  A3[16][8] = 1.0;
            A3[17][0] = 1.0;	  A3[17][1] = 1.0;	  A3[17][2] = 1.0;	  A3[17][4] = 1.0;	  A3[17][7] = 1.0;	  A3[17][8] = 1.0;
            A3[18][0] = 1.0;	  A3[18][1] = -1.0;	  A3[18][2] = -1.0;	  A3[18][3] = 1.0;	  A3[18][4] = 1.0;	  A3[18][5] = -1.0;	  A3[18][6] = -1.0;	  A3[18][7] = 1.0;	  A3[18][8] = 1.0;	  A3[18][9] = 1.0;
            A3[19][0] = 1.0;	  A3[19][2] = -1.0;	  A3[19][3] = 1.0;	  A3[19][6] = -1.0;	  A3[19][8] = 1.0;	  A3[19][9] = 1.0;
            A3[20][0] = 1.0;	  A3[20][1] = 1.0;	  A3[20][2] = -1.0;	  A3[20][3] = 1.0;	  A3[20][4] = -1.0;	  A3[20][5] = 1.0;	  A3[20][6] = -1.0;	  A3[20][7] = 1.0;	  A3[20][8] = 1.0;	  A3[20][9] = 1.0;
            A3[21][0] = 1.0;	  A3[21][1] = -1.0;	  A3[21][3] = 1.0;	  A3[21][5] = -1.0;	  A3[21][7] = 1.0;	  A3[21][9] = 1.0;
            A3[22][0] = 1.0;	  A3[22][3] = 1.0;	  A3[22][9] = 1.0;
            A3[23][0] = 1.0;	  A3[23][1] = 1.0;	  A3[23][3] = 1.0;	  A3[23][5] = 1.0;	  A3[23][7] = 1.0;	  A3[23][9] = 1.0;
            A3[24][0] = 1.0;	  A3[24][1] = -1.0;	  A3[24][2] = 1.0;	  A3[24][3] = 1.0;	  A3[24][4] = -1.0;	  A3[24][5] = -1.0;	  A3[24][6] = 1.0;	  A3[24][7] = 1.0;	  A3[24][8] = 1.0;	  A3[24][9] = 1.0;
            A3[25][0] = 1.0;	  A3[25][2] = 1.0;	  A3[25][3] = 1.0;	  A3[25][6] = 1.0;	  A3[25][8] = 1.0;	  A3[25][9] = 1.0;
            A3[26][0] = 1.0;	  A3[26][1] = 1.0;	  A3[26][2] = 1.0;	  A3[26][3] = 1.0;	  A3[26][4] = 1.0;	  A3[26][5] = 1.0;	  A3[26][6] = 1.0;	  A3[26][7] = 1.0;	  A3[26][8] = 1.0;	  A3[26][9] = 1.0;

            // *** Initialization of variables
            vnl_matrix< double > tempM;
            switch(Dimension)
                {
                case 2:
                    tempM.set_size(9,6);
                    for(int ii=0; ii<9; ii++)  // there must be an easier matrix initialization than this nested for-loops
                        for(int jj=0; jj<6; jj++)
                            tempM.put(ii, jj, A2[ii][jj] );
                    break;
                case 3:
                    tempM.set_size(27,10);
                    for(int ii=0; ii<27; ii++)  // there must be an easier matrix initialization than this nested for-loops
                        for(int jj=0; jj<10; jj++)
                            tempM.put(ii, jj, A3[ii][jj] );
                    break;
                default:
                    assert(0);
                }

            // constA = inv(A'*A)*A'    // according to Eq(13) in Sub-sample Disp..., Zahiri, Goksel, Salcudean: TMI 2010
            constA = vnl_matrix_inverse<double>( vnl_transpose(tempM) * tempM ) * vnl_transpose(tempM);
            
            int nNewSamples=this->m_config.nSubsamples;//(gridSize-2)*(gridSize-2);
            cout<<nNewSamples<<endl;
            nRegistrationLabels=nNewSamples;
            m_nodeMappingInfo= std::vector< NodeInformation >(0);
            //computing new subsampled labels and label costs
            for (int r=0;r<this->m_nRegistrationNodes;++r){
                GridInformation originalRegistrationCosts;
                originalRegistrationCosts.dim[0] = gridSize;   originalRegistrationCosts.dim[1] = gridSize;   originalRegistrationCosts.dim[2] = gridSize;
                IndexType imageIndex=this->getImageIndexFromCoarseGraphIndex(r);
                for (int l=0;l<this->m_nDisplacementLabels;++l){
                    double tmp=Superclass::getUnaryRegistrationPotential(r,l);
                    RegistrationLabelType label=LabelMapperType::getLabel(l);
                    if (l==1) assert(label[0]==-(gridSize-1)/2+1 &&label[1]==-(gridSize-1)/2);
                    originalRegistrationCosts.costs.push_back(tmp);
                    //if (r==0) std::cout<<r<<" "<<l<<" "<<label<<" "<< tmp<<endl;
                }
                
                // now compute novel mapping and costs and store in m_nodeMappingInfo
                //for the beginning it might be safe to leave m_nRegistrationLabels at its old value. otherwise we'd need to do some copying/overwritng to be safe
                m_nodeMappingInfo.push_back(Subsample(originalRegistrationCosts,nNewSamples,imageIndex));

            }
        }


    
      
	
		virtual NodeInformation Subsample( GridInformation &gridCosts, int nNewSamples, IndexType imageIndex)
		{
			/*  Subsampling from regular discrete grid      */
			/*  Copyright  2011 (c) Orcun Goksel             */
			NodeInformation nodeInfo;
			// Find all requested labels
			std::vector<int> localMinima, localCenterMinima;
			for(int labelNum=0; labelNum<nNewSamples; labelNum++)
                {
                    // center of subsample grid (local minima)
                    RegistrationLabelType locMin;
                    //??? locMin = gridCosts.arg_min();
                    // Why doesn't this work, old vnl version?  Do the long way below...
                    double minVal = vnl_bignum("+Infinity");   int ind=-1;
                    for(int ii = 0; ii<gridCosts.costs.size(); ii++)
                        {
                            if (gridCosts.costs[ii]<minVal &&
                                localMinima.end()==std::find(localMinima.begin(), localMinima.end(), ii) )
                                {
                                    ind = ii;
                                    minVal = gridCosts.costs[ii];
                                }
                        }
                    if(ind==-1) break;   //no more local minima found (handle outside)
                    localCenterMinima.push_back(ind);
                    // find subGrid location & bracket it inside the grid
                    if (Dimension==3) {
                        div_t qr = div(ind, gridCosts.dim[0] * gridCosts.dim[1]);
                        locMin[2] = std::max<int>(1, std::min<int>(gridCosts.dim[2]-2, qr.quot ) );
                        ind = qr.rem;
                    }
                    div_t qr = div(ind, gridCosts.dim[0]);
                    locMin[1] = std::max<int>(1, std::min<int>(gridCosts.dim[1]-2, qr.quot ) );
                    locMin[0] = std::max<int>(1, std::min<int>(gridCosts.dim[0]-2, qr.rem ) );
                    
                    // Push this grid location AND its neighbours to a list (to omit as next labels)
                    for(int kk = (Dimension==3 ? locMin[2]-1 : 0) ;
					    kk < (Dimension==3 ? locMin[2]+2 : 1) ; kk++)
                        for(int jj=locMin[1]-1; jj<locMin[1]+2; jj++)
                            for(int ii=locMin[0]-1; ii<locMin[0]+2; ii++)
                                localMinima.push_back(kk*gridCosts.dim[1]*gridCosts.dim[0] + jj*gridCosts.dim[0] + ii);

                    // vector for subsample grid ( center + 8 neighbour costs )
                    // copy the subsample grid (minus 1's are in order to get the corner of 3x3 window)
                    std::vector<double> subGridCosts;
                    for(int kk = (Dimension==3 ? locMin[2]-1 : 0) ;
					    kk < (Dimension==3 ? locMin[2]+2 : 1) ; kk++)
                        for(int jj=locMin[1]-1; jj<locMin[1]+2; jj++)
                            for(int ii=locMin[0]-1; ii<locMin[0]+2; ii++)
                                subGridCosts.push_back( gridCosts.costs[kk*gridCosts.dim[1]*gridCosts.dim[0] + jj*gridCosts.dim[0] + ii] );

                    // compute pseudo-inverse solution
                    vnl_matrix_fixed< double, 3*3*(Dimension==3?3:1), 1> subGridCostsMat;
                    subGridCostsMat.copy_in( &subGridCosts[0] );  // no range checking!!
                    vnl_matrix_fixed<double,4*Dimension-2,1> aMat = constA * subGridCostsMat;
                    double *a = aMat.data_block();

                    vnl_matrix_fixed<double,Dimension,Dimension> tempM;
                    if(Dimension==2)
                        {
                            tempM.put(0, 0, 2*a[4] );
                            tempM.put(1, 1, 2*a[5] );
                            tempM.put(0, 1,   a[3] );
                            tempM.put(1, 0,   a[3] );
                        } else 
                        { // Dimension == 3
                            tempM.put(0, 0, 2*a[7] );
                            tempM.put(1, 1, 2*a[8] );
                            tempM.put(2, 2, 2*a[9] );
                            tempM.put(0, 1,   a[4] );
                            tempM.put(1, 0,   a[4] );
                            tempM.put(0, 2,   a[5] );
                            tempM.put(2, 0,   a[5] );
                            tempM.put(1, 2,   a[6] );
                            tempM.put(2, 1,   a[6] );
                        }
                    vnl_matrix_fixed<double,Dimension,1> dMat = vnl_matrix_inverse<double>( tempM ) * aMat.extract(Dimension,1,1,0);
                    if(Dimension==2) dMat*=-1;  // why is this?  I don't know, just taking from Eq(15) in the TMI,2010 subsampling paper
                    double d[Dimension];
                    dMat.copy_out( d );

                    // bracket the displacement inside the subGrid [-1,1]
                    RegistrationLabelType label;
                    double distanceToOriginal=0.0;
                    for(int ii=0; ii<Dimension; ii++)
                        {
                            d[ii] = std::max<double>(-1.0, std::min<double>(1.0,d[ii]));
                            label[ii] = d[ii] + locMin[ii] - (gridCosts.dim[ii]-1)/2;
                            double tmp=label[ii]-locMin[ii];
                            tmp*=tmp;
                            distanceToOriginal+=tmp;
                        }
                    distanceToOriginal=sqrt(distanceToOriginal);
                    double result;
                    
                    if(Dimension==2){
                        result= a[0] + a[1]*d[0] + a[2]*d[1] + a[3]*d[0]*d[1] + a[4]*d[0]*d[0] + a[5]*d[1]*d[1] ;
                    }
                    else
                        result=a[0] + a[1]*d[0] + a[2]*d[1] + a[3]*d[2] + a[4]*d[0]*d[1] + a[5]*d[0]*d[2] + a[6]*d[1]*d[2] + a[7]*d[0]*d[0] + a[8]*d[1]*d[1] + a[9]*d[2]*d[2];

                    if ((a[4]<0 || a[5]<0) || result<minVal){
                        bool test= ((a[4]>0 || a[5]>0) && result<minVal);
                        cout<<test << endl;
                        nodeInfo.indexToSubsampledDisplacementMapping.push_back( label );
                        RegistrationLabelType l;
                        l=LabelMapperType::scaleDisplacement(label,this->getDisplacementFactor());
                        double trueValue=this->m_unaryRegFunction->getPotential(imageIndex,l)/this->m_nRegistrationNodes;
                        //nodeInfo.subsampledNodeCosts.push_back( result );
                        nodeInfo.subsampledNodeCosts.push_back( trueValue );

                        //                        cout<<labelNum<<" "<<result<<" "<<minVal<<" "<<distanceToOriginal<<" "<<result-minVal<<" "<<trueValue<<" "<<result-trueValue<<endl;
                        //cout<<"COSTS "<<result<<" "<<minVal<<" "<<trueValue<<" "<<fabs(result-trueValue)<<endl;
                    }else{
                        nodeInfo.subsampledNodeCosts.push_back( minVal );
                        nodeInfo.indexToSubsampledDisplacementMapping.push_back( LabelMapperType::getLabel(ind) );
                    }

                    // Hack :) to remove this local minimum in order to get other possible labels :: Fix Later!
                    // !!! Modifying pass by reference !!! assuming the cost grid will not be needed outside this function
                    //				gridCosts.costs[ind] += minVal;
                }
#if 1
            for(int ii = 0; ii<gridCosts.costs.size()&&nodeInfo.subsampledNodeCosts.size()<nNewSamples; ii++){
                if ( localCenterMinima.end()==std::find(localCenterMinima.begin(), localCenterMinima.end(), ii) ){
                    RegistrationLabelType label;
                    if (Dimension==3) {
                        div_t qr = div(ii, gridCosts.dim[0] * gridCosts.dim[1]);
                        label[3] = std::max<int>(1, std::min<int>(gridCosts.dim[2]-2, qr.quot ) );
                        ii = qr.rem;
                    }
                    div_t qr = div(ii, gridCosts.dim[0]);
                    label[1] = qr.quot;
                    label[0] = qr.rem ;
                    nodeInfo.indexToSubsampledDisplacementMapping.push_back(label);
                    nodeInfo.subsampledNodeCosts.push_back(gridCosts.costs[ii]);
                }
            }
            assert(nodeInfo.subsampledNodeCosts.size()==nNewSamples);
#endif
            while(nodeInfo.subsampledNodeCosts.size()<nNewSamples){
				nodeInfo.indexToSubsampledDisplacementMapping.push_back( nodeInfo.indexToSubsampledDisplacementMapping.front() );
				nodeInfo.subsampledNodeCosts.push_back( 10000000 );
			} 
            return nodeInfo;
		}
        virtual int nRegLabels(){
            return nRegistrationLabels;
        }

 
      
        //this is now all interfacing stuff i needed to rewrite because of the optimizer, don't bother :)
        double getUnaryRegistrationPotential(int nodeIndex,int labelIndex){
            double result;
#if 0
            //either recompute the costs
            IndexType imageIndex=this->getImageIndexFromCoarseGraphIndex(nodeIndex);
            RegistrationLabelType l=m_nodeMappingInfo[nodeIndex].indexToSubsampledDisplacementMapping[labelIndex];
            l=LabelMapperType::scaleDisplacement(l,this->getDisplacementFactor());
            result=this->m_unaryRegFunction->getPotential(imageIndex,l);
            return result/this->m_nRegistrationNodes;
#else
            //or take the actual interpolated values
            result=m_nodeMappingInfo[nodeIndex].subsampledNodeCosts[labelIndex];
            return result;
#endif
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
    template<class TImage, 
             class TUnaryRegistrationFunction, 
             class TPairwiseRegistrationFunction, 
             class TUnarySegmentationFunction, 
             class TPairwiseSegmentationFunction,
             class TPairwiseSegmentationRegistrationFunction,
             class TLabelMapper>
    class SubsamplingGraphModel2: 
        public SubsamplingGraphModel<TImage,
                                     TUnaryRegistrationFunction,
                                     TPairwiseRegistrationFunction,
                                     TUnarySegmentationFunction, 
                                     TPairwiseSegmentationFunction,
                                     TPairwiseSegmentationRegistrationFunction,
                                     TLabelMapper>
    {
    public:
        typedef SubsamplingGraphModel2 Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef SubsamplingGraphModel<TImage,
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
        typedef typename Superclass::NodeInformation NodeInformation;
        typedef typename Superclass::GridInformation GridInformation;
    protected:
        vnl_matrix_fixed< double, 4*Dimension-2-ISFITCENTER, 3*3*(Dimension==3?3:1)-ISFITCENTER > constA;  //template for Dimension=[2,3]
         
    public:

        void Init(){
            this->gridSize=LabelMapperType::nDisplacementSamples*2+1;
            double A2[9][6] = {0};    // constant matrix for 2D paraboloid fitting
            A2[0][0] = 1.0;	  A2[0][1] = -1.0;	  A2[0][2] = -1.0;	  A2[0][3] = 1.0;	  A2[0][4] = 1.0;	  A2[0][5] = 1.0;
            A2[1][0] = 1.0;	  A2[1][2] = -1.0;	  A2[1][5] = 1.0;
            A2[2][0] = 1.0;	  A2[2][1] = 1.0;	  A2[2][2] = -1.0;	  A2[2][3] = -1.0;	  A2[2][4] = 1.0;	  A2[2][5] = 1.0;
            A2[3][0] = 1.0;	  A2[3][1] = -1.0;	  A2[3][4] = 1.0;
            A2[4][0] = 1.0 * SCALECENTER;
            A2[5][0] = 1.0;	  A2[5][1] = 1.0;	  A2[5][4] = 1.0;
            A2[6][0] = 1.0;	  A2[6][1] = -1.0;	  A2[6][2] = 1.0;	  A2[6][3] = -1.0;	  A2[6][4] = 1.0;	  A2[6][5] = 1.0;
            A2[7][0] = 1.0;	  A2[7][2] = 1.0;	  A2[7][5] = 1.0;
            A2[8][0] = 1.0;	  A2[8][1] = 1.0;	  A2[8][2] = 1.0;	  A2[8][3] = 1.0;	  A2[8][4] = 1.0;	  A2[8][5] = 1.0;

            // syms x y z real;  f = inline([1,x,y,z,x*y,x*z,y*z,x*x,y*y,z*z],'x','y','z');
            // [x,y,z] = ndgrid(-1:1,-1:1,-1:1);  A=[];
            // for i=1:numel(x), A(i,:)=f(x(i),y(i),z(i)); end; A3=sym(A),  regexprep(ccode(A3),'\n','\t')
            double A3[27][10] = {0};    // constant matrix for 2D paraboloid fitting
            A3[0][0] = 1.0;	  A3[0][1] = -1.0;	  A3[0][2] = -1.0;	  A3[0][3] = -1.0;	  A3[0][4] = 1.0;	  A3[0][5] = 1.0;	  A3[0][6] = 1.0;	  A3[0][7] = 1.0;	  A3[0][8] = 1.0;	  A3[0][9] = 1.0;
            A3[1][0] = 1.0;	  A3[1][2] = -1.0;	  A3[1][3] = -1.0;	  A3[1][6] = 1.0;	  A3[1][8] = 1.0;	  A3[1][9] = 1.0;
            A3[2][0] = 1.0;	  A3[2][1] = 1.0;	  A3[2][2] = -1.0;	  A3[2][3] = -1.0;	  A3[2][4] = -1.0;	  A3[2][5] = -1.0;	  A3[2][6] = 1.0;	  A3[2][7] = 1.0;	  A3[2][8] = 1.0;	  A3[2][9] = 1.0;
            A3[3][0] = 1.0;	  A3[3][1] = -1.0;	  A3[3][3] = -1.0;	  A3[3][5] = 1.0;	  A3[3][7] = 1.0;	  A3[3][9] = 1.0;	  A3[4][0] = 1.0;	  A3[4][3] = -1.0;	  A3[4][9] = 1.0;
            A3[5][0] = 1.0;	  A3[5][1] = 1.0;	  A3[5][3] = -1.0;	  A3[5][5] = -1.0;	  A3[5][7] = 1.0;	  A3[5][9] = 1.0;
            A3[6][0] = 1.0;	  A3[6][1] = -1.0;	  A3[6][2] = 1.0;	  A3[6][3] = -1.0;	  A3[6][4] = -1.0;	  A3[6][5] = 1.0;	  A3[6][6] = -1.0;	  A3[6][7] = 1.0;	  A3[6][8] = 1.0;	  A3[6][9] = 1.0;
            A3[7][0] = 1.0;	  A3[7][2] = 1.0;	  A3[7][3] = -1.0;	  A3[7][6] = -1.0;	  A3[7][8] = 1.0;	  A3[7][9] = 1.0;
            A3[8][0] = 1.0;	  A3[8][1] = 1.0;	  A3[8][2] = 1.0;	  A3[8][3] = -1.0;	  A3[8][4] = 1.0;	  A3[8][5] = -1.0;	  A3[8][6] = -1.0;	  A3[8][7] = 1.0;	  A3[8][8] = 1.0;	  A3[8][9] = 1.0;
            A3[9][0] = 1.0;	  A3[9][1] = -1.0;	  A3[9][2] = -1.0;	  A3[9][4] = 1.0;	  A3[9][7] = 1.0;	  A3[9][8] = 1.0;
            A3[10][0] = 1.0;	  A3[10][2] = -1.0;	  A3[10][8] = 1.0;
            A3[11][0] = 1.0;	  A3[11][1] = 1.0;	  A3[11][2] = -1.0;	  A3[11][4] = -1.0;	  A3[11][7] = 1.0;	  A3[11][8] = 1.0;
            A3[12][0] = 1.0;	  A3[12][1] = -1.0;	  A3[12][7] = 1.0;
            A3[13][0] = 1.0 * SCALECENTER;
            A3[14][0] = 1.0;	  A3[14][1] = 1.0;	  A3[14][7] = 1.0;
            A3[15][0] = 1.0;	  A3[15][1] = -1.0;	  A3[15][2] = 1.0;	  A3[15][4] = -1.0;	  A3[15][7] = 1.0;	  A3[15][8] = 1.0;
            A3[16][0] = 1.0;	  A3[16][2] = 1.0;	  A3[16][8] = 1.0;
            A3[17][0] = 1.0;	  A3[17][1] = 1.0;	  A3[17][2] = 1.0;	  A3[17][4] = 1.0;	  A3[17][7] = 1.0;	  A3[17][8] = 1.0;
            A3[18][0] = 1.0;	  A3[18][1] = -1.0;	  A3[18][2] = -1.0;	  A3[18][3] = 1.0;	  A3[18][4] = 1.0;	  A3[18][5] = -1.0;	  A3[18][6] = -1.0;	  A3[18][7] = 1.0;	  A3[18][8] = 1.0;	  A3[18][9] = 1.0;
            A3[19][0] = 1.0;	  A3[19][2] = -1.0;	  A3[19][3] = 1.0;	  A3[19][6] = -1.0;	  A3[19][8] = 1.0;	  A3[19][9] = 1.0;
            A3[20][0] = 1.0;	  A3[20][1] = 1.0;	  A3[20][2] = -1.0;	  A3[20][3] = 1.0;	  A3[20][4] = -1.0;	  A3[20][5] = 1.0;	  A3[20][6] = -1.0;	  A3[20][7] = 1.0;	  A3[20][8] = 1.0;	  A3[20][9] = 1.0;
            A3[21][0] = 1.0;	  A3[21][1] = -1.0;	  A3[21][3] = 1.0;	  A3[21][5] = -1.0;	  A3[21][7] = 1.0;	  A3[21][9] = 1.0;
            A3[22][0] = 1.0;	  A3[22][3] = 1.0;	  A3[22][9] = 1.0;
            A3[23][0] = 1.0;	  A3[23][1] = 1.0;	  A3[23][3] = 1.0;	  A3[23][5] = 1.0;	  A3[23][7] = 1.0;	  A3[23][9] = 1.0;
            A3[24][0] = 1.0;	  A3[24][1] = -1.0;	  A3[24][2] = 1.0;	  A3[24][3] = 1.0;	  A3[24][4] = -1.0;	  A3[24][5] = -1.0;	  A3[24][6] = 1.0;	  A3[24][7] = 1.0;	  A3[24][8] = 1.0;	  A3[24][9] = 1.0;
            A3[25][0] = 1.0;	  A3[25][2] = 1.0;	  A3[25][3] = 1.0;	  A3[25][6] = 1.0;	  A3[25][8] = 1.0;	  A3[25][9] = 1.0;
            A3[26][0] = 1.0;	  A3[26][1] = 1.0;	  A3[26][2] = 1.0;	  A3[26][3] = 1.0;	  A3[26][4] = 1.0;	  A3[26][5] = 1.0;	  A3[26][6] = 1.0;	  A3[26][7] = 1.0;	  A3[26][8] = 1.0;	  A3[26][9] = 1.0;

            // *** Initialization of variables
            vnl_matrix< double > tempM;
            switch(Dimension)
                {
                case 2:
                    tempM.set_size(9-ISFITCENTER,6-ISFITCENTER);
                    for(int ii=0; ii<9-ISFITCENTER; ii++)  // there must be an easier matrix initialization than this nested for-loops
                        for(int jj=0; jj<6-ISFITCENTER; jj++)
                            tempM.put(ii, jj, A2[ii+(ISFITCENTER && ii>=4)][jj+ISFITCENTER] );
                    break;
                case 3:
                    tempM.set_size(27-ISFITCENTER,10-ISFITCENTER);
                    for(int ii=0; ii<27-ISFITCENTER; ii++)  // there must be an easier matrix initialization than this nested for-loops
                        for(int jj=0; jj<10-ISFITCENTER; jj++)
                            tempM.put(ii, jj, A3[ii+(ISFITCENTER && ii>=13)][jj+ISFITCENTER] );
                    break;
                default:
                    assert(0);
                }
            // constA = inv(A'*A)*A'    // according to Eq(13) in Sub-sample Disp..., Zahiri, Goksel, Salcudean: TMI 2010
            constA = vnl_matrix_inverse<double>( vnl_transpose(tempM) * tempM ) * vnl_transpose(tempM);
	            
            int nNewSamples=this->m_config.nSubsamples;//(gridSize-2)*(gridSize-2);
            cout<<nNewSamples<<endl;
            this->nRegistrationLabels=nNewSamples;
            this->m_nodeMappingInfo= std::vector< NodeInformation >(0);
            //computing new subsampled labels and label costs
            for (int r=0;r<this->m_nRegistrationNodes;++r){
                GridInformation originalRegistrationCosts;
                originalRegistrationCosts.dim[0] = this->gridSize;   originalRegistrationCosts.dim[1] = this->gridSize;   originalRegistrationCosts.dim[2] = this->gridSize;
                IndexType imageIndex=this->getImageIndexFromCoarseGraphIndex(r);
                for (int l=0;l<this->m_nDisplacementLabels;++l){
                    double tmp=Superclass::Superclass::getUnaryRegistrationPotential(r,l);
                    RegistrationLabelType label=LabelMapperType::getLabel(l);
                    if (l==1) assert(label[0]==-(this->gridSize-1)/2+1 &&label[1]==-(this->gridSize-1)/2);
                    originalRegistrationCosts.costs.push_back(tmp);
                    //if (r==0) std::cout<<r<<" "<<l<<" "<<label<<" "<< tmp<<endl;
                }
                
                // now compute novel mapping and costs and store in m_nodeMappingInfo
                //for the beginning it might be safe to leave m_nRegistrationLabels at its old value. otherwise we'd need to do some copying/overwritng to be safe
                this->m_nodeMappingInfo.push_back(Subsample(originalRegistrationCosts,nNewSamples));

            }
        }


        virtual NodeInformation Subsample( GridInformation &gridCosts, int nNewSamples)
        {
            /*  Subsampling from regular discrete grid      */
            /*  Copyright  2011 (c) Orcun Goksel             */
            NodeInformation nodeInfo;
            std::vector<int> localMinima, neighs;
            for(int kk = (Dimension==3 ? -1 : 0) ;
                kk < (Dimension==3 ? 2 : 1) ; kk++)
                for(int jj=-1; jj<2; jj++)
                    for(int ii=-1; ii<2; ii++)
                        neighs.push_back( kk*gridCosts.dim[1]*gridCosts.dim[0] + jj*gridCosts.dim[0] + ii );
            // Find all requested labels
            cout<<nNewSamples<<endl;
            for(int labelNum=0; labelNum<nNewSamples ; ++labelNum){
                bool boundary=false;
                // center of subsample grid (local minima)
                RegistrationLabelType locMin; int ind=-1;
                double minVal = 1E7;
                if (labelNum==0) {
                    ind = std::distance(gridCosts.costs.begin(), std::min_element(gridCosts.costs.begin(), gridCosts.costs.end()));
                    minVal = gridCosts.costs[ind];
                }
                else for(int ii = 0; ii<gridCosts.costs.size(); ii++)	{
                        if (gridCosts.costs[ii]<minVal &&
                            localMinima.end()==std::find(localMinima.begin(), localMinima.end(), ii) )	{
                            ind = ii;
                            minVal = gridCosts.costs[ii];
                        }
                    }
                if(ind==-1) break;   //no more local minima found (handle outside)

                int tmpInd=ind;
                int newInd=0;
                // find subGrid location & bracket it inside the grid
                if (Dimension==3) {
                    div_t qr = div(ind, gridCosts.dim[0] * gridCosts.dim[1]);
                    locMin[2] = std::max<int>(1, std::min<int>(gridCosts.dim[2]-2, qr.quot ) );
                    boundary=(boundary || qr.quot>gridCosts.dim[2]-2 || qr.quot<1);
                    ind = qr.rem;
                    newInd+=locMin[2]*gridCosts.dim[0] * gridCosts.dim[1];
                }
                div_t qr = div(ind, gridCosts.dim[0]);
                locMin[1] = std::max<int>(1, std::min<int>(gridCosts.dim[1]-2, qr.quot ) );
                boundary=(boundary || qr.quot>gridCosts.dim[1]-2 || qr.quot<1);
                newInd+=locMin[1]*gridCosts.dim[0];
                locMin[0] = std::max<int>(1, std::min<int>(gridCosts.dim[0]-2, qr.rem ) );
                boundary=(boundary || qr.quot>gridCosts.dim[0]-2 || qr.quot<1);
                newInd+=locMin[0];

                ind=tmpInd;
                // Push this grid location AND its neighbours to a list (to omit as next labels)
                for(int ii=0; ii<neighs.size(); ii++){
                    localMinima.push_back( newInd + neighs[ii] );
                }
                // vector for subsample grid ( center + 8 neighbour costs )
                // copy the subsample grid (minus 1's are in order to get the corner of 3x3 window)
                std::vector<double> subGridCosts;
                for(int ii=0; ii<neighs.size(); ii++)
                    if(ISFITCENTER && !neighs[ii]) continue;
                    else subGridCosts.push_back( gridCosts.costs[ newInd + neighs[ii] ] * ( (newInd+neighs[ii])==ind ?  SCALECENTER : 1) );

                // compute pseudo-inverse solution
                vnl_matrix_fixed< double, 3*3*(Dimension==3?3:1)-ISFITCENTER, 1> subGridCostsMat;
                subGridCostsMat.copy_in( &subGridCosts[0] );  // no range checking!!
                vnl_matrix_fixed<double,4*Dimension-2,1> aMat;
                aMat.update( constA * subGridCostsMat, ISFITCENTER, 0 );
                if(ISFITCENTER || CENTERFITHACK ) aMat.put(0, 0, minVal);
				
                double *a = aMat.data_block();

                vnl_matrix_fixed<double,Dimension,Dimension> tempM;
                if(Dimension==2)
                    {
                        tempM.put(0, 0, 2*a[4] );
                        tempM.put(1, 1, 2*a[5] );
                        tempM.put(0, 1,   a[3] );
                        tempM.put(1, 0,   a[3] );
                    } else { // Dimension == 3
                    tempM.put(0, 0, 2*a[7] );
                    tempM.put(1, 1, 2*a[8] );
                    tempM.put(2, 2, 2*a[9] );
                    tempM.put(0, 1,   a[4] );
                    tempM.put(1, 0,   a[4] );
                    tempM.put(0, 2,   a[5] );
                    tempM.put(2, 0,   a[5] );
                    tempM.put(1, 2,   a[6] );
                    tempM.put(2, 1,   a[6] );
                }
                vnl_matrix_fixed<double,Dimension,1> dMat = vnl_matrix_inverse<double>( tempM ) * aMat.extract(Dimension,1,1,0);
                if(Dimension==2) dMat*=-1;  // why is this?  I don't know, just taking from Eq(15) in the TMI,2010 subsampling paper
                double d[Dimension];
                dMat.copy_out( d );

                // bracket the displacement inside the subGrid [-1,1]
                RegistrationLabelType label;
                for(int ii=0; ii<Dimension; ii++)
                    {
                        d[ii] = std::max<double>(-1.0, std::min<double>(1.0,d[ii]));
                        label[ii] = d[ii] + locMin[ii] - (gridCosts.dim[ii]-1)/2;
                    }

                double result;

                if(Dimension==2)
                    result= a[0] + a[1]*d[0] + a[2]*d[1] + a[3]*d[0]*d[1] + a[4]*d[0]*d[0] + a[5]*d[1]*d[1];
                else
                    result= a[0] + a[1]*d[0] + a[2]*d[1] + a[3]*d[2] + a[4]*d[0]*d[1] + a[5]*d[0]*d[2] + a[6]*d[1]*d[2] + a[7]*d[0]*d[0] + a[8]*d[1]*d[1] + a[9]*d[2]*d[2] ;

                bool concave=a[4]<0 || a[5]<0;
                if (concave ||result>minVal)  {
                    if (this->m_config.verbose){
                        if (result>minVal) {
                            cout<<"LARGE "<<result<<" "<<minVal<<" bound:"<<boundary<<" conc:"<<concave<<endl;
                            cout<<"ERROR "<<fabs(result-minVal)<<endl;
                        }
                        else cout<<"not convex "<<a[4]<<" "<<a[5]<<" buond:"<<boundary<<endl;
                    }
                    result=minVal;
                    label=LabelMapperType::getLabel(ind);
                } //concavity check

                nodeInfo.indexToSubsampledDisplacementMapping.push_back( label );
                nodeInfo.subsampledNodeCosts.push_back( result );
                if (this->m_config.verbose) std::cout<<labelNum<<" "<<result<<" "<<minVal<<" "<<label<<" "<<LabelMapperType::getLabel(ind)<<endl;
            }
            while(nodeInfo.subsampledNodeCosts.size()<nNewSamples){
                RegistrationLabelType label;
                vnl_random rgen( 9667566 );  // the seed is provided to ensure repeatipility through experiments

                for(int ii=0; ii<Dimension; ii++)  label[ii] = rgen.lrand32(gridCosts.dim[ii]);
                int ind = label[1]*gridCosts.dim[0] + label[0];
                if(Dimension==3)   ind += label[2]*gridCosts.dim[1]*gridCosts.dim[0];
                nodeInfo.subsampledNodeCosts.push_back( gridCosts.costs.at( ind ) );

                for(int ii=0; ii<Dimension; ii++)  label[ii] -= (gridCosts.dim[ii]-1)/2;
                nodeInfo.indexToSubsampledDisplacementMapping.push_back( label );
                cout<<label<<" "<< gridCosts.costs.at( ind ) << endl;
            }
            return nodeInfo;
        }
    };//ssgraph2
    template<class TImage, 
             class TUnaryRegistrationFunction, 
             class TPairwiseRegistrationFunction, 
             class TUnarySegmentationFunction, 
             class TPairwiseSegmentationFunction,
             class TPairwiseSegmentationRegistrationFunction,
             class TLabelMapper>
    class SortedSubsamplingGraphModel: 
        public SubsamplingGraphModel<TImage,
                                     TUnaryRegistrationFunction,
                                     TPairwiseRegistrationFunction,
                                     TUnarySegmentationFunction, 
                                     TPairwiseSegmentationFunction,
                                     TPairwiseSegmentationRegistrationFunction,
                                     TLabelMapper>
    {
    public:
        typedef SortedSubsamplingGraphModel Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
        typedef SubsamplingGraphModel<TImage,
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
        typedef typename Superclass::NodeInformation NodeInformation;
        typedef typename Superclass::GridInformation GridInformation;
        
    public:

        void Init(){
            this->gridSize=LabelMapperType::nDisplacementSamples*2+1;
        
            int nNewSamples=this->m_config.nSubsamples;//(gridSize-2)*(gridSize-2);
            cout<<nNewSamples<<endl;
            this->nRegistrationLabels=nNewSamples;
            this->m_nodeMappingInfo= std::vector< NodeInformation >(0);
            //computing new subsampled labels and label costs
            for (int r=0;r<this->m_nRegistrationNodes;++r){
                std::vector<std::pair<double,int> > originalRegistrationCosts;
                IndexType imageIndex=this->getImageIndexFromCoarseGraphIndex(r);
                for (int l=0;l<this->m_nDisplacementLabels;++l){
                    double tmp=Superclass::Superclass::getUnaryRegistrationPotential(r,l);
                    originalRegistrationCosts.push_back(make_pair(tmp,l));
                }
                
                std::sort(originalRegistrationCosts.begin(),originalRegistrationCosts.end(),sort_pred());
                NodeInformation n;
                for (int i=0;i<nNewSamples;++i){
                    n.subsampledNodeCosts.push_back(originalRegistrationCosts[i].first);
                    n.indexToSubsampledDisplacementMapping.push_back(LabelMapperType::getLabel(originalRegistrationCosts[i].second));
                    //cout<<n.subsampledNodeCosts[i]<<" "<< n.indexToSubsampledDisplacementMapping[i]<<endl;
                }
                this->m_nodeMappingInfo.push_back(n);
            }
        }
    protected:
        struct sort_pred {
        bool operator()(const std::pair<double,int>& left, const std::pair<double,int> & right)
        {
            return left.first < right.first;
        }
        };
      
    };//ssgraph2
}//namespace

#endif /* GRIm_dim_H_ */
