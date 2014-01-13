#include "Log.h"
/*
 * Grid.h
 *
 *  Created on: Nov 25, 2010
 *      Author: gasst
 */

#ifndef GRAPH_H
#define GRAPH_H
 
#include <vector>
#include <assert.h>
#include "itkConstNeighborhoodIterator.h"
#include <limits>
#include "SRSConfig.h"
#include "Log.h"
#include "TransformationUtils.h"
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
             class TPairwiseCoherenceFunction,
             class TLabelMapper>
    class GraphModel: public itk::Object{
    public:
        typedef GraphModel Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
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
        typedef TPairwiseCoherenceFunction PairwiseCoherenceFunctionType;
        typedef typename PairwiseCoherenceFunctionType::Pointer PairwiseCoherenceFunctionPointerType;
    
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType RegistrationLabelType;
        typedef typename itk::Image<RegistrationLabelType,ImageType::ImageDimension> RegistrationLabelImageType;
        typedef typename RegistrationLabelImageType::Pointer RegistrationLabelImagePointerType;

        typedef int SegmentationLabelType;
        typedef typename itk::Image<SegmentationLabelType,ImageType::ImageDimension> SegmentationLabelImageType;
        typedef typename SegmentationLabelImageType::Pointer SegmentationLabelImagePointerType;

        typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
        typedef typename FloatImageType::Pointer FloatImagePointerType;
        typedef typename ImageUtils<FloatImageType>::ImageIteratorType FloatImageIteratorType;
    
    protected:
    
        SizeType m_totalSize,m_imageLevelDivisors,m_graphLevelDivisors,m_gridSize, m_imageSize;
        ImagePointerType m_coarseGraphImage,m_borderOfSegmentationROI;
        //grid spacing in unit pixels
        SpacingType m_gridPixelSpacing;
        //grid spacing in mm
        SpacingType m_gridSpacing, m_imageSpacing;
    
        //spacing of the displacement labels
        SpacingType m_labelSpacing;

        PointType m_origin;
        double m_DisplacementScalingFactor;
        static const unsigned int m_dim=TImage::ImageDimension;
        int m_nNodes,m_nVertices, m_nRegistrationNodes, m_nSegmentationNodes;
        int m_nRegEdges,m_nSegEdges,m_nSegRegEdges, m_nEdges;
        int m_nSegmentationLabels,m_nDisplacementLabels;

        double m_segmentationUnaryNormalizer;

        //ImageInterpolatorType m_ImageInterpolator,m_SegmentationInterpolator,m_BoneConfidenceInterploator;
        UnaryRegistrationFunctionPointerType m_unaryRegFunction;
        UnarySegmentationFunctionPointerType m_unarySegFunction;
        PairwiseSegmentationFunctionPointerType m_pairwiseSegFunction;
        PairwiseCoherenceFunctionPointerType m_pairwiseSegRegFunction;
        PairwiseRegistrationFunctionPointerType m_pairwiseRegFunction;
  
        //PairwiseFunctionPointerType m_pairwiseFunction;
        bool verbose;
        bool m_haveLabelMap;
        ConstImagePointerType m_targetImage,m_targetSegmentationImage;
        ConstImageNeighborhoodIteratorType m_targetNeighborhoodIterator;
        SRSConfig m_config;
        int m_maxRegSegNeighbors;

        bool m_normalizePotentials;

        std::vector<int> m_mapIdx1,m_mapIdx1Rev;
        bool m_reducedSegNodes;

    public:
        int getMaxRegSegNeighbors(){return m_maxRegSegNeighbors;}
        GraphModel(){
            assert(m_dim>1);
            assert(m_dim<4);
            m_haveLabelMap=false;
            verbose=false;
            m_nSegmentationLabels=LabelMapperType::nSegmentations;
            m_nDisplacementLabels=LabelMapperType::nDisplacements;
            m_DisplacementScalingFactor=1.0;
            m_normalizePotentials=false;
            m_reducedSegNodes=false;
        };
        ~GraphModel(){
            //delete m_targetNeighborhoodIterator;
        }
        void setConfig(SRSConfig c){
            m_config=c;
            verbose=c.verbose;
            m_normalizePotentials=c.normalizePotentials; 
        }
        void setTargetImage(ConstImagePointerType targetImage){
            m_targetImage=targetImage;
        }
        ImagePointerType getCoarseGraphImage(){ return this->m_coarseGraphImage;}
        void initGraph(int nGraphNodesPerEdge){
            assert(m_targetImage);
            logSetStage("Graph initialization");
            //image size
            m_imageSize=m_targetImage->GetLargestPossibleRegion().GetSize();
            m_imageSpacing=m_targetImage->GetSpacing();
            LOGV(1)<<"Full image resolution for graph initialization: "<<m_imageSize<<endl;
            m_nSegmentationNodes=1;
            m_nRegistrationNodes=1;
            m_DisplacementScalingFactor=1.0;
                        
            //calculate graph spacing
            setSpacing(nGraphNodesPerEdge);
            if (LabelMapperType::nDisplacementSamples){
#ifdef PIXELTRANSFORM
                m_labelSpacing=0.4*m_gridPixelSpacing/(LabelMapperType::nDisplacementSamples);
                LOGV(1)<<"Graph pixel spacing :"<<m_gridPixelSpacing<<std::endl; 

#else
                m_labelSpacing=0.4*m_gridSpacing/(LabelMapperType::nDisplacementSamples);
#endif
                LOGV(1)<<LabelMapperType::nDisplacementSamples<<" displacment samples per direction; "<<"with "<<m_labelSpacing<<" pixels spacing"<<std::endl;
                LOGV(1)<<"Max displacement per axis: " <<m_labelSpacing * LabelMapperType::nDisplacementSamples * m_DisplacementScalingFactor <<"mm" << endl;
            }
            for (int d=0;d<(int)m_dim;++d){

                //origin is original origin
                m_origin[d]=m_targetImage->GetOrigin()[d];
                //

            
                m_nRegistrationNodes*=m_gridSize[d];
                m_nSegmentationNodes*=m_imageSize[d];

                //level divisors are used to simplify the calculation of image indices from integer indices
                if (d>0){
                    m_imageLevelDivisors[d]=m_imageLevelDivisors[d-1]*m_imageSize[d-1];
                    m_graphLevelDivisors[d]=m_graphLevelDivisors[d-1]*m_gridSize[d-1];
                }else{
                    m_imageLevelDivisors[d]=1;
                    m_graphLevelDivisors[d]=1;
                }
            }
            //allocate helper image whihc can be used for coordinate transforms between fine and coarse level
            this->m_coarseGraphImage=ImageType::New();
            typename ImageType::RegionType region;
            region.SetSize(m_gridSize);
            this->m_coarseGraphImage->SetOrigin(m_targetImage->GetOrigin());
            this->m_coarseGraphImage->SetSpacing(m_gridSpacing);
            this->m_coarseGraphImage->SetRegions(region);
            this->m_coarseGraphImage->SetDirection(m_targetImage->GetDirection());
            this->m_coarseGraphImage->Allocate();


            m_nNodes=m_nRegistrationNodes+m_nSegmentationNodes;
            LOGV(1)<<"Total size of coarse graph: "<< m_gridSize<<std::endl;;
            LOGV(3)<<"Grid physical spacing: "<<m_gridSpacing<<std::endl;;
            LOGV(3)<<"Grid pixel spacing: "<<m_gridPixelSpacing<<std::endl;;
            //nvertices is not used!?
            if (m_dim>=2){
                m_nRegEdges=m_gridSize[1]*(m_gridSize[0]-1)+m_gridSize[0]*(m_gridSize[1]-1);
                m_nSegEdges=m_imageSize[1]*(m_imageSize[0]-1)+m_imageSize[0]*(m_imageSize[1]-1);
            }
            if (m_dim==3){
                m_nRegEdges=m_nRegEdges*this->m_gridSize[2]+(this->m_gridSize[2]-1)*this->m_gridSize[1]*this->m_gridSize[0];
                m_nSegEdges=m_nSegEdges*this->m_imageSize[2]+(this->m_imageSize[2]-1)*this->m_imageSize[1]*this->m_imageSize[0];
            }

            typename ConstImageNeighborhoodIteratorType::RadiusType r;
            //controls the size of the neighborhood for registration-to-segmentation edges
            double reductionFactor=1;
            m_maxRegSegNeighbors=1;
            for (int d=0;d<(int)m_dim;++d){
                r[d]= m_targetImage->GetLargestPossibleRegion().GetSize()[d]/this->m_coarseGraphImage->GetLargestPossibleRegion().GetSize()[d];//(m_gridPixelSpacing[d]/(2*reductionFactor));
                m_maxRegSegNeighbors*=(2*r[d]+1);
            }
            m_targetNeighborhoodIterator=ConstImageNeighborhoodIteratorType(r,m_targetImage,m_targetImage->GetLargestPossibleRegion());
            m_nSegRegEdges=m_nSegmentationNodes/pow(reductionFactor,m_dim);
            m_nEdges=m_nRegEdges+m_nSegEdges+m_nSegRegEdges;
            LOGV(2)<<"Theoretical numbers of nodes/edges:"<<std::endl;
            LOGV(2)<<" totalNodes:"<<m_nNodes<<" totalEdges:"<<m_nRegEdges+m_nSegEdges+m_nSegRegEdges<<" labels:"<<LabelMapperType::nLabels<<std::endl;
            LOGV(2)<<" Segnodes:"<<m_nSegmentationNodes<<"\t SegEdges :"<<m_nSegEdges<<std::endl ;
            LOGV(2) <<" Regnodes:"<<m_nRegistrationNodes<<"\t\t RegEdges :"<<m_nRegEdges<<std::endl;
            LOGV(2)                <<" SegRegEdges:"<<m_nSegRegEdges<<std::endl;
                         
        
            m_segmentationUnaryNormalizer=m_nSegmentationNodes;
            LOGV(1)<<" finished graph init" <<std::endl;
            logResetStage;
        }
        //can be used to initialize stuff right before potentials are called
        virtual void Init(){};
        virtual void setSpacing(int shortestN){
            assert(m_targetImage);
            this->m_coarseGraphImage=ImageType::New();
            
            unsigned int minDim=999999;
            unsigned int minSize=999999;
            LOGV(8)<<"original image spacing "<<m_imageSpacing<<endl;
            //get shortest image edge
            for (int d=0;d<ImageType::ImageDimension;++d){
                if(m_imageSize[d]<minSize) {minSize=m_imageSize[d]; minDim=d;}
            }
            LOGV(8)<<"shortest edge has size :"<<minSize<<" in dimension :"<<minDim<<" which has spacing :"<<m_imageSpacing[minDim]<<endl;

            //calculate spacing for resizing the shortest edge to shortestN
            double minSpacing=m_imageSpacing[minDim]*(m_imageSize[minDim]-1)/(shortestN-1);
            minSpacing=minSpacing>=1?minSpacing:1.0;
            LOGV(8)<<"spacing for resampling this edge to "<<shortestN<<" pixels :"<<minSpacing<<endl;

            //calculate spacing and size for all image dimensions using
            for (int d=0;d<ImageType::ImageDimension;++d){

                int div= (1.0*m_imageSpacing[d]/minSpacing*(m_imageSize[d]-1))+1 ;
                m_gridSpacing[d]=1.0*m_imageSpacing[d]*(m_imageSize[d]-1)/(div-1);
                //m_gridPixelSpacing[d]= (m_imageSize[d]-1)/(div-1);
                LOGV(8)<<d<<" "<<div<<" "<< m_gridSpacing[d] <<" "<< m_gridPixelSpacing[d]<<" "<<m_imageSpacing[d]<<endl;
                m_gridSize[d]=div;
            }

            this->m_coarseGraphImage->SetSpacing(m_gridSpacing);
            typename ImageType::RegionType region;
            region.SetSize(m_gridSize);

            this->m_coarseGraphImage->SetRegions(region);
            this->m_coarseGraphImage->SetOrigin(m_targetImage->GetOrigin());
            this->m_coarseGraphImage->SetDirection(m_targetImage->GetDirection());
            this->m_coarseGraphImage->Allocate();
            this->m_coarseGraphImage->FillBuffer(1);
            
            //ImageUtils<ImageType>::writeImage("coarsegraph.nii",this->m_coarseGraphImage);
            LOGV(8)<<"physical coordinate consistency check"<<endl;
            for (int d=0;d<ImageType::ImageDimension;++d){
                IndexType idx;
                PointType pt;
                idx.Fill(0);
                idx[d]=m_gridSize[d]-1;
                this->m_coarseGraphImage->TransformIndexToPhysicalPoint(idx,pt);
                LOGV(8)<<d<<" Graph :"<<idx<<" "<<pt<<endl;
                idx[d]=m_imageSize[d]-1;
                m_targetImage->TransformIndexToPhysicalPoint(idx,pt);
                LOGV(8)<<d<<" Image :"<<idx<<" "<<pt<<endl;
            }

        }
    

        void SetTargetSegmentation(ConstImagePointerType seg){m_targetSegmentationImage=seg;}
        int GetTargetSegmentationAtIdx(int idx){
            return 0;
            if (m_targetSegmentationImage.IsNotNull()){
                IndexType pos=getImageIndex(idx);
                return m_targetSegmentationImage->GetPixel(pos);
                
            }else
                return 0;
        }

        void ReduceSegmentationNodesByCoherencePotential(double thresh){
            int maxLabel=this->m_nSegmentationLabels-1;
            LOGV(1)<<"Removing all segmentation nodes with coherence potential larger "<<thresh<<" for label "<<maxLabel<<endl;
            FloatImagePointerType dist=m_pairwiseSegRegFunction->GetDistanceTransform(maxLabel);
            m_reducedSegNodes=false;
            m_borderOfSegmentationROI=FilterUtils<FloatImageType,ImageType>::createEmptyFrom(dist);
            m_borderOfSegmentationROI->FillBuffer(0);
            int actualIdx=0,concurrentIdx=0;
            int nNodes=this->m_targetImage->GetLargestPossibleRegion().GetNumberOfPixels();
            LOGV(5)<<VAR(dist->GetLargestPossibleRegion().GetSize())<<" "<<this->m_targetImage->GetLargestPossibleRegion().GetSize()<<endl;
            m_mapIdx1=std::vector<int>(nNodes,-1);
            m_mapIdx1Rev=std::vector<int>(nNodes,-1);
            for (;actualIdx<nNodes;++actualIdx){
                
                IndexType position1=getImageIndex(actualIdx);
                PointType pt;
                m_targetImage->TransformIndexToPhysicalPoint(position1,pt);
                IndexType position2;
                dist->TransformPhysicalPointToIndex(pt,position2);
                float distAtPos=dist->GetPixel(position2);
                LOGV(9)<<VAR(distAtPos)<<" "<<VAR(thresh)<<endl;
                if (distAtPos<thresh){
                    m_mapIdx1[actualIdx]=concurrentIdx;
                    m_mapIdx1Rev[concurrentIdx]=actualIdx;
                    ++concurrentIdx;
                    m_borderOfSegmentationROI->SetPixel(position1,1);
                    
                }
            }
            //erosion can give strange results, fixing by thresholding
            m_borderOfSegmentationROI=FilterUtils<ImageType>::substract(m_borderOfSegmentationROI,FilterUtils<ImageType>::binaryThresholding(FilterUtils<ImageType>::erosion(m_borderOfSegmentationROI,1),1,1));

            LOGI(6,ImageUtils<ImageType>::writeImage("ROI.nii",m_borderOfSegmentationROI));
            m_nSegmentationNodes=concurrentIdx;
            LOG<<"Reduced number of segmentation nodes to "<<100.0*concurrentIdx/actualIdx<<"%; "<<actualIdx<<"->"<<concurrentIdx<<endl;
            m_mapIdx1Rev.resize(concurrentIdx);
            m_reducedSegNodes=true;
            
            


        }
     
        //return position index in coarse graph from coarse graph node index
        virtual IndexType  getGraphIndex(int nodeIndex){
            IndexType position;
            for ( int d=m_dim-1;d>=0;--d){
                //position[d] is now the index in the coarse graph (image)
                position[d]=nodeIndex/m_graphLevelDivisors[d];
                nodeIndex-=position[d]*m_graphLevelDivisors[d];
            }
            return position;
        }
        //#define MANUALCONVERSION
        //return position in full image from coarse graph node index
        virtual IndexType  getImageIndexFromCoarseGraphIndex(int idx){
            IndexType position;
#ifdef MANUALCONVERSION
            for ( int d=m_dim-1;d>=0;--d){
                //position[d] is now the index in the coarse graph (image)
                position[d]=idx/m_graphLevelDivisors[d];
                idx-=position[d]*m_graphLevelDivisors[d];
                //now calculate the fine image index from the coarse graph index
                position[d]*=m_gridSpacing[d]/m_imageSpacing[d];
            }
#else
            position=getGraphIndex(idx);
            PointType physicalPoint;
            this->m_coarseGraphImage->TransformIndexToPhysicalPoint(position,physicalPoint);
            m_targetImage->TransformPhysicalPointToIndex(physicalPoint,position);
#endif
            if (!m_targetImage->GetLargestPossibleRegion().IsInside(position)){
                LOG<<"BROKEN :"<<position<<" not in target image region. target image size: "<<m_targetImage->GetLargestPossibleRegion().GetSize()<<std::endl;
                LOG<<idx<<" "<<m_graphLevelDivisors<<" "<<m_gridSpacing[0]/m_imageSpacing[0]<<endl;
            }
            assert(m_targetImage->GetLargestPossibleRegion().IsInside(position));
            return position;
        }
        virtual IndexType getClosestGraphIndex(IndexType imageIndex){
            IndexType position;
#ifdef MANUALCONVERSION
            for (unsigned int d=0;d<m_dim;++d){
                position[d]=floor(1.0*imageIndex[d]*m_imageSpacing[d]/m_gridSpacing[d]+0.5);
                //position[d]<<std::target << std::setprecision(0) << imageIndex[d]*m_imageSpacing[d]/m_gridSpacing[d];
            }
#else
            PointType physicalPoint;
            m_targetImage->TransformIndexToPhysicalPoint(imageIndex,physicalPoint);
            this->m_coarseGraphImage ->TransformPhysicalPointToIndex(physicalPoint,position);
#endif
            return position;
        }
        virtual IndexType getLowerGraphIndex(IndexType imageIndex){
            IndexType position;
            for (unsigned int d=0;d<m_dim;++d){
                position[d]=int(imageIndex[d]*m_imageSpacing[d]/m_gridSpacing[d]);
                //position[d]<<std::target << std::setprecision(0) << imageIndex[d]*m_imageSpacing[d]/m_gridSpacing[d];
            }
            return position;
        }
        virtual int  getGraphIntegerIndex(IndexType gridIndex){
            int i=0;
            for (unsigned int d=0;d<m_dim;++d){
                i+=gridIndex[d]*m_graphLevelDivisors[d];
            }
            return i;
        }
        virtual int  getImageIntegerIndex(IndexType imageIndex){
            int i=0;
            for (unsigned int d=0;d<m_dim;++d){
                i+=imageIndex[d]*m_imageLevelDivisors[d];
            }
            if (m_reducedSegNodes) {
                i=m_mapIdx1[i];
             
            }
            return i;
        }

        //return position in full image depending on fine graph nodeindex
        virtual IndexType getImageIndex(int idx){
            IndexType position;
            if (m_reducedSegNodes) {
                idx=m_mapIdx1Rev[idx];
            }
            for ( int d=m_dim-1;d>=0;--d){
                position[d]=idx/m_imageLevelDivisors[d];
                idx-=position[d]*m_imageLevelDivisors[d];
            }
            assert(m_targetImage->GetLargestPossibleRegion().IsInside(position));
            return position;
        }
        virtual double getUnaryRegistrationPotential(int nodeIndex,int labelIndex){
            IndexType imageIndex=getImageIndexFromCoarseGraphIndex(nodeIndex);
            RegistrationLabelType l=LabelMapperType::getLabel(labelIndex);
            l=LabelMapperType::scaleDisplacement(l,getDisplacementFactor());
            double result=m_unaryRegFunction->getPotential(imageIndex,l);
            if (m_normalizePotentials) result/=m_nRegistrationNodes;
            return result;//m_nRegistrationNodes;
        }
        virtual double getUnarySegmentationPotential(int nodeIndex,int labelIndex){
            IndexType imageIndex=getImageIndex(nodeIndex);
                
            if (m_targetSegmentationImage.IsNotNull()){
                labelIndex=m_targetSegmentationImage->GetPixel(imageIndex);
            }
            if (  labelIndex>0 && m_reducedSegNodes ){
                //if trying to label a pixel at the border of the segmentation ROI as FG; penalize strongly
                if (m_borderOfSegmentationROI->GetPixel(imageIndex) )
                    return 1000;
            }
                    


            //Segmentation:labelIndex==segmentationlabel
            double result=m_unarySegFunction->getPotential(imageIndex,labelIndex);///m_nSegmentationNodes;
            if (result<0){
                LOG<<"unary segmentation potential <0"<<std::endl;
                LOG<<imageIndex<<" " <<result<<std::endl;
            }
            if (m_normalizePotentials) result/=m_segmentationUnaryNormalizer;

            return result;
        };
        virtual inline double getPairwiseRegistrationPotential(int nodeIndex1, int nodeIndex2, int labelIndex1, int labelIndex2){
            
            //IndexType graphIndex1=getImageIndexFromCoarseGraphIndex(nodeIndex1);
            //IndexType graphIndex2=getImageIndexFromCoarseGraphIndex(nodeIndex2);
            IndexType graphIndex1=getGraphIndex(nodeIndex1);
            IndexType graphIndex2=getGraphIndex(nodeIndex2);
            PointType pt1,pt2;
            this->m_coarseGraphImage->TransformIndexToPhysicalPoint(graphIndex1,pt1);
            this->m_coarseGraphImage->TransformIndexToPhysicalPoint(graphIndex2,pt2);
            RegistrationLabelType l2=LabelMapperType::getLabel(labelIndex2);
            l2=LabelMapperType::scaleDisplacement(l2,getDisplacementFactor());
            RegistrationLabelType l1=LabelMapperType::getLabel(labelIndex1);
            l1=LabelMapperType::scaleDisplacement(l1,getDisplacementFactor());
            //return m_pairwiseRegFunction->getPotential(graphIndex1, graphIndex2, l1,l2);//m_nRegEdges;
            double result=m_pairwiseRegFunction->getPotential(pt1, pt2, l1,l2);//m_nRegEdges;
            if (m_normalizePotentials) result/=m_nRegEdges;
            return result;
        };
        virtual double getPairwiseSegRegPotential(int nodeIndex1, int nodeIndex2, int labelIndex1, int segmentationLabel){
            assert(false);
            IndexType graphIndex=getImageIndexFromCoarseGraphIndex(nodeIndex1);
            IndexType imageIndex=getImageIndex(nodeIndex2);
            //compute distance between center index and patch index
            double dist=0;
            for (unsigned int d=0;d<m_dim;++d){
                dist+=fabs(graphIndex[d]-imageIndex[d]);
            }
            double weight=1;//exp(-dist/2);
            if (weight<0){ LOG<<"weight smaller zero!! :"<<weight<<std::endl; weight=0;}
            RegistrationLabelType registrationLabel=LabelMapperType::getLabel(labelIndex1);
            registrationLabel=LabelMapperType::scaleDisplacement(registrationLabel,getDisplacementFactor());
            double result=m_pairwiseSegRegFunction->getPotential(imageIndex,imageIndex,registrationLabel,segmentationLabel);//m_nSegRegEdges;
            if (m_normalizePotentials) result/=m_nSegRegEdges;
            return result;
            
            //        return m_pairwiseSegRegFunction->getPotential(graphIndex,imageIndex,registrationLabel,segmentationLabel)/m_nSegRegEdges;
        }

        //#define MULTISEGREGNEIGHBORS
        virtual inline double getPairwiseRegSegPotential(int nodeIndex1, int nodeIndex2, int labelIndex1, int segmentationLabel){
    
            IndexType imageIndex=getImageIndex(nodeIndex2);
            if (m_targetSegmentationImage.IsNotNull()){
                segmentationLabel=m_targetSegmentationImage->GetPixel(imageIndex);
            }
            //compute distance between center index and patch index
            double weight=1.0;
            //#ifdef MULTISEGREGNEIGHBORS
#if 1
            PointType imagePoint,graphPoint;
            this->m_targetImage->TransformIndexToPhysicalPoint(imageIndex,imagePoint);
            IndexType graphIndex=getGraphIndex(nodeIndex1);
            this->m_coarseGraphImage->TransformIndexToPhysicalPoint(graphIndex,graphPoint);
            double dist=1;
            for (unsigned int d=0;d<m_dim;++d){
                //            LOG<<dist<<" "<<graphIndex[d]-imageIndex[d]<<" "<<std::endl;
                dist*=1.0-fabs((graphPoint[d]-imagePoint[d])/(m_gridSpacing[d]));
            }
            //       if (dist<0.1) dist=0.1;
            weight=dist;
#endif
            //        if (true){ LOG<<graphIndex<<" "<<imageIndex<<" "<<m_gridPixelSpacing<<" "<<weight<<std::endl;}
            RegistrationLabelType registrationLabel=LabelMapperType::getLabel(labelIndex1);
            registrationLabel=LabelMapperType::scaleDisplacement(registrationLabel,getDisplacementFactor());
            double result = weight*m_pairwiseSegRegFunction->getPotential(imageIndex,imageIndex,registrationLabel,segmentationLabel);//m_nSegRegEdges;
            //        return m_pairwiseSegRegFunction->getPotential(graphIndex,imageIndex,registrationLabel,segmentationLabel)/m_nSegRegEdges;
            if (m_normalizePotentials) result/=m_nSegRegEdges;
            return result;
        }
        
        virtual inline double getPairwiseRegSegPotential(int nodeIndex2, int labelIndex1, int segmentationLabel){
    
            IndexType imageIndex=getImageIndex(nodeIndex2);
            if (m_targetSegmentationImage.IsNotNull()){
                segmentationLabel=m_targetSegmentationImage->GetPixel(imageIndex);
            }
            //compute distance between center index and patch index
            double weight=1.0;
            //#ifdef MULTISEGREGNEIGHBORS
#if 1
            PointType imagePoint,graphPoint;
            this->m_targetImage->TransformIndexToPhysicalPoint(imageIndex,imagePoint);
            IndexType graphIndex=getClosestGraphIndex(imageIndex);
            this->m_coarseGraphImage->TransformIndexToPhysicalPoint(graphIndex,graphPoint);
            double dist=1;
            for (unsigned int d=0;d<m_dim;++d){
                //            LOG<<dist<<" "<<graphIndex[d]-imageIndex[d]<<" "<<std::endl;
                dist*=1.0-fabs((graphPoint[d]-imagePoint[d])/(m_gridSpacing[d]));
            }
            //       if (dist<0.1) dist=0.1;
            weight=dist;
#endif
            //        if (true){ LOG<<graphIndex<<" "<<imageIndex<<" "<<m_gridPixelSpacing<<" "<<weight<<std::endl;}
            RegistrationLabelType registrationLabel=LabelMapperType::getLabel(labelIndex1);
            registrationLabel=LabelMapperType::scaleDisplacement(registrationLabel,getDisplacementFactor());
            double result =  weight*m_pairwiseSegRegFunction->getPotential(imageIndex,imageIndex,registrationLabel,segmentationLabel);//m_nSegRegEdges;
            //        return m_pairwiseSegRegFunction->getPotential(graphIndex,imageIndex,registrationLabel,segmentationLabel)/m_nSegRegEdges;
            if (m_normalizePotentials) result/=m_nSegRegEdges;
            return result;
        }
        
        virtual inline double getPairwiseSegmentationPotential(int nodeIndex1, int nodeIndex2, int label1, int label2){
            IndexType imageIndex1=getImageIndex(nodeIndex1);
            IndexType imageIndex2=getImageIndex(nodeIndex2);
            if (m_targetSegmentationImage.IsNotNull()){
                label1=m_targetSegmentationImage->GetPixel(imageIndex1);
                label2=m_targetSegmentationImage->GetPixel(imageIndex2);
            }

            double result=m_pairwiseSegFunction->getPotential(imageIndex1,imageIndex2,label1, label2);//m_nSegEdges;
            if (m_normalizePotentials) result/=m_nSegEdges;

            return result;
        }
        virtual inline double getSegmentationWeight(int nodeIndex1, int nodeIndex2){
            IndexType imageIndex1=getImageIndex(nodeIndex1);
            IndexType imageIndex2=getImageIndex(nodeIndex2);
            double result=m_unarySegFunction->getWeight(imageIndex1,imageIndex2)/m_nSegEdges;
            LOG<<"I don't think we should be here..."<<std::endl;
            if (result<0){
                
                LOG<<imageIndex1<<" "<<imageIndex2<<" "<<result<<std::endl;
            }
            return result;
        }
   
    
        std::vector<int> getForwardRegistrationNeighbours(int index){
            IndexType position=getGraphIndex(index);
            std::vector<int> neighbours;
            for ( int d=0;d<(int)m_dim;++d){
                OffsetType off;
                off.Fill(0);
                if ((int)position[d]<(int)m_gridSize[d]-1){
                    off[d]+=1;
                    neighbours.push_back(getGraphIntegerIndex(position+off));
                }
            }
            return neighbours;
        }
        std::vector<int> getForwardSegmentationNeighbours(int index){
            IndexType position=getImageIndex(index);
            std::vector<int> neighbours;
            for ( int d=0;d<(int)m_dim;++d){
                OffsetType off;
                off.Fill(0);
                if ((int)position[d]<(int)m_imageSize[d]-1){
                    off[d]+=1;
                    int idx=getImageIntegerIndex(position+off);
                    if (idx>0)neighbours.push_back(idx);
                }
            }
            return neighbours;
        }
        std::vector<int>  getRegSegNeighbors(int index){            
            IndexType imagePosition=getImageIndexFromCoarseGraphIndex(index);
            std::vector<int> neighbours;
            m_targetNeighborhoodIterator.SetLocation(imagePosition);
            for (unsigned int i=0;i<m_targetNeighborhoodIterator.Size();++i){
                IndexType idx=m_targetNeighborhoodIterator.GetIndex(i);
                if (m_targetImage->GetLargestPossibleRegion().IsInside(idx)){
                    int inIdx=getImageIntegerIndex(idx);
                    if (inIdx>0) neighbours.push_back(inIdx);
                }
            }
            return neighbours;
        }
    
        std::vector<int> getSegRegNeighbors(int index){
            std::vector<int> neighbours;
#ifdef MULTISEGREGNEIGHBORS
            IndexType position=getLowerGraphIndex(getImageIndex(index));

            neighbours.push_back(getGraphIntegerIndex(position));

        
            OffsetType off;
            off.Fill(0);
            for (int d=1;d<pow(2,m_dim);++d){
                int carry=1;
                bool inBounds=true;
                for (unsigned int d2=0;d2<m_dim;++d2){
                    if (carry && off[d2]>0 ){
                        off[d2]=0;
                    }
                    else{
                        off[d2]=1;
                        break;
                    }
                }
                for (unsigned int d2=0;d2<m_dim;++d2){
                    if (position[d2]+off[d2]>=m_gridSize[d2]){
                        inBounds=false;
                        break;
                    }
                }
                if (inBounds){
                    //LOG<<getImageIndex(index)<<" "<<position<<" "<<off<<" "<<position+off<<" "<<getGraphIntegerIndex(position+off)<<std::endl;
                    neighbours.push_back(getGraphIntegerIndex(position+off));
                }
            }
#else
            IndexType idx=getImageIndex(index);
         
            // for (int d=0;d<m_dim;++d){
            //     if (false && idx[d]%2)
            //         return neighbours;
            // }

            IndexType position=getClosestGraphIndex(idx);
            neighbours.push_back(getGraphIntegerIndex(position));
 
#endif
            return neighbours;
        }
    
        virtual RegistrationLabelImagePointerType getDeformationImage(std::vector<int>  labels){
            RegistrationLabelImagePointerType result=RegistrationLabelImageType::New();
            typename RegistrationLabelImageType::RegionType region;
            region.SetSize(m_gridSize);
            result->SetRegions(region);
            result->SetSpacing(m_gridSpacing);
            result->SetDirection(m_targetImage->GetDirection());
            result->SetOrigin(m_origin);
            result->Allocate();
            typename itk::ImageRegionIterator<RegistrationLabelImageType> it(result,region);
            unsigned int i=0;
            for (it.GoToBegin();!it.IsAtEnd();++it,++i){
                assert(i<labels.size());
                RegistrationLabelType l=LabelMapperType::getLabel(labels[i]);
                l=LabelMapperType::scaleDisplacement(l,getDisplacementFactor());
                it.Set(l);
            }
            assert(i==(labels.size()));
            //LOGV(8)<<"git "<<labels.size()<<" registration labels which were transformed into a deformation field with parameters : "<<result<<endl;
            return result;
        }
        
        //empty deformation image
        virtual RegistrationLabelImagePointerType getDeformationImage(){
            RegistrationLabelImagePointerType result=RegistrationLabelImageType::New();
            typename RegistrationLabelImageType::RegionType region;
            region.SetSize(m_gridSize);
            result->SetRegions(region);
            result->SetSpacing(m_gridSpacing);
            result->SetDirection(m_targetImage->GetDirection());
            result->SetOrigin(m_origin);
            result->Allocate();
            typename itk::ImageRegionIterator<RegistrationLabelImageType> it(result,region);
            unsigned int i=0;
            for (it.GoToBegin();!it.IsAtEnd();++it,++i){
                RegistrationLabelType l=LabelMapperType::getLabel(LabelMapperType::nDisplacements/2);
                it.Set(l);
            }
            return result;
        }
        virtual ImagePointerType getParameterImage(){
            ImagePointerType result=ImageType::New();
            typename ImageType::RegionType region;
            region.SetSize(m_gridSize);
            result->SetRegions(region);
            result->SetSpacing(m_gridSpacing);
            result->SetDirection(m_targetImage->GetDirection());
            result->SetOrigin(m_origin);
            result->Allocate();
            return result;
        }
        

        ImagePointerType getSegmentationImage(std::vector<int> labels){
            ImagePointerType result=ImageType::New();
            result->SetRegions(m_targetImage->GetLargestPossibleRegion());
            result->SetSpacing(m_targetImage->GetSpacing());
            result->SetDirection(m_targetImage->GetDirection());
            result->SetOrigin(m_targetImage->GetOrigin());
            result->Allocate();
            LOGV(10)<<"target segmentation image: "<<result->GetLargestPossibleRegion()<<" "<<labels.size()<<" "<<m_nSegmentationLabels<<endl;
            typename itk::ImageRegionIterator<ImageType> it(result,result->GetLargestPossibleRegion());
            unsigned int i=0;
            if (m_nSegmentationLabels){
                for (it.GoToBegin();!it.IsAtEnd();++it,++i){
                    if (m_reducedSegNodes){
                        int idx=m_mapIdx1[i];
                        if (idx>-1){
                            it.Set(labels[idx]);
                        }
                        else
                            it.Set(0);
                    }else{
                        it.Set(labels[i]);
                    }
                }
            }else{  for (it.GoToBegin();!it.IsAtEnd();++it,++i){
                    it.Set(0);
                }
            }
            return result;
        }
        SizeType getImageSize() const
        {
            return m_imageSize;
        }
        SizeType getGridSize() const
        {return m_gridSize;}

        int nNodes(){return m_nNodes;}
        int nEdges(){return m_nEdges;}
        int nRegEdges(){return m_nRegEdges;}
        int nSegEdges(){return m_nSegEdges;}

        ImagePointerType getTargetImage(){
            return m_targetImage;
        }
        void setUnaryRegistrationFunction(UnaryRegistrationFunctionPointerType unaryFunc){
            m_unaryRegFunction=unaryFunc;
        }
        void setUnarySegmentationFunction(UnarySegmentationFunctionPointerType func){
            m_unarySegFunction=func;
        }
        void setPairwiseSegmentationFunction(PairwiseSegmentationFunctionPointerType func){
            m_pairwiseSegFunction=func;
        }
        void setPairwiseCoherenceFunction( PairwiseCoherenceFunctionPointerType func){
            m_pairwiseSegRegFunction=func;
        }
        void setPairwiseRegistrationFunction( PairwiseRegistrationFunctionPointerType func){
            m_pairwiseRegFunction=func;
        }

        typename ImageType::DirectionType getDirection(){return m_targetImage->GetDirection();}

        void setDisplacementFactor(double fac){
            m_DisplacementScalingFactor=fac;
            LOGV(1)<<"Max displacement per axis: " <<m_labelSpacing * LabelMapperType::nDisplacementSamples * m_DisplacementScalingFactor <<"mm" << endl;
        }
        double getMaxDisplacementFactor(){
            double maxSpacing=-1;
            for (unsigned int d=0;d<m_dim;++d){
                if (m_labelSpacing[d]>maxSpacing) maxSpacing=m_labelSpacing[d];
            }
            return maxSpacing*m_DisplacementScalingFactor;
        }
        SpacingType getDisplacementFactor(){return m_labelSpacing*m_DisplacementScalingFactor;}
        SpacingType getSpacing(){return m_gridSpacing;}	
        //SpacingType getPixelSpacing(){return m_gridPixelSpacing;}
        PointType getOrigin(){return m_origin;}
        int nRegNodes(){
            return m_nRegistrationNodes;
        }
        int nSegNodes(){
            return m_nSegmentationNodes;
        }
        virtual int nRegLabels(){
            return LabelMapperType::nDisplacements;
        }
        int nSegLabels(){
            return LabelMapperType::nSegmentations;
        }
    }; //GraphModel

    template<class TImage, 
             class TUnaryRegistrationFunction, 
             class TPairwiseRegistrationFunction, 
             class TUnarySegmentationFunction, 
             class TPairwiseSegmentationFunction,
             class TPairwiseCoherenceFunction,
             class TLabelMapper>
    class FastGraphModel: public GraphModel<TImage,TUnaryRegistrationFunction,TPairwiseRegistrationFunction,TUnarySegmentationFunction,TPairwiseSegmentationFunction,TPairwiseCoherenceFunction,TLabelMapper>
    {
    public:
        typedef FastGraphModel Self;
        typedef SmartPointer<Self>        Pointer;
        typedef SmartPointer<const Self>  ConstPointer;
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
        typedef TUnaryRegistrationFunction UnaryRegistrationFunctionType;
        typedef typename UnaryRegistrationFunctionType::Pointer UnaryRegistrationFunctionPointerType;
        typedef TLabelMapper LabelMapperType;
        typedef typename LabelMapperType::LabelType RegistrationLabelType;
    public:
        virtual void Init(){
            //#define moarcaching
            this->m_unaryRegFunction->setCoarseImage(this->m_coarseGraphImage);
#ifdef moarcaching
            std::vector<RegistrationLabelType> displacementList(this->m_nDisplacementLabels);
            for (int n=0;n<this->m_nDisplacementLabels;++n){
                LOGV(25)<<"Caching unary registration potentials for label "<<n<<endl;
                displacementList[n]=LabelMapperType::scaleDisplacement(LabelMapperType::getLabel(n),this->getDisplacementFactor());
            }
            this->m_unaryRegFunction->setDisplacements(displacementList);
            this->m_unaryRegFunction->compute();
#endif
            
        }
        virtual void cacheRegistrationPotentials(int labelIndex){
#ifndef moarcaching
            LOGV(25)<<"Caching unary registration function for label " << labelIndex<<endl;
            
            this->m_unaryRegFunction->cachePotentials(LabelMapperType::scaleDisplacement(LabelMapperType::getLabel(labelIndex),this->getDisplacementFactor()));
#endif
        }
        virtual double getUnaryRegistrationPotential(int nodeIndex,int labelIndex){
            IndexType index=this->getGraphIndex(nodeIndex);
            LOGV(90)<<VAR(index);
#ifdef moarcaching
            double result=  this->m_unaryRegFunction->getPotential(index,labelIndex);//this->m_nRegistrationNodes;
#else
            double result=  this->m_unaryRegFunction->getPotential(index);//this->m_nRegistrationNodes;
#endif

            if (this->m_normalizePotentials) result/=this->m_nRegistrationNodes;
            return result;

        }


    };

    
}//namespace

#endif /* GRIm_dim_H_ */
