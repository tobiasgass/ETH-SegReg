/*
 * Graph.h
 *
 *  Created on: Nov 25, 2010
 *      Author: gasst
 */

#pragma once

#include "Log.h"

#include <vector>
#include <assert.h>
#include "itkConstNeighborhoodIterator.h"
#include <limits>
#include "SRSConfig.h"
#include "Log.h"
#include "TransformationUtils.h"
#include "Potential-Registration-Unary.h"
#include "Potential-Registration-Pairwise.h"
#include "Potential-Segmentation-Unary.h"
#include "Potential-Segmentation-Pairwise.h"
#include "Potential-Coherence-Pairwise.h"
#include "BaseLabel.h"



namespace SRS{

  /** \brief
   * General Graph class which provides access to potential functions and maps from node space to image space
   * 
   * 
   */
  template<class TImage,
    class TUnaryRegistrationFunction=FastUnaryPotentialRegistrationNCC<TImage>,
    class TPairwiseRegistrationFunction= PairwisePotentialRegistration<TImage>,
    class TUnarySegmentationFunction=UnaryPotentialSegmentation<TImage>,
    class TPairwiseSegmentationFunction=PairwisePotentialSegmentation<TImage>,
    class TPairwiseCoherenceFunction=PairwisePotentialCoherence<TImage> >
    class GraphModel: public itk::Object{
  public:
  typedef GraphModel Self;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;
  itkNewMacro(Self);

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
    
  typedef typename TransfUtils<ImageType>::DisplacementType RegistrationLabelType;
  typedef BaseLabelMapper<ImageType,RegistrationLabelType> LabelMapperType;

  typedef typename itk::Image<RegistrationLabelType,ImageType::ImageDimension> RegistrationLabelImageType;
  typedef typename RegistrationLabelImageType::Pointer RegistrationLabelImagePointerType;
        
  typedef int SegmentationLabelType;
  typedef typename itk::Image<SegmentationLabelType,ImageType::ImageDimension> SegmentationLabelImageType;
  typedef typename SegmentationLabelImageType::Pointer SegmentationLabelImagePointerType;

  typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
  typedef typename FloatImageType::Pointer FloatImagePointerType;
  typedef typename ImageUtils<FloatImageType>::ImageIteratorType FloatImageIteratorType;
    
  protected:
  LabelMapperType * m_labelMapper;
    
  SizeType m_totalSize,m_imageLevelDivisors,m_graphLevelDivisors,m_gridSize, m_imageSize;
  ImagePointerType m_coarseGraphImage,m_borderOfSegmentationROI;
  SpacingType m_gridPixelSpacing;        //grid spacing in unit pixels
  SpacingType m_gridSpacing, m_imageSpacing;         //grid spacing in mm
  SpacingType m_labelSpacing;        //spacing of the displacement labels
  PointType m_origin;
  double m_DisplacementScalingFactor;
  static const unsigned int m_dim=TImage::ImageDimension;
  int m_nNodes,m_nVertices, m_nRegistrationNodes, m_nSegmentationNodes;
  int m_nRegEdges,m_nSegEdges,m_nSegRegEdges, m_nEdges;
  int m_nSegmentationLabels,m_nDisplacementLabels,m_nDisplacementSamplesPerAxis;

  double m_segmentationUnaryNormalizer;

  UnaryRegistrationFunctionPointerType m_unaryRegFunction;
  UnarySegmentationFunctionPointerType m_unarySegFunction;
  PairwiseSegmentationFunctionPointerType m_pairwiseSegFunction;
  PairwiseCoherenceFunctionPointerType m_pairwiseSegRegFunction;
  PairwiseRegistrationFunctionPointerType m_pairwiseRegFunction;
  
  bool verbose;
  bool m_haveLabelMap;
  ConstImagePointerType m_targetImage,m_targetSegmentationImage;
  ConstImageNeighborhoodIteratorType m_targetNeighborhoodIterator;
  SRSConfig m_config;
  int m_maxRegSegNeighbors;

  bool m_normalizePotentials;

  std::vector<int> m_mapIdx1,m_mapIdx1Rev;
  bool m_reducedSegNodes;
  double m_coherenceThresh;

  public:
  int getMaxRegSegNeighbors(){return m_maxRegSegNeighbors;}
  GraphModel(){
    assert(m_dim>1);
    assert(m_dim<4);
    m_haveLabelMap=false;
    verbose=false;
    
    m_DisplacementScalingFactor=1.0;
    m_normalizePotentials=false;
    m_reducedSegNodes=false;
    m_labelMapper=NULL;
  };
  ~GraphModel(){
  }
  void setConfig(SRSConfig c){
    m_config=c;
    verbose=c.verbose;
    m_normalizePotentials=c.normalizePotentials; 
  }
  void setTargetImage(ConstImagePointerType targetImage){
    m_targetImage=targetImage;
  }
  LabelMapperType * getLabelMapper(){return m_labelMapper;}
  void setLabelMapper( LabelMapperType * lm){m_labelMapper=lm;}
        
  ImagePointerType getCoarseGraphImage(){ return this->m_coarseGraphImage;}
  void initGraph(int nGraphNodesPerEdge){
    if (!m_labelMapper){
      LOG<<"ERROR: Labelmapper not set"<<endl;
      exit(0);
    }
    m_nSegmentationLabels=m_labelMapper->getNumberOfSegmentationLabels();
    m_nDisplacementLabels=m_labelMapper->getNumberOfDisplacementLabels();
    m_nDisplacementSamplesPerAxis=m_labelMapper->getNumberOfDisplacementSamplesPerAxis();
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
    if (this->m_nDisplacementSamplesPerAxis){
#ifdef PIXELTRANSFORM
      m_labelSpacing=0.4*m_gridPixelSpacing/(this->m_nDisplacementSamplesPerAxis);
      LOGV(1)<<"Graph pixel spacing :"<<m_gridPixelSpacing<<std::endl; 

#else
      m_labelSpacing=0.4*m_gridSpacing/(this->m_nDisplacementSamplesPerAxis);
#endif
      LOGV(1)<<this->m_nDisplacementSamplesPerAxis<<" displacment samples per direction; "<<"with "<<m_labelSpacing<<" pixels spacing"<<std::endl;
      LOGV(1)<<"Max displacement per axis: " <<m_labelSpacing * this->m_nDisplacementSamplesPerAxis * m_DisplacementScalingFactor <<"mm" << endl;
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
    LOGV(2)<<" totalNodes:"<<m_nNodes<<" totalEdges:"<<m_nRegEdges+m_nSegEdges+m_nSegRegEdges<<" labels:"<<this->m_labelMapper->getTotalNumberOfLabels()<<std::endl;
    LOGV(2)<<" Segnodes:"<<m_nSegmentationNodes<<"\t SegEdges :"<<m_nSegEdges<<std::endl ;
    LOGV(2) <<" Regnodes:"<<m_nRegistrationNodes<<"\t\t RegEdges :"<<m_nRegEdges<<std::endl;
    LOGV(2)                <<" SegRegEdges:"<<m_nSegRegEdges<<std::endl;
                         
        
    m_segmentationUnaryNormalizer=m_nSegmentationNodes;
    LOGV(1)<<" finished graph init" <<std::endl;
    logResetStage;
  }
  ///can be used to initialize stuff right before potentials are called
  void Init(){};
  ///set coarse graph size/resolution/spacing based on target image and desired number of nodes on the shortest edge
  void setSpacing(int shortestN){
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

    //calculate spacingq and size for all image dimensions using
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

  ///reduces the nodes for which segmentation labels are computed, based on the coherence potential
  void ReduceSegmentationNodesByCoherencePotential(double thresh){
    m_coherenceThresh=thresh;
    LOGV(1)<<"Removing all segmentation nodes with coherence potential larger "<<thresh<<" for all non-aux labels."<<endl;

    //get distance transform potential for neutral deformation
    FloatImagePointerType dist=m_pairwiseSegRegFunction->GetDistanceTransform(0);
    m_reducedSegNodes=false;
    m_borderOfSegmentationROI=FilterUtils<ImageType>::createEmpty(m_targetImage);
    m_borderOfSegmentationROI->FillBuffer(0);
    int actualIdx=0,concurrentIdx=0;
    int nNodes=this->m_targetImage->GetLargestPossibleRegion().GetNumberOfPixels();

    //allocate forward and backward map to map consecutive node indices to actual node indices
    m_mapIdx1=std::vector<int>(nNodes,-1);
    m_mapIdx1Rev=std::vector<int>(nNodes,-1);
    //iterate over all nodes
    for (;actualIdx<nNodes;++actualIdx){
                
      IndexType position1=getImageIndex(actualIdx);
      PointType pt;
      m_targetImage->TransformIndexToPhysicalPoint(position1,pt);
             
      float distAtPos=m_pairwiseSegRegFunction->getMinZeroPotential(pt);//dist->GetPixel(position2);
      LOGV(9)<<VAR(distAtPos)<<" "<<VAR(thresh)<<endl;
      //if the minimum coherence potential at that node is smaller than the threshold, add it to the list of nodes for which a segmentation is to be computed
      if (distAtPos<thresh){
	m_mapIdx1[actualIdx]=concurrentIdx;
	m_mapIdx1Rev[concurrentIdx]=actualIdx;
	++concurrentIdx;
	//add this node to an image such that a mask is created
	m_borderOfSegmentationROI->SetPixel(position1,1);
                    
      }
    }
    LOGI(6,ImageUtils<ImageType>::writeImage("ROI.nii",m_borderOfSegmentationROI));
    ///erode/dilate mask such that only the 1-pixel border remains
    ///this allows for clamping the segmentation labels of that border to the atlas segmentation
    ///erosion can give strange results, fixing by thresholding
    m_borderOfSegmentationROI=FilterUtils<ImageType>::substract(m_borderOfSegmentationROI,FilterUtils<ImageType>::binaryThresholding(FilterUtils<ImageType>::erosion(m_borderOfSegmentationROI,2),1,1));

    LOGI(6,ImageUtils<ImageType>::writeImage("BorderOfSegmentationROI.nii",m_borderOfSegmentationROI));
    m_nSegmentationNodes=concurrentIdx;
    LOG<<"Reduced number of segmentation nodes to "<<100.0*concurrentIdx/actualIdx<<"%; "<<actualIdx<<"->"<<concurrentIdx<<endl;
    m_mapIdx1Rev.resize(concurrentIdx);
    m_reducedSegNodes=true;

  }
     
  ///return position index in coarse graph from coarse graph node index
  inline  IndexType  getGraphIndex(int nodeIndex){
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
  inline IndexType  getImageIndexFromCoarseGraphIndex(int idx){
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

  /// return nearest neighbor node of the coarse graph to an image index
  inline IndexType getClosestGraphIndex(IndexType imageIndex){
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

  ///return 'previous' node of the coarse graph
  inline IndexType getLowerGraphIndex(IndexType imageIndex){
    IndexType position;
    for (unsigned int d=0;d<m_dim;++d){
      position[d]=int(imageIndex[d]*m_imageSpacing[d]/m_gridSpacing[d]);
      //position[d]<<std::target << std::setprecision(0) << imageIndex[d]*m_imageSpacing[d]/m_gridSpacing[d];
    }
    return position;
  }

  /// get integer index of a graph node
  inline int  getGraphIntegerIndex(IndexType gridIndex){
    int i=0;
    for (unsigned int d=0;d<m_dim;++d){
      i+=gridIndex[d]*m_graphLevelDivisors[d];
    }
    return i;
  }

  //get integer index of the target image
  inline int  getImageIntegerIndex(IndexType imageIndex){
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
  inline IndexType getImageIndex(int idx){
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

  /**
   * Get Unary registration potential for node/label combination
   */
  inline double getUnaryRegistrationPotential(int nodeIndex,int labelIndex){
    IndexType imageIndex=getImageIndexFromCoarseGraphIndex(nodeIndex);
    RegistrationLabelType l=this->m_labelMapper->getLabel(labelIndex);
    l=this->m_labelMapper->scaleDisplacement(l,getDisplacementFactor());
    double result=m_unaryRegFunction->getPotential(imageIndex,l);
    if (m_normalizePotentials) result/=m_nRegistrationNodes;
    return result;//m_nRegistrationNodes;
  }

  /**
   * Get Unary segmentation potential for node/label combination
   */
  inline double getUnarySegmentationPotential(int nodeIndex,int labelIndex){
    IndexType imageIndex=getImageIndex(nodeIndex);
             
    /// use labelIndex from provided segmentation image if it is set, overwriting the input labelIndex
    /// this is needed for ARS, where a segmentation estimate of the target image is availble
    if (m_targetSegmentationImage.IsNotNull()){
      labelIndex=m_targetSegmentationImage->GetPixel(imageIndex);
    }

    /// return a large potential if segmentation nodes are reduced and the current node/label combination has a coherence potential larger than m_coherenceThresh
    if ( m_reducedSegNodes ){
      if (sqrt(2*m_pairwiseSegRegFunction->getPotential(imageIndex,IndexType(),this->m_labelMapper->getZeroDisplacement(),labelIndex))>m_coherenceThresh)
	//inelegant solution! This returns a large magic number to the optimizer in case the pixel is _outside_ of the ROI, and a slightly smaller if inside.
	if (this->m_borderOfSegmentationROI->GetPixel(imageIndex)){
	  return 9999;
	}else{
	  return 10000;
	}
    }
                    


    //Segmentation:labelIndex==segmentationlabel
    double result=m_unarySegFunction->getPotential(imageIndex,labelIndex);// /m_nSegmentationNodes;
    if (result<0){
      LOG<<"unary segmentation potential <0"<<std::endl;
      LOG<<imageIndex<<" " <<result<<std::endl;
    }
    if (m_normalizePotentials) result/=m_segmentationUnaryNormalizer;

    return result;
  };

  /**
   * Get pairwise registration potential for node/label,node/label combination
   */
  inline double getPairwiseRegistrationPotential(int nodeIndex1, int nodeIndex2, int labelIndex1, int labelIndex2){
            
    /// get graph coordinates
    IndexType graphIndex1=getGraphIndex(nodeIndex1);
    IndexType graphIndex2=getGraphIndex(nodeIndex2);
    ///get physical coordinates
    PointType pt1,pt2;
    this->m_coarseGraphImage->TransformIndexToPhysicalPoint(graphIndex1,pt1);
    this->m_coarseGraphImage->TransformIndexToPhysicalPoint(graphIndex2,pt2);
    /// get displacement vectors
    RegistrationLabelType l2=this->m_labelMapper->getLabel(labelIndex2);
    l2=this->m_labelMapper->scaleDisplacement(l2,getDisplacementFactor());
    RegistrationLabelType l1=this->m_labelMapper->getLabel(labelIndex1);
    l1=this->m_labelMapper->scaleDisplacement(l1,getDisplacementFactor());
    //return m_pairwiseRegFunction->getPotential(graphIndex1, graphIndex2, l1,l2);//m_nRegEdges;
    double result=m_pairwiseRegFunction->getPotential(pt1, pt2, l1,l2);
    if (m_normalizePotentials) result/=m_nRegEdges;
    return result;
  };

   /**
   * Get pairwise coherence potential for seg node/label, reg node/label combination
   */
  double getPairwiseSegRegPotential(int nodeIndex1, int nodeIndex2, int labelIndex1, int segmentationLabel){
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
    RegistrationLabelType registrationLabel=this->m_labelMapper->getLabel(labelIndex1);
    registrationLabel=this->m_labelMapper->scaleDisplacement(registrationLabel,getDisplacementFactor());
    double result=m_pairwiseSegRegFunction->getPotential(imageIndex,imageIndex,registrationLabel,segmentationLabel);//m_nSegRegEdges;
    if (m_normalizePotentials) result/=m_nSegRegEdges;
    return result;
            
    //        return m_pairwiseSegRegFunction->getPotential(graphIndex,imageIndex,registrationLabel,segmentationLabel)/m_nSegRegEdges;
  }

  //#define MULTISEGREGNEIGHBORS
   /**
   * Get pairwise coherence potential for reg node/label, seg node/label combination
   */
  inline double getPairwiseRegSegPotential(int nodeIndex1, int nodeIndex2, int labelIndex1, int segmentationLabel){
    
    IndexType imageIndex=getImageIndex(nodeIndex2);
    if (m_targetSegmentationImage.IsNotNull()){
      segmentationLabel=m_targetSegmentationImage->GetPixel(imageIndex);
    }
    //compute distance between center index and patch index
    double weight=1.0;
    // #ifdef MULTISEGREGNEIGHBORS
#if 0
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
    RegistrationLabelType registrationLabel=this->m_labelMapper->getLabel(labelIndex1);
    registrationLabel=this->m_labelMapper->scaleDisplacement(registrationLabel,getDisplacementFactor());
    double result = weight*m_pairwiseSegRegFunction->getPotential(imageIndex,imageIndex,registrationLabel,segmentationLabel);//m_nSegRegEdges;
    //        return m_pairwiseSegRegFunction->getPotential(graphIndex,imageIndex,registrationLabel,segmentationLabel)/m_nSegRegEdges;
    if (m_normalizePotentials) result/=m_nSegRegEdges;
    return result;
  }

   /**
   * Get pairwise coherence potential for reg node/label, seg label combination
   * simplified calculation when NN interpolation is used and the segmentationnode can be directly inferred from the registration node (closest)
   */
  inline double getPairwiseRegSegPotential(int nodeIndex2, int labelIndex1, int segmentationLabel){
    
    IndexType imageIndex=getImageIndex(nodeIndex2);
    if (m_targetSegmentationImage.IsNotNull()){
      segmentationLabel=m_targetSegmentationImage->GetPixel(imageIndex);
    }
    //compute distance between center index and patch index
    double weight=1.0;
    //    #ifdef MULTISEGREGNEIGHBORS
#if 0
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
    RegistrationLabelType registrationLabel=this->m_labelMapper->getLabel(labelIndex1);
    registrationLabel=this->m_labelMapper->scaleDisplacement(registrationLabel,getDisplacementFactor());
    double result =  weight*m_pairwiseSegRegFunction->getPotential(imageIndex,imageIndex,registrationLabel,segmentationLabel);//m_nSegRegEdges;
    //        return m_pairwiseSegRegFunction->getPotential(graphIndex,imageIndex,registrationLabel,segmentationLabel)/m_nSegRegEdges;
    if (m_normalizePotentials) result/=m_nSegRegEdges;
    return result;
  }
        
   /**
   * Get pairwise segmentation potential for seg node/label, seg node/label combination
   */
  inline double getPairwiseSegmentationPotential(int nodeIndex1, int nodeIndex2, int label1, int label2){

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

   /**
   * DEPRECATED
   */
  inline double getSegmentationWeight(int nodeIndex1, int nodeIndex2){
    IndexType imageIndex1=getImageIndex(nodeIndex1);
    IndexType imageIndex2=getImageIndex(nodeIndex2);
    double result=m_unarySegFunction->getWeight(imageIndex1,imageIndex2)/m_nSegEdges;
    LOG<<"I don't think we should be here..."<<std::endl;
    if (result<0){
                
      LOG<<imageIndex1<<" "<<imageIndex2<<" "<<result<<std::endl;
    }
    return result;
  }
   
   /**
   * get list of neighbors of a registration node according to internal neighborhood structure
   * only nodes with an index greater than the input nodes are considered since in MRFs the edges are usually unidirectional
   */
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

   /**
   * get list of neighbors of a segmentation node according to internal neighborhood structure
   * only nodes with an index greater than the input nodes are considered since in MRFs the edges are usually unidirectional
   */
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

   /**
   * get list of neighbors of a registration node in the segmentation graph according to internal neighborhood structure
   */
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

  /**
   * get list of neighbors of a segmentation node in the registration graph according to internal neighborhood structure
   */
  std::vector<int> getSegRegNeighbors(int index){
    std::vector<int> neighbours;
#ifdef MULTISEGREGNEIGHBORS
    ///only valid if a segmentation node can have multiple registration graph neighbors, eg when linear++ interpolation is used
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
    /// standard NN interpolation, only one neighbor
    IndexType position=getClosestGraphIndex(idx);
    neighbours.push_back(getGraphIntegerIndex(position));
 
#endif
    return neighbours;
  }

  ///convert result label vector into a displacement field
  RegistrationLabelImagePointerType getDeformationImage(std::vector<int>  labels){
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
      RegistrationLabelType l=this->m_labelMapper->getLabel(labels[i]);
      l=this->m_labelMapper->scaleDisplacement(l,getDisplacementFactor());
      it.Set(l);
    }
    assert(i==(labels.size()));
    //LOGV(8)<<"git "<<labels.size()<<" registration labels which were transformed into a deformation field with parameters : "<<result<<endl;
    return result;
  }
        
  //empty deformation image
  RegistrationLabelImagePointerType getDeformationImage(){
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
      RegistrationLabelType l;
      l.Fill(0);
      it.Set(l);
    }
    return result;
  }

  ///get empty image of the grid which can hold the labels of the coarse grid
  ImagePointerType getParameterImage(){
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
        
  ///convert vector of segmentation node labels into a segmentation image
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
    LOGV(1)<<"Max displacement per axis: " <<m_labelSpacing * this->m_nDisplacementSamplesPerAxis * m_DisplacementScalingFactor <<"mm" << endl;
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
  int nRegLabels(){
    return this->m_nDisplacementLabels;
  }
  int nSegLabels(){
    return this->m_nSegmentationLabels;
  }
  }; //GraphModel

    
}//namespace


