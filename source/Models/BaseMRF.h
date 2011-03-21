//////////////////////////////////////////////////
//// SRS-MRF
//// Tobias Gass
//// ETH Zurich
//// No license yet!
/////////////////////////////////////////////////

#ifndef BASEMRF_H
#define BASEMRF_H

#include "Graph.h"


template<class TGraphModel>
class BaseMRFSolver{

public:
	//	typedefs
	typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::ImageType ImageType;
	typedef typename GraphModelType::LabelType LabelType;
//	typedef typename itk::Image<LabelType, ImageType::ImageDimension> LabelImageType;
	typedef typename GraphModelType::LabelImageType LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
protected:
	int m_nNodes,m_nLabels,m_nPairs;
	GraphModelType * m_GraphModel;
public:
	// the constructor only sets member variables
	BaseMRFSolver(GraphModelType * graphModel):m_GraphModel(graphModel)
	{
		m_nNodes=m_GraphModel->nNodes();
		m_nLabels=LabelType::nLabels;
		this->m_nPairs=this->m_GraphModel->nVertices();

	}

	//pure virtual functions have to be implemented by derived classes

	//create MRF graph, implementatiuon depends on the used optimisation library
	virtual void createGraph()=0;
	//finalize initialization and call the optimisation
	virtual void optimize()=0;


	virtual LabelType getLabelAtIndex(int index)=0;

	virtual LabelImagePointerType getLabelImage(){
			LabelImagePointerType labelImage=LabelImageType::New();
			typename LabelImageType::RegionType region;
			typename LabelImageType::SizeType size=m_GraphModel->getGridSize();//=m_GraphModel->getFixedImage()->GetLargestPossibleRegion().GetSize();

			typename LabelImageType::PointType origin;
			region.SetSize(size);//m_GraphModel->getSpacing());
			labelImage->SetRegions(region);
			typename LabelImageType::SpacingType spacing=(m_GraphModel->getSpacing());
			for (int d=0;d<LabelImageType::ImageDimension;++d){
				spacing[d]=spacing[d];
				origin[d]=100;
			}
			labelImage->SetSpacing(spacing);
//			std::cout<<size<<" "<<m_GraphModel->getSpacing()<<std::endl;
			labelImage->Allocate();
			itk::ImageRegionIterator<LabelImageType>  it( labelImage, labelImage->GetLargestPossibleRegion() );
			it.GoToBegin();
			typedef typename LabelImageType::PixelType PixelType;
			typedef typename ImageType::IndexType IndexType;
			for (int i=0;i<m_nNodes;++i){
				LabelType label=getLabelAtIndex(i);
				IndexType idx=m_GraphModel->getGridPositionAtIndex(i);
//				std::cout<<i<<" "<<it.GetIndex()<<" "<<idx<<" "<<label<<std::endl;
				it.Set(label);
//				labelImage->SetPixel(idx,label);
				++it;

			}
			return labelImage;
		}

};//MRFSolver




#endif //MRF_H
