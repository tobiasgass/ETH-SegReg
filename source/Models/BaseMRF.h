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
	typedef typename itk::VectorImage<LabelType, ImageType::ImageDimension> LabelImageType;
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
			labelImage->SetRegions(m_GraphModel->getFixedImage()->GetLargestPossibleRegion());
			labelImage->SetSpacing(1.0);
			labelImage->Allocate();
			itk::ImageRegionIterator<LabelImageType>  it( labelImage, labelImage->GetLargestPossibleRegion() );
			it.GoToBegin();
			for (int i=0;i<m_nNodes;++i){
				LabelType label=getLabelAtIndex(i);
				it.Set(label);
				++it;
			}
#if 0
			m_grid->gotoBegin();
			for (int i=0;i<m_nNodes;++i){
				int index=m_grid->getIndex();
				LabelType label=getLabelAtIndex(index);
				labelImage->SetPixel(m_grid->getGridPositionAtIndex(index),label);
				m_grid->next();
			}
#endif
			return labelImage;
		}

};//MRFSolver




#endif //MRF_H
