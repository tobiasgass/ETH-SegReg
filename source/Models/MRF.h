//////////////////////////////////////////////////
//// SRS-MRF
//// Tobias Gass
//// ETH Zurich
//// No license yet!
/////////////////////////////////////////////////

#ifndef MRF_H
#define MRF_H

#include "Potentials.h"
#include "Label.h"
#include "Grid.h"


template<class TUnaryPotential, class TPairwisePotential>
class MRFSolver{

public:
	//	typedefs
	typedef TUnaryPotential UnaryPotentialType;
	typedef TPairwisePotential PairwisePotentialType;
	typedef typename UnaryPotentialType::Pointer UnaryPotentialPointerType;
	typedef typename PairwisePotentialType::Pointer PairwisePotentialPointerType;
	typedef typename TUnaryPotential::LabelConverterType LabelConverterType;
	typedef	typename LabelConverterType::ImageType ImageType;
	typedef typename ImageType::Pointer ImagePointerType;
	typedef Grid<ImageType> GridType;
	typedef typename LabelConverterType::LabelType LabelType;
	typedef typename ImageType::IndexType IndexType;

//	//this is necessary if the field element to be saved differs from the label type
//	//eg, label=offset, field=itk::vector
//	typedef typename LabelConverterType::FieldElementType FieldElementType;
//	typedef typename LabelConverterType::LabelFieldType LabelFieldType;
//	typedef typename LabelFieldType::Pointer LabelFieldPointerType;
	typedef typename itk::Image<LabelType> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;

protected:
	//	grid on which the MRF is created
	GridType * m_grid;
	//	pointers to fixed and moving images
	ImagePointerType m_fixedImage,m_movingImage;
	//	object responsible for converting labels to integer and image space indices
	LabelConverterType * m_labelConverter;
	//	potential functions
	PairwisePotentialPointerType m_pairwisePotentialFunction;
	UnaryPotentialPointerType m_unaryPotentialFunction;
	int m_nNodes;
public:
	// the constructor only sets member variables
	MRFSolver(ImagePointerType fixedImage, ImagePointerType movingImage,
			GridType * grid,PairwisePotentialPointerType pairwisePotential,
			UnaryPotentialPointerType unaryPotential)
	:m_fixedImage(fixedImage),m_movingImage(movingImage),
	 m_grid(grid),m_pairwisePotentialFunction(pairwisePotential),
	 m_unaryPotentialFunction(unaryPotential)
	{
		m_labelConverter=m_unaryPotentialFunction->getLabelConverter();
		m_nNodes=m_grid->nNodes();
	}

	//pure virtual functions have to be implemented by derived classes

	//create MRF graph, implementatiuon depends on the used optimisation library
	virtual void createGraph()=0;
	//finalize initialization and call the optimisation
	virtual void optimize()=0;

	//apply the transformation to an image, could possibly also be implemented here?
//	virtual ImagePointerType transformImage(ImagePointerType img)=0;
	//..
	virtual LabelType getLabelAtIndex(int index)=0;

	//..
//	virtual LabelFieldPointerType getLabelField(){
//		LabelFieldPointerType labelField=LabelFieldType::New();//(m_fixedImage->GetLargestPossibleRegion().GetSize());
//		labelField->SetRegions(m_fixedImage->GetLargestPossibleRegion());
//		labelField->Allocate();
//		m_grid->gotoBegin();
//		for (int i=0;i<m_nNodes;++i){
//			int index=m_grid->getIndex();
//			LabelType label=getLabelAtIndex(index);
//			FieldElementType fieldElement=m_labelConverter->getFieldElement(label);
//			labelField->SetPixel(m_grid->getGridPositionAtIndex(index),fieldElement);
//			m_grid->next();
//		}
//		return labelField;
//	}
	virtual LabelImagePointerType getLabelImage(){
			LabelImagePointerType labelImage=LabelImageType::New();
			labelImage->SetRegions(m_fixedImage->GetLargestPossibleRegion());
			labelImage->Allocate();
			m_grid->gotoBegin();
			for (int i=0;i<m_nNodes;++i){
				int index=m_grid->getIndex();
				LabelType label=getLabelAtIndex(index);
				labelImage->SetPixel(m_grid->getGridPositionAtIndex(index),label);
				m_grid->next();
			}
			return labelImage;
		}

};//MRFSolver




#endif //MRF_H
