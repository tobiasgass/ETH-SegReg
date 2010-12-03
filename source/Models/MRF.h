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

protected:
	GridType * m_grid;
	ImagePointerType m_fixedImage,m_movingImage;
	LabelConverterType * m_labelConverter;
	PairwisePotentialPointerType m_pairwisePotentialFunction;
	UnaryPotentialPointerType m_unaryPotentialFunction;
public:
	MRFSolver(ImagePointerType fixedImage, ImagePointerType movingImage,
			GridType * grid,PairwisePotentialPointerType pairwisePotential,
			UnaryPotentialPointerType unaryPotential)
	:m_fixedImage(fixedImage),m_movingImage(movingImage),
	 m_grid(grid),m_pairwisePotentialFunction(pairwisePotential),
	 m_unaryPotentialFunction(unaryPotential)
	{
		m_labelConverter=new LabelConverterType(fixedImage,movingImage,grid->getResolution(),10);
	}
	virtual void createGraph()=0;
	virtual void optimize()=0;
	virtual ImagePointerType transformImage(ImagePointerType img)=0;



};//MRFSolver




#endif //MRF_H
