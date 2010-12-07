/*
 * TRW-S-Registration.h
 *
 *  Created on: Dec 3, 2010
 *      Author: gasst
 */

#ifndef TRW_S_REGISTRATION_H_
#define TRW_S_REGISTRATION_H_
#include "MRF.h"
#include "typeTruncatedQuadratic2D.h"
#include "MRFEnergy.h"
#include "minimize.cpp"
#include "treeProbabilities.cpp"
template<class TUnaryPotential, class TPairwisePotential>
class TRWS_MRFSolver : public MRFSolver<TUnaryPotential,TPairwisePotential>{
public:

	typedef MRFSolver<TUnaryPotential,TPairwisePotential> Superclass;
	typedef typename Superclass::LabelConverterType LabelConverterType;
	typedef typename Superclass::PairwisePotentialPointerType PairwisePotentialPointerType;
	typedef typename Superclass::UnaryPotentialPointerType UnaryPotentialPointerType;
	typedef	typename Superclass::ImageType ImageType;
	typedef typename Superclass::ImagePointerType ImagePointerType;
	typedef typename Superclass::GridType GridType;
	typedef typename Superclass::LabelType LabelType;
	typedef typename Superclass::IndexType IndexType;
	typedef Graph::Real Real;
	typedef MRFEnergy<TypeTruncatedQuadratic2D> MRFType;
	typedef typename MRFType::NodeId NodeType;
	typedef typename Superclass::DeformationFieldType DeformationFieldType;

protected:
	int nLabels,nNodes,nPairs,labelSampling;
	MRFType* optimizer;
	NodeType* nodes;
public:
	TRWS_MRFSolver(ImagePointerType fixedImage , ImagePointerType movingImage , GridType * grid,PairwisePotentialPointerType pairwisePotential,UnaryPotentialPointerType unaryPotential)
	:Superclass(fixedImage,movingImage,grid, pairwisePotential,unaryPotential)
	{
		nLabels=this->m_labelConverter->nLabels();
		labelSampling=this->m_labelConverter->labelSampling();
		nNodes=this->m_grid->nNodes();
		nPairs=this->m_grid->nVertices();
		createGraph();
	}
	virtual void createGraph(){
		TypeTruncatedQuadratic2D::GlobalSize globalSize(labelSampling,labelSampling);
		optimizer = new MRFType(globalSize);
		nodes = new NodeType[nNodes];
		std::cout<<"starting graph init"<<std::endl;
		clock_t start = clock();

		GridType * grid=this->m_grid;
		grid->gotoBegin();
		int runningIndex=0;
		//		traverse grid
		TypeTruncatedQuadratic2D::REAL D[nLabels];
		for (int i=0;i<nNodes;++i){
			// get current indices bot integer, in the grid plane and in the image plane
			int currentIntIndex=grid->getIndex();
			IndexType currentGridIndex=grid->getCurrentGridPosition();
			IndexType currentImageIndex=grid->getCurrentImagePosition();
			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1)
			{
				LabelType label=this->m_labelConverter->getLabel(l1);
				D[l1]=this->m_unaryPotentialFunction->getPotential(currentImageIndex,label);
			}
			nodes[currentIntIndex] = optimizer->AddNode(TypeTruncatedQuadratic2D::LocalSize(), TypeTruncatedQuadratic2D::NodeData(D));
			grid->next();
		}
		clock_t finish1 = clock();
		float t = (float) ((double)(finish1 - start) / CLOCKS_PER_SEC);
		std::cout<<"Finished unary potential initialisation after "<<t<<" seconds"<<std::endl;
		grid->gotoBegin();
		double weight=5000;
		for (int i=0;i<nNodes;++i){
			int currentIntIndex=grid->getIndex();
			std::vector<int> neighbours= grid->getCurrentForwardNeighbours();
			int nNeighbours=neighbours.size();
			for (int i=0;i<nNeighbours;++i){
				//				std::cout<<"adding edge, "<<currentIntIndex<< " to "<<neighbours[i]<<std::endl;
				optimizer->AddEdge(nodes[currentIntIndex], nodes[neighbours[i]], TypeTruncatedQuadratic2D::EdgeData(weight, weight, 8*weight));
			}
			grid->next();
		}
		clock_t finish = clock();
		t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		std::cout<<"Finished init after "<<t<<" seconds"<<std::endl;

	}

	virtual void optimize(){
		MRFEnergy<TypeTruncatedQuadratic2D>::Options options;
		TypeTruncatedQuadratic2D::REAL energy, lowerBound;
		options.m_iterMax = 10; // maximum number of iterations
		clock_t start = clock();
		optimizer->Minimize_TRW_S(options, lowerBound, energy);
		clock_t finish = clock();
		float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		std::cout<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound"<< lowerBound << std::endl;

	}
	ImagePointerType transformImage(ImagePointerType img){
		ImagePointerType transformedImage(this->m_fixedImage);
		GridType * grid=this->m_grid;
		grid->gotoBegin();
		for (int i=0;i<nNodes;++i){
			int currentIntIndex=grid->getIndex();
			IndexType currentImageIndex=grid->getCurrentImagePosition();
			TypeTruncatedQuadratic2D::Label l=optimizer->GetSolution(nodes[currentIntIndex]);
			int labelIndex=l.m_kx+l.m_ky*labelSampling;

			LabelType label=this->m_labelConverter->getLabel(labelIndex);
			IndexType movingIndex=this->m_labelConverter->getMovingIndex(currentImageIndex,labelIndex);

			transformedImage->SetPixel(currentImageIndex,img->GetPixel(movingIndex));//(label[1]+15)*65535/15);//img->GetPixel(movingIndex));
			grid->next();
		}
		return transformedImage;
	}
	virtual LabelType getLabelAtIndex(int index){
		TypeTruncatedQuadratic2D::Label l=optimizer->GetSolution(nodes[index]);
		int labelIndex=l.m_kx+l.m_ky*labelSampling;
		return this->m_labelConverter->getLabel(labelIndex);

	}
};

#endif /* TRW_S_REGISTRATION_H_ */
