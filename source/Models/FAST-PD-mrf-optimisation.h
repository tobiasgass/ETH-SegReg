/*
 * FAST-PD-Registration-mrf.h
 *
 *  Created on: Nov 30, 2010
 *      Author: gasst
 */

#ifndef FAST_PD_REGISTRATION_MRF_H_
#define FAST_PD_REGISTRATION_MRF_H_
#include "MRF.h"
#include "Fast_PD2.h"

template<class TUnaryPotential, class TPairwisePotential>
class FastPDMRFSolver : public MRFSolver<TUnaryPotential,TPairwisePotential>{
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
//	typedef typename Superclass::LabelFieldType LabelFieldType;

	typedef Graph::Real Real;

protected:
	int nLabels,nNodes,nPairs;
	std::vector<int> pairs;
	Real * unaryPotentials;
	Real * pairwisePotentials;
	Real * pairwisePotentials2;

	std::vector<Real>  edgeWeights;
	CV_Fast_PD * optimizer;
	double m_pairwiseWeight,m_unaryWeight;
	bool secondPairwise;
public:
	FastPDMRFSolver(ImagePointerType fixedImage , ImagePointerType movingImage , GridType * grid,PairwisePotentialPointerType pairwisePotential,UnaryPotentialPointerType unaryPotential, double unaryWeight=1.0, double pairwiseWeight=1.0, bool secondPairwisePotential=false)
	:Superclass(fixedImage,movingImage,grid, pairwisePotential,unaryPotential),secondPairwise(secondPairwisePotential)
	{
		nLabels=this->m_labelConverter->nLabels();
		nNodes=this->m_nNodes;
		nPairs=this->m_grid->nVertices();
		//		pairs is an array of the form [na nb nc nd...] where (na,nb),(nc,nd) are edges in the graph and na,... are indices of the nodes
		std::cout<<"allocating pairs"<<std::endl;
		pairs=std::vector<int>(nPairs*2);
		std::cout<<"allocating unary potentials"<<std::endl;
		unaryPotentials=new Real[nNodes*nLabels];
		std::cout<<"allocating pairwise potentials"<<std::endl;
		pairwisePotentials= new Real[nLabels*nLabels];
		if (secondPairwise)
			pairwisePotentials2= new Real[nLabels*nLabels];
		std::cout<<"allocating edgeweights"<<std::endl;
		edgeWeights=std::vector<Real>(nPairs,1.0);//new Real[nPairs];//={1.0};
		m_unaryWeight=unaryWeight;
		m_pairwiseWeight=pairwiseWeight;
		std::cout<<nNodes<<" "<<nLabels<<" "<<std::endl;
		//memset( edgeWeights, 1.0,nPairs*sizeof(Real) );

		createGraph();
	}
	/*
	 * initialize graph, compute potentials, set up neighbourhood structure
	 */
	virtual void createGraph(){
		std::cout<<"starting graph init"<<std::endl;
		GridType * grid=this->m_grid;
		grid->gotoBegin();
		int runningIndex=0;
		//		traverse grid
		while(!grid->atEnd()){
			// get current indices both integer, in the grid plane and in the image plane
			int currentIntIndex=grid->getIndex();
			IndexType currentGridIndex=grid->getCurrentGridPosition();
			IndexType currentImageIndex=grid->getCurrentImagePosition();
			// get forward neighbors of current grid point, both grid index and image plane index
			std::vector<int> neighbours= grid->getCurrentForwardNeighbours();
			int nNeighbours=neighbours.size();
			for (int i=0;i<nNeighbours;++i){
				pairs[runningIndex+i*2]=currentIntIndex;
				pairs[runningIndex+i*2+1]=neighbours[i];
				edgeWeights[(runningIndex+i*2)/2]=this->m_pairwisePotentialFunction->getWeight(currentIntIndex,neighbours[i]);
			}
			runningIndex+=nNeighbours*2;
			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1){
				LabelType label=this->m_labelConverter->getLabel(l1);
				unaryPotentials[l1*nNodes+currentIntIndex]=m_unaryWeight*this->m_unaryPotentialFunction->getPotential(currentImageIndex,label);
			}
			grid->next();
		}
		for (int i=0;i<nNodes*nLabels;++i){
			if (unaryPotentials[i]>1 || unaryPotentials[i]<0){
			}
		}
		std::cout<<"initialised basic graph structure and unary potentials"<<std::endl;
		//		traverse labels
		for (int l1=0;l1<nLabels;++l1){
			for (int l2=0;l2<nLabels;++l2){
				pairwisePotentials[l1*nLabels+l2]=m_pairwiseWeight*this->m_pairwisePotentialFunction->getPotential(this->m_labelConverter->getLabel(l1),this->m_labelConverter->getLabel(l2));
				if (secondPairwise){
					pairwisePotentials2[l1*nLabels+l2]=m_pairwiseWeight*this->m_pairwisePotentialFunction->getPotential2(this->m_labelConverter->getLabel(l1),this->m_labelConverter->getLabel(l2));
				}
			}
		}
		//create optimizer object
		std::cout<<"initialising fastPD with "<<nNodes<<" nodes, "<< nLabels<<" labels, "<<nPairs<<" pairs"<<std::endl;
		if (secondPairwise){
			optimizer= new CV_Fast_PD(nNodes,nLabels,unaryPotentials,nPairs,&pairs[0],pairwisePotentials, pairwisePotentials2,20,&edgeWeights[0],int(nLabels/4));
		}else{
			optimizer= new CV_Fast_PD(nNodes,nLabels,unaryPotentials,nPairs,&pairs[0],pairwisePotentials, pairwisePotentials,20,&edgeWeights[0],int(nLabels/4));
		}
		this->m_unaryPotentialFunction->freeMemory();
	}
	virtual void optimize(){
		optimizer->run();
	}


	virtual LabelType getLabelAtIndex(int index){
		return this->m_labelConverter->getLabel(optimizer->_pinfo[index].label);
	}
};

#endif /* FAST_PD_REGISTRATION_MRF_H_ */
