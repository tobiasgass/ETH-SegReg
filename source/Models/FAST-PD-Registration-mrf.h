/*
 * FAST-PD-Registration-mrf.h
 *
 *  Created on: Nov 30, 2010
 *      Author: gasst
 */

#ifndef FAST_PD_REGISTRATION_MRF_H_
#define FAST_PD_REGISTRATION_MRF_H_
#include "MRF.h"
#include "Fast_PD.h"

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
	typedef Graph::Real Real;

protected:
	int nLabels,nNodes,nPairs;
	std::vector<int> pairs;
	Real * unaryPotentials;
	Real * pairwisePotentials;
	std::vector<Real>  edgeWeights;
	CV_Fast_PD * optimizer;
public:
	FastPDMRFSolver(ImagePointerType fixedImage , ImagePointerType movingImage , GridType * grid,PairwisePotentialPointerType pairwisePotential,UnaryPotentialPointerType unaryPotential)
	:Superclass(fixedImage,movingImage,grid, pairwisePotential,unaryPotential)
	{
		nLabels=this->m_labelConverter->nLabels();
		nNodes=this->m_grid->nNodes();
		nPairs=this->m_grid->nVertices();
		//		pairs is an array of the form [na nb nc nd...] where (na,nb),(nc,nd) are edges in the graph and na,... are indices of the nodes
		pairs=std::vector<int>(nPairs*2);
		unaryPotentials=new Real[nNodes*nLabels];
		pairwisePotentials= new Real[nLabels*nLabels];
		edgeWeights=std::vector<Real>(nPairs,10);//new Real[nPairs];//={1.0};
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
			// get current indices bot integer, in the grid plane and in the image plane
			int currentIntIndex=grid->getIndex();
			IndexType currentGridIndex=grid->getCurrentGridPosition();
			IndexType currentImageIndex=grid->getCurrentImagePosition();
			// get forward neighbours of current grid point, both grid index and image plane index
			std::vector<int> neighbours= grid->getCurrentForwardNeighbours();
			int nNeighbours=neighbours.size();
			for (int i=0;i<nNeighbours;++i){
				pairs[runningIndex+i*2]=currentIntIndex;
				pairs[runningIndex+i*2+1]=neighbours[i];
			}
			runningIndex+=nNeighbours*2;

			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1){
				LabelType label=this->m_labelConverter->getLabel(l1);
				unaryPotentials[l1*nNodes+currentIntIndex]=this->m_unaryPotentialFunction->getPotential(currentImageIndex,label);
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
				pairwisePotentials[l1*nLabels+l2]=this->m_pairwisePotentialFunction->getPotential(this->m_labelConverter->getLabel(l1),this->m_labelConverter->getLabel(l2));

			}
		}

		//create optimizer object
		std::cout<<"initialising fastPD with "<<nNodes<<" nodes, "<< nLabels<<" labels, "<<nPairs<<" pairs"<<std::endl;
		optimizer= new CV_Fast_PD(nNodes,nLabels,unaryPotentials,nPairs,&pairs[0],pairwisePotentials,20,&edgeWeights[0]);
	}
	virtual void optimize(){
		optimizer->run();
	}
	ImagePointerType transformImage(ImagePointerType img){
		ImagePointerType transformedImage(this->m_fixedImage);
		GridType * grid=this->m_grid;
		grid->gotoBegin();
		while(!grid->atEnd()){
			int currentIntIndex=grid->getIndex();
			IndexType currentImageIndex=grid->getCurrentImagePosition();
			int labelIndex=optimizer->_pinfo[currentIntIndex].label;
			LabelType label=this->m_labelConverter->getLabel(labelIndex);
			IndexType movingIndex=this->m_labelConverter->getMovingIndex(currentImageIndex,labelIndex);

//			std::cout<<currentIntIndex<<" "<<currentImageIndex<<" "<<label<<" "<<movingIndex<<std::endl;
//			std::cout<<labelIndex<<" "<<label<<std::endl;
//			std::cout<<this->m_unaryPotentialFunction->getPotential(currentImageIndex,label)<<std::endl;

			transformedImage->SetPixel(currentImageIndex,img->GetPixel(movingIndex));//(label[1]+15)*65535/15);//img->GetPixel(movingIndex));
			grid->next();
		}
		return transformedImage;
	}
};

#endif /* FAST_PD_REGISTRATION_MRF_H_ */
