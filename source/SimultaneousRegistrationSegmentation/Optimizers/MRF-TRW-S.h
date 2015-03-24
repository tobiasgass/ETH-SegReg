/*
 * MRF-TRW-S.h
 *
 *  Created on: Dec 3, 2010
 *      Author: gasst
 */

#ifndef TRW_S_SRS_H_
#define TRW_S_SRS_H_
#include "Log.h"
#include "BaseMRF.h"

//#include "typeGeneral.h"
#include "MRFEnergy.h"
//#include "minimize.cpp"
//#include "treeProbabilities.cpp"
#include <vector>
#include <limits.h>


namespace SRS{
/** \brief
   * Wrapper for the TRW-S MRF solver package
   */

  template<class TGraphModel>
    class TRWS_SRSMRFSolver : public BaseMRFSolver<TGraphModel> {
  public:


    typedef TGraphModel GraphModelType;
    typedef typename GraphModelType::Pointer GraphModelPointerType;

    typedef TypeGeneral TRWType;
    typedef MRFEnergy<TRWType> MRFType;
    typedef TRWType::REAL Real;
    typedef typename MRFType::NodeId NodeType;


  protected:
    MRFType m_optimizer;
    double m_unarySegmentationWeight,m_pairwiseSegmentationWeight;
    double m_unaryRegistrationWeight,m_pairwiseRegistrationWeight;
    double m_pairwiseSegmentationRegistrationWeight;
    int verbose;
    GraphModelPointerType m_GraphModel;
    int nNodes, nRegNodes, nSegNodes, nEdges;
    std::vector<NodeType> segNodes;
    std::vector<NodeType> regNodes;
    clock_t m_start;
    int nRegLabels;
    int nSegLabels;
    bool m_segment, m_register,m_coherence;
    double m_lastLowerBound;
    std::vector<int> m_labelOrder;
    
  public:
  TRWS_SRSMRFSolver(GraphModelPointerType  graphModel,
		    double unaryRegWeight=1.0, 
		    double pairwiseRegWeight=1.0, 
		    double unarySegWeight=1.0, 
		    double pairwiseSegWeight=1.0, 
		    double pairwiseSegRegWeight=1.0,
		    int vverbose=false)
    :m_optimizer(TRWType::GlobalSize()),m_GraphModel(graphModel)
    {
      verbose=vverbose;
      m_unarySegmentationWeight=unarySegWeight;
      m_pairwiseSegmentationWeight=pairwiseSegWeight;
      m_unaryRegistrationWeight=unaryRegWeight;
      m_pairwiseRegistrationWeight=pairwiseRegWeight;
      m_pairwiseSegmentationRegistrationWeight=pairwiseSegRegWeight;
      m_labelOrder=std::vector<int>(this->m_GraphModel->nRegLabels());
      m_labelOrder[0]=(this->m_GraphModel->nRegLabels())/2;
      for (int l=0;l<(this->m_GraphModel->nRegLabels());++l){
	if (l < m_labelOrder[0])
	  m_labelOrder[l+1]=l;
	else if (l>m_labelOrder[0])
	  m_labelOrder[l]=l;
      }
    }
    
  TRWS_SRSMRFSolver()  :m_optimizer(TRWType::GlobalSize()){}
    ~TRWS_SRSMRFSolver()
      {
      }


    /// create optimizer object, and fill it with the information from the graphModel
    virtual void createGraph(){
      clock_t start = clock();
      {
	m_segment=false; 
	m_register=false;
	m_segment=0;
	m_register=0;
      }
      LOGV(1)<<"starting graph init"<<std::endl;
      m_optimizer=MRFType(TRWType::GlobalSize());
      this->m_GraphModel->Init();
      clock_t endUnary = clock();
      double t1 = (float) ((double)(endUnary - start) / CLOCKS_PER_SEC);
      tUnary+=t1;       

      nNodes=this->m_GraphModel->nNodes();
      nEdges=this->m_GraphModel->nEdges();
      nRegNodes=this->m_GraphModel->nRegNodes();
      nSegNodes=this->m_GraphModel->nSegNodes();

      segNodes = std::vector<NodeType>(nSegNodes,NULL);
      regNodes = std::vector<NodeType>(nRegNodes,NULL);
   
      m_start=start;

      int edgeCount=0;
      nRegLabels=this->m_GraphModel->nRegLabels();
      nSegLabels=this->m_GraphModel->nSegLabels();
      m_register=((m_pairwiseSegmentationRegistrationWeight>0 || m_unaryRegistrationWeight>0 || m_pairwiseRegistrationWeight>0) && nRegLabels>1);
      m_segment=((m_pairwiseSegmentationRegistrationWeight>0 || m_unarySegmentationWeight>0 || m_pairwiseSegmentationWeight)  && nSegLabels>1);
      m_coherence=m_pairwiseSegmentationRegistrationWeight>0;
      LOGV(6)<<VAR(m_register)<<" "<<VAR(m_segment)<<" "<<VAR(m_coherence)<<std::endl;
      logSetStage("Potential functions caching");
      //		traverse grid
      if (m_register){
	//RegUnaries
	clock_t startUnary = clock();

	TRWType::REAL D1[nRegLabels];
	//
	for (int l1=0;l1<nRegLabels;++l1) D1[l1]=0;
	//firstly allocate registration nodes with zero potentials
	for (int d=0;d<nRegNodes;++d){
	  regNodes[d] = 
	    m_optimizer.AddNode(TRWType::LocalSize(nRegLabels), TRWType::NodeData(D1));
	}
	//now compute&set all potentials
	for (int l1=0;l1<nRegLabels;++l1)
	  {
	    int regLabel=m_labelOrder[l1];
	    this->m_GraphModel->cacheRegistrationPotentials(regLabel);
	    for (int d=0;d<nRegNodes;++d){
	      double pot=this->m_GraphModel->getUnaryRegistrationPotential(d,regLabel);
	      pot*=m_unaryRegistrationWeight;
	      //in case of coherence weight, but no direct segmentation optimization, add coherence potential to registration unaries
	      if (m_coherence && !m_segment){
		//pretty inefficient as the reg neighbors are recomputed #registrationLabels times for each registration node.
		std::vector<int> regSegNeighbors=this->m_GraphModel->getRegSegNeighbors(d);
		int nNeighbours=regSegNeighbors.size();
		if (nNeighbours==0) {LOG<<"ERROR: node "<<d<<" seems to have no neighbors."<<std::endl;}
		for (int i=0;i<nNeighbours;++i){
		  double coherencePot=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(d,regSegNeighbors[i],regLabel,0);
		  //LOGV(10)<<VAR(d)<<" "<<VAR(regLabel)<<" "<<VAR(pot)<<" "<<VAR(coherencePot)<<std::endl;
		  pot+=coherencePot;
                                
		}
                          
	      }
	      m_optimizer.SetNodeDataPos(regNodes[d],regLabel,pot);
                        
	    }
	  }
            
	TRWType::REAL Vreg[nRegLabels*nRegLabels];
	for (int l1=0;l1<nRegLabels;++l1){
	  for (int l2=0;l2<nRegLabels;++l2){
	    Vreg[l1*nRegLabels+l2]=0;
	  }
	}
	clock_t endUnary = clock();
	double t = (float) ((double)(endUnary - startUnary) / CLOCKS_PER_SEC);
	LOGV(1)<<"Registration Unaries took "<<t<<" seconds."<<std::endl;
	tUnary+=t;
	/// Pairwise potentials
	/// pure Registration
	for (int d=0;d<nRegNodes;++d){
	  ///iterate over node indices (of the registration graph)
	  {
	    /// get neighbours of each node
	    std::vector<int> neighbours= this->m_GraphModel->getForwardRegistrationNeighbours(d);
	    int nNeighbours=neighbours.size();
	    /// iterate over neighbours
	    for (int i=0;i<nNeighbours;++i){
	      //iterate over all registration label combinations
	      for (int l1=0;l1<nRegLabels;++l1){
		for (int l2=0;l2<nRegLabels;++l2){
		  if (m_pairwiseRegistrationWeight>0){
		    /// get potential and store in array
		    Vreg[l1+l2*nRegLabels]=m_pairwiseRegistrationWeight*this->m_GraphModel->getPairwiseRegistrationPotential(d,neighbours[i],l1,l2);
		  }
		  else{
		    Vreg[l1*nRegLabels+l2]=0;
		  }
		}
	      }
	      /// add edge with stored potentials to external optimizer object
	      m_optimizer.AddEdge(regNodes[d], regNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,Vreg));
	      edgeCount++;
	    }
                
	  }
	}
	clock_t endPairwise = clock();
         
	t = (float) ((double)(endPairwise-endUnary ) / CLOCKS_PER_SEC);
	LOGV(1)<<"Registration pairwise took "<<t<<" seconds."<<std::endl;

	tPairwise+=t;
      }
      if (m_segment){
	//SegUnaries
	clock_t startUnary = clock();
	TRWType::REAL D2[nSegLabels];

	for (int d=0;d<nSegNodes;++d){
	  std::vector<int> segRegNeighbors=this->m_GraphModel->getSegRegNeighbors(d);
	  for (int l1=0;l1<nSegLabels;++l1)
	    {
	      
	      D2[l1]=m_unarySegmentationWeight*this->m_GraphModel->getUnarySegmentationPotential(d,l1);
	      //in case of coherence weight, but no direct registration optimization, add coherence potential to registration unaries
	      if (m_coherence && !m_register){
		for (int i=0;i<segRegNeighbors.size();++i){
		  D2[l1]+=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(segRegNeighbors[i],d,0,l1);
		}
	      }
	    }
	  segNodes[d] = 
	    m_optimizer.AddNode(TRWType::LocalSize(nSegLabels), TRWType::NodeData(D2));
                
	  //  LOG<<" reg and segreg pairwise pots" <<std::endl;
       
	}
	clock_t endUnary = clock();
	double t = (float) ((double)(endUnary - startUnary) / CLOCKS_PER_SEC);
	LOGV(1)<<"Segmentation Unaries took "<<t<<" seconds."<<std::endl;
	LOGV(1)<<"Approximate size of seg unaries: "<<1.0/(1024*1024)*nSegNodes*nSegLabels*sizeof(double)<<" mb."<<std::endl;

	TRWType::REAL VsrsBack[nRegLabels*nSegLabels];
	int nSegEdges=0,nSegRegEdges=0;
	for (int d=0;d<nSegNodes;++d){   
	  TRWType::REAL Vseg[nSegLabels*nSegLabels];
	  //pure Segmentation
	  std::vector<int> neighbours= this->m_GraphModel->getForwardSegmentationNeighbours(d);
	  int nNeighbours=neighbours.size();
	  for (int i=0;i<nNeighbours;++i){
	    nSegEdges++;
	    for (int l1=0;l1<nSegLabels;++l1){
	      for (int l2=0;l2<nSegLabels;++l2){
		double lambda =m_pairwiseSegmentationWeight*this->m_GraphModel->getPairwiseSegmentationPotential(d,neighbours[i],l1,l2);
		Vseg[l1+nSegLabels*l2]=lambda;
	      }
	    }
	    m_optimizer.AddEdge(segNodes[d], segNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,Vseg));
	    edgeCount++;
                    
	  }
	  if (m_register && m_coherence){
	    std::vector<int> segRegNeighbors=this->m_GraphModel->getSegRegNeighbors(d);
	    nNeighbours=segRegNeighbors.size();
	    if (nNeighbours==0) {LOG<<"ERROR: node "<<d<<" seems to have no neighbors."<<std::endl;}
	    for (int i=0;i<nNeighbours;++i){
	      for (int l1=0;l1<nRegLabels;++l1){
		for (int l2=0;l2<nSegLabels;++l2){
		  //forward
		  VsrsBack[l1+l2*nRegLabels]=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(segRegNeighbors[i],d,l1,l2);

		}
	      }
	      m_optimizer.AddEdge(regNodes[segRegNeighbors[i]], segNodes[d], TRWType::EdgeData(TRWType::GENERAL,VsrsBack));
                  
	      edgeCount++;
	      nSegRegEdges++;
	    }
	  }
                
	}
	clock_t endPairwise = clock();
	t = (float) ((double)(endPairwise-endUnary ) / CLOCKS_PER_SEC);
	LOGV(1)<<"Segmentation + SRS pairwise took "<<t<<" seconds."<<std::endl;
	LOGV(1)<<"Approximate size of seg pairwise: "<<1.0/(1024*1024)*nSegEdges*nSegLabels*nSegLabels*sizeof(double)<<" mb."<<std::endl;
	LOGV(1)<<"Approximate size of SRS pairwise: "<<1.0/(1024*1024)*nSegRegEdges*nSegLabels*nRegLabels*sizeof(double)<<" mb."<<std::endl;
            
      }
      clock_t finish = clock();
      double t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
      //tInterpolation+=t;
      LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;
      nEdges=edgeCount;
      logResetStage;
    }
    

    virtual double optimize(int maxIter=20){
      LOGV(5)<<"Total number of MRF edges: " <<nEdges<<std::endl;
      //m_optimizer.SetAutomaticOrdering();

      MRFEnergy<TRWType>::Options options;
      TRWType::REAL energy=-1, lowerBound=-1;
      options.m_iterMax = maxIter; // maximum number of iterations
      options.m_printMinIter=1;
      options.m_printIter=1;
      //options.verbose=verbose;
      options.m_eps=1e-6;
      logSetStage("TRWOptimizer");
      clock_t opt_start=clock();
      m_optimizer.Minimize_TRW_S(options, lowerBound, energy);
      clock_t finish = clock();
      tOpt+=((double)(finish-opt_start)/CLOCKS_PER_SEC);
      float t = (float) ((double)(finish - m_start) / CLOCKS_PER_SEC);
      LOG<<"Finished optimization after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound <<std::endl;
      logResetStage;         
      return energy;

    }
    virtual double optimizeOneStep(int currentIter , bool & converged){
      LOGV(1)<<"Total number of MRF edges: " <<nEdges<<std::endl;
      //m_optimizer.SetAutomaticOrdering();
      MRFEnergy<TRWType>::Options options;
      TRWType::REAL energy=-1, lowerBound=-1;
      options.m_iterMax = 1; // maximum number of iterations
      options.m_printMinIter=0;
      options.m_printIter=0;
      //options.verbose=0;
      options.m_eps=1e-6;
      logSetStage("Optimizer");
      clock_t opt_start=clock();
      m_optimizer.Minimize_TRW_S(options, lowerBound, energy);
      clock_t finish = clock();
      logResetStage;         
      tOpt+=((double)(finish-opt_start)/CLOCKS_PER_SEC);
      float t = (float) ((double)(finish -  opt_start) / CLOCKS_PER_SEC);
      LOG<<VAR(currentIter)<<" Finished optimization after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound <<std::endl;
      converged=(energy==lowerBound);
      if (currentIter>0){
	converged= (converged || (fabs(lowerBound-m_lastLowerBound) < 1e-6 * m_lastLowerBound ));
      }
      if ( 0.0 < (m_lastLowerBound - lowerBound) )  {
	LOGV(2)<<"something might be strange, "<<VAR(m_lastLowerBound)<<" greater than " << VAR(lowerBound)<< " " <<VAR(m_lastLowerBound - lowerBound )<<std::endl;
      }
      m_lastLowerBound=lowerBound;
      return energy;

    }
    virtual std::vector<int> getDeformationLabels(){
      std::vector<int> labels(nRegNodes,0);
      if (m_register){
	for (int i=0;i<nRegNodes;++i){
	  labels[i]=m_optimizer.GetSolution(regNodes[i]);
	}
      }
      return labels;
    }
    virtual std::vector<int> getSegmentationLabels(){
      std::vector<int> labels(nSegNodes,0);
      if (m_segment){
	for (int i=0;i<nSegNodes;++i){
	  labels[i]=m_optimizer.GetSolution(segNodes[i]);
	  //labels[i]=this->m_GraphModel->getUnarySegmentationPotential(i,1)*nSegNodes;//m_optimizer.GetSolution(segNodes[i]);
	}
      }
      return labels;
    }

    void evalSolution(){
      double sumUReg=0,sumUSeg=0,sumPSeg=0,sumPSegReg=0;
        
      clock_t start = clock();
      m_start=start;
      if (nRegLabels){
	for (int d=0;d<nRegNodes;++d){
	  sumUReg+=m_unaryRegistrationWeight*this->m_GraphModel->getUnaryRegistrationPotential(d,m_optimizer.GetSolution(regNodes[d]));
	}
      }
      if (nSegLabels){
	for (int d=0;d<nSegNodes;++d){
	  sumUSeg+=m_unarySegmentationWeight*this->m_GraphModel->getUnarySegmentationPotential(d,m_optimizer.GetSolution(segNodes[d]));
	  if (nRegLabels){
	    std::vector<int> neighbours= this->m_GraphModel->getForwardSegmentationNeighbours(d);
	    int nNeighbours=neighbours.size();
	    for (int i=0;i<nNeighbours;++i){
	      sumPSeg+=m_pairwiseSegmentationWeight*this->m_GraphModel->getSegmentationWeight(d,neighbours[i])*(m_optimizer.GetSolution(segNodes[d])!=m_optimizer.GetSolution(segNodes[neighbours[i]]));
	    }
	    std::vector<int> segRegNeighbors=this->m_GraphModel->getSegRegNeighbor(d);
	    for (unsigned int n=0;n<segRegNeighbors.size();++n){
	      sumPSegReg+=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(segRegNeighbors[n],d,m_optimizer.GetSolution(regNodes[segRegNeighbors[n]]),m_optimizer.GetSolution(segNodes[d]));
	    }
	  }


	}   
      }
      LOG<<"RegU :\t\t"<<sumUReg<<std::endl
	 <<"SegU :\t\t"<<sumUSeg<<std::endl
	 <<"SegP :\t\t"<<sumPSeg<<std::endl
	 <<"SegRegP :\t"<<sumPSegReg<<std::endl;
      clock_t finish = clock();
      double t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
      LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;
    }

  };
}
#endif /* TRW_S_REGISTRATION_H_ */
