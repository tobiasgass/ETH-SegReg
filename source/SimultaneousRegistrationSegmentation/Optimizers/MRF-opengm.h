#pragma once
#include "BaseMRF.h"
#include "Log.h"

#include <vector>

#ifdef WITH_TCMALLOC
#include <google/heap-profiler.h>
#endif

#include <limits.h>
#include <time.h>
#include <map>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/sparsemarray.hxx>
#include <opengm/inference/alphaexpansion.hxx>
//#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#include <opengm/inference/auxiliary/minstcutboost.hxx>
//#include <opengm/inference/external/trws.hxx>


namespace SRS{
/** \brief
   * Wrapper for the openGM MRF solver package
   * 
   * 
   */
template<class TGraphModel>
class OPENGM_SRSMRFSolver :public BaseMRFSolver<TGraphModel>{
public:

    //typedef short EnergyType;
    typedef float EnergyType;

    typedef OPENGM_SRSMRFSolver<TGraphModel> Self;
    typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::Pointer GraphModelPointerType;

    static const int D = GraphModelType::ImageType::ImageDimension;

    typedef typename   opengm::DiscreteSpace<size_t, size_t> OpenGMLabelSpace;
    typedef typename opengm::ExplicitFunction<EnergyType, size_t, size_t> FunctionType;
    typedef typename opengm::GraphicalModel<EnergyType, 
                                            typename opengm::Adder,
                                            FunctionType,
                                            OpenGMLabelSpace
                                            > GraphicalModelType;
    typedef typename opengm::MinSTCutBoost<size_t, EnergyType,opengm::KOLMOGOROV> MinStCutType;
    //typedef typename opengm::external::MinSTCutKolmogorov<size_t,EnergyType> MinStCutType;

    typedef typename opengm::GraphCut<GraphicalModelType, typename opengm::Minimizer, MinStCutType> MinGraphCut;
    typedef typename opengm::AlphaExpansion<GraphicalModelType, MinGraphCut> MinAlphaExpansion;
    //typedef typename opengm::external::TRWS<GraphicalModelType> TRWS;
    

protected:
    GraphicalModelType * m_gm;
    double m_unaryRegistrationWeight,m_unarySegmentationWeight;
    double m_pairwiseSegmentationRegistrationWeight,m_pairwiseRegistrationWeight,m_pairwiseSegmentationWeight;
	int verbose;
    
    int nNodes, nRegNodes, nSegNodes, nEdges;
    clock_t m_start;
    int nRegLabels;
    int nSegLabels;
    bool m_segment, m_register,m_coherence;
    double m_lastLowerBound;
    vector<int> m_labelOrder;
    int m_zeroDisplacementLabel;
    bool m_deleteRegNeighb;
    vector<size_t> m_solution;

   
    int GLOBALnRegNodes,GLOBALnSegNodes,GLOBALnSegLabels,GLOBALnRegLabels;

    GraphModelPointerType m_GraphModel;

public:
    OPENGM_SRSMRFSolver(GraphModelPointerType  graphModel,
                     double unaryRegWeight=1.0, 
                     double pairwiseRegWeight=1.0, 
                     double unarySegWeight=1.0, 
                     double pairwiseSegWeight=1.0, 
                     double pairwiseSegRegWeight=1.0,
                     int vverbose=false)
        
	{
        this->m_GraphModel=graphModel;
		verbose=vverbose;
		m_unarySegmentationWeight=unarySegWeight;
		m_pairwiseSegmentationWeight=pairwiseSegWeight;
        m_unaryRegistrationWeight=unaryRegWeight;
        m_pairwiseRegistrationWeight=pairwiseRegWeight;
        m_pairwiseSegmentationRegistrationWeight=pairwiseSegRegWeight;
        //createGraph();
        //init();
        m_gm=NULL;
        m_labelOrder=vector<int>(this->m_GraphModel->nRegLabels());
        //order registration labels such that they start with zero displacement
        m_zeroDisplacementLabel=this->m_GraphModel->getLabelMapper()->getZeroDisplacementIndex();
        m_labelOrder[0]=m_zeroDisplacementLabel;
        for (int l=0;l<(this->m_GraphModel->nRegLabels());++l){
            if (l < m_labelOrder[0])
                m_labelOrder[l+1]=l;
            else if (l>m_labelOrder[0])
                m_labelOrder[l]=l;
        }
        srand ( time(NULL) );
     
      
    }
    OPENGM_SRSMRFSolver()  {
        
    }
	~OPENGM_SRSMRFSolver()
    { 

        LOGV(1)<<"Deleting OPENGM_MRF Sovler " << endl;
     
        delete m_gm;
        
    }

    

    virtual void createGraph(){
        clock_t start = clock();
        {
            m_segment=false; 
            m_register=false;
            m_segment=0;
            m_register=0;
        }
        LOGV(1)<<"starting graph init"<<std::endl;
        this->m_GraphModel->Init();
        clock_t endUnary = clock();
        double t1 = (float) ((double)(endUnary - start) / CLOCKS_PER_SEC);
        tUnary+=t1;       

        nNodes=this->m_GraphModel->nNodes();
        nEdges=this->m_GraphModel->nEdges();
        nRegNodes=this->m_GraphModel->nRegNodes();
        nSegNodes=this->m_GraphModel->nSegNodes();

        m_start=start;

        int edgeCount=0;
        
        nRegLabels=this->m_GraphModel->nRegLabels();
        nSegLabels=this->m_GraphModel->nSegLabels();
        m_register=((m_pairwiseSegmentationRegistrationWeight>0 || m_unaryRegistrationWeight>0 || m_pairwiseRegistrationWeight>0) && nRegLabels>1);
        m_segment=((m_pairwiseSegmentationRegistrationWeight>0 || m_unarySegmentationWeight>0 || m_pairwiseSegmentationWeight)  && nSegLabels>1);
        m_coherence=m_pairwiseSegmentationRegistrationWeight>0;
        GLOBALnRegNodes= m_register*nRegNodes;
        GLOBALnSegNodes= m_segment*nSegNodes;
        GLOBALnRegLabels=m_register*nRegLabels;
        GLOBALnSegLabels=m_segment*nSegLabels;
 
        if (m_gm) delete m_gm;
        m_gm=new GraphicalModelType;
        OpenGMLabelSpace space; 
        typedef typename  GraphicalModelType::FunctionIdentifier FunctionIdentifier;
        std::vector<unsigned char> nLabels(GLOBALnRegNodes+GLOBALnSegNodes);
        if (m_register){
            //add registration variables
            for (int d=0;d<nRegNodes;++d){
                //space.addVariable(nRegLabels);
                nLabels[d]=nRegLabels;
            }
        }
        if (m_segment){
            //add segmentation variables
            for (int d=0;d<nSegNodes;++d){
                //space.addVariable(nSegLabels);
                nLabels[d+GLOBALnRegNodes]=nSegLabels;
            }
        }
        space=OpenGMLabelSpace(nLabels.begin(),nLabels.end());
        m_gm=new GraphicalModelType(space);

        float functionDefaultValue=0.0;
        
        if (m_register){
            //add factors
            const size_t shape[] = {nRegLabels};
            std::vector<  FunctionType > f(nRegNodes, FunctionType(shape, shape + 1));
  
            for (int l1=0;l1<nRegLabels;++l1){
                this->m_GraphModel->cacheRegistrationPotentials(l1);
                for (int d=0;d<nRegNodes;++d){
                    //unary factors
                    f[d](l1)=m_unaryRegistrationWeight*this->m_GraphModel->getUnaryRegistrationPotential(d,l1);
                }
            }
            for (int d=0;d<nRegNodes;++d){
                 FunctionIdentifier fid=m_gm->addFunction(f[d]);
                 size_t vi[]={d};
                 m_gm->addFactor(fid,vi,vi+1);
                 //pairwise factors
                 {
                     const size_t shape[] = {nRegLabels,nRegLabels};
                     FunctionType f(shape, shape + 2,functionDefaultValue);
                    std::vector<int> neighbours= this->m_GraphModel->getForwardRegistrationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                        //LOG<<d<<" "<<regNodes[d]<<" "<<i<<" "<<neighbours[i]<<std::endl;
                        
                        for (int l1=0;l1<nRegLabels;++l1){
                            for (int l2=0;l2<nRegLabels;++l2){
                                double cost=m_pairwiseRegistrationWeight*this->m_GraphModel->getPairwiseRegistrationPotential(d,neighbours[i],l1,l2);
                                if (cost !=functionDefaultValue)
                                    f(l1,l2)=cost;
                            }
                        }
                        FunctionIdentifier fid= m_gm->addFunction(f);
                        size_t vi[]={d,neighbours[i]};
                        m_gm->addFactor(fid,vi,vi+2);
                        
                    }
                }
            }

        }
        
        if (m_segment){
            //add factors
           
            for (int d=0;d<nSegNodes;++d){
                {
                    //unary factors
                    const size_t shape[] = {nSegLabels};
                    FunctionType f(shape, shape + 1);
                    for (int l1=0;l1<nSegLabels;++l1){
                        
                        f(l1)=m_unarySegmentationWeight*this->m_GraphModel->getUnarySegmentationPotential(d,l1);
                    }
                    FunctionIdentifier fid=m_gm->addFunction(f);
                    size_t vi[]={d+GLOBALnRegNodes};
                    m_gm->addFactor(fid,vi,vi+1);
                    
                }
                //pairwise factors
                {
                    const size_t shape[] = {nSegLabels,nSegLabels};
                    FunctionType f(shape, shape + 2, functionDefaultValue);
                    std::vector<int> neighbours= this->m_GraphModel->getForwardSegmentationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                      
                        for (int l1=0;l1<nSegLabels;++l1){
                            for (int l2=0;l2<nSegLabels;++l2){
                                double cost = m_pairwiseSegmentationWeight*this->m_GraphModel->getPairwiseSegmentationPotential(d,neighbours[i],l1,l2);
                                 if (cost !=functionDefaultValue)
                                    f(l1,l2)=cost;
                            }
                        }
                        FunctionIdentifier fid= m_gm->addFunction(f);
                        size_t vi[]={d+GLOBALnRegNodes,neighbours[i]+GLOBALnRegNodes};
                        m_gm->addFactor(fid,vi,vi+2);
                        
                    }
                }
            }

        }

        if (m_coherence){
            const size_t shape[] = {nSegLabels,nRegLabels};

            for (int d=0;d<nSegNodes;++d){
                std::vector<int> segRegNeighbors=this->m_GraphModel->getSegRegNeighbors(d);
                int nNeighbours=segRegNeighbors.size();
                for (int i=0;i<nNeighbours;++i){
                    FunctionType f(shape, shape + 2,functionDefaultValue);
                    for (int l1=0;l1<nSegLabels;++l1){
                        for (int l2=0;l2<nRegLabels;++l2){
                            double cost=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(segRegNeighbors[i],d,l2,l1);
                            if (cost !=functionDefaultValue)
                                f(l1,l2)=cost;
                        }
                    }
                    FunctionIdentifier fid= m_gm->addFunction(f);
                    size_t vi[]={segRegNeighbors[i],d+GLOBALnRegNodes};
                    m_gm->addFactor(fid,vi,vi+2);
                }

                

            }
            
        }



    }
    

    virtual double optimize(int maxIter=20){
        logSetStage("OPENGM-Optimizer");
#if 1
        MinAlphaExpansion solver(*m_gm);
#else
        typename TRWS::Parameter param;
        param.numberOfIterations_=maxIter;
        param.useZeroStart_=true;
        param.energyType_=TRWS::Parameter::TABLES;
        TRWS solver(*m_gm,param);
#endif

        clock_t opt_start=clock();
        double energy;//=m_optimizer->compute_energy();
        //LOGV(2)<<VAR(energy)<<endl;
        try{
            solver.infer();
            //m_optimizer->swap(maxIter);
        }catch (...){

        }
        energy=solver.value();
        clock_t finish = clock();
        tOpt+=((double)(finish-opt_start)/CLOCKS_PER_SEC);
        float t = (float) ((double)(finish - m_start) / CLOCKS_PER_SEC);
        LOG<<"Finished optimization after "<<t<<" , resulting energy is "<<energy<<std::endl;
       
        solver.arg(m_solution);
        logResetStage;         
        return energy;

    }
    virtual double optimizeOneStep(int currentIter , bool & converged){
        clock_t opt_start=clock();
       
        return -1;

    }
    virtual std::vector<int> getDeformationLabels(){
        std::vector<int> Labels(nRegNodes,0);
        if (m_register){
            for (int i=0;i<nRegNodes;++i){
                Labels[i]=m_solution[i];
                LOGV(20)<<"DEF "<<VAR(i)<<" "<<VAR(Labels[i])<<endl;
            }
        }
        return Labels;
    }
    virtual std::vector<int> getSegmentationLabels(){
        std::vector<int> Labels(nSegNodes,0);
        if (m_segment){
            for (int i=0;i<nSegNodes;++i){
                Labels[i]=m_solution[i+GLOBALnRegNodes]-GLOBALnRegLabels;
                LOGV(20)<<"SEG "<<VAR(i)<<" "<<VAR(Labels[i])<<endl;
            }
        }
        return Labels;
    }

    void evalSolution(){
        LOG<<"NYI"<<std::endl;
    }

  
};
}
