#include "Log.h"
//////////////////////////////////////////////////
//// SRS-MRF
//// Tobias Gass
//// ETH Zurich
//// No license yet!
/////////////////////////////////////////////////

#ifndef BASEMRF_H
#define BASEMRF_H

#include "Graph.h"

namespace SRS{

  /** \brief
   * Abstract class for MRF wrappers
   */
  template<class TGraphModel>
    class BaseMRFSolver{

  public:
    //	typedefs
    typedef TGraphModel GraphModelType;
    typedef typename GraphModelType::Pointer GraphModelPointerType;
  public:
    BaseMRFSolver(){};
   // virtual ~BaseMRFSolver(){};

    // the constructor only sets member variables
    BaseMRFSolver(GraphModelPointerType  graphModel,
		  double unaryRegWeight=1.0, 
		  double pairwiseRegWeight=1.0, 
		  double unarySegWeight=1.0, 
		  double pairwiseSegWeight=1.0, 
		  double pairwiseSegRegWeight=1.0,
          int vverbose=false){};
            ///pure virtual functions have to be implemented by derived classes

    ///create MRF graph, implementatiuon depends on the used optimisation library
        virtual void createGraph()=0;
    ///finalize initialization and call the optimisation
        virtual double optimize(int maxIter)=0;
    ///store MRF result in vectors of int(labels)
    virtual std::vector<int> getDeformationLabels()=0;
    virtual std::vector<int> getSegmentationLabels()=0;
    virtual double optimizeOneStep(int currentIter , bool & converged)=0;
    virtual void setPotentialCaching(bool b){} 

  };//MRFSolver
}//namespace



#endif //MRF_H
