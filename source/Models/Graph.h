//////////////////////////////////////////////////
//// SRS-MRF
//// Tobias Gass
//// ETH Zurich
//// No license yet!
/////////////////////////////////////////////////

#ifndef GRAPH_H
#define GRAPH_H

#include "Node.h"
#include "Edge.h"

//do we need to typedef/template over nodes and edges here??
class mGraph{

	/** Add a Node to the Graph.*/
    virtual void addNode(Node n)=0;
    /** Add an edge to the Graph.*/
    virtual void addEdge(Edge E)=0;
    
};




#endif //GRAPH_H
