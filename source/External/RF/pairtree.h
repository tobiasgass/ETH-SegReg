#ifndef PAIR_TREE_H_
#define PAIR_TREE_H_

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include "hyperparameters.h"
#include <boost/numeric/ublas/matrix.hpp>
#include "utilities.h"
#include <libxml/tree.h>
#include <libxml/parser.h>
using namespace boost::numeric::ublas;
using std::cout;
using std::endl;

#include "data.h"
#include "pairnode.h"

class PairTree
{
public:
    PairTree(const HyperParameters &hp);
    ~PairTree();

    void train(const std::vector<Pair>& pairs);

    inline int eval(const std::vector<float>& x) const { return m_rootNode->eval( x ); };

    inline int numLeafs() const;

    std::vector<int> getPath(const std::vector<float>& x);

    int getDepth();

    inline std::vector<int> getLeafNodeIds() { return m_rootNode->getLeafNodeIds(); };
private:
    HyperParameters m_hp;
    PairNode::Ptr m_rootNode;
};

inline int PairTree::numLeafs() const
{
    return m_rootNode->numLeafs();
}
#endif /* PairTree_H_ */

