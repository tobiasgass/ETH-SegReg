#include "pairtree.h"

PairTree::PairTree(const HyperParameters &hp) : m_hp ( hp )
{
    m_rootNode = PairNode::Ptr(new PairNode(hp,0,0));
}

PairTree::~PairTree()
{
}


void PairTree::train(const std::vector<Pair>& pairs)
{
    m_rootNode->train(pairs);
}


std::vector<int> PairTree::getPath(const std::vector<float>& x)
{
    std::vector<int> path;
    m_rootNode->getPath(x,path);
    return path;
}


int PairTree::getDepth()
{
    int treeDepth = m_rootNode->getDepth(0);
    return ++treeDepth;
}



