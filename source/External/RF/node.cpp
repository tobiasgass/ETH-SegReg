#include "node.h"
#include "utilities.h"
#include <boost/foreach.hpp>

int Node::m_numNodes;

Node::Node(const HyperParameters &hp, int depth) : m_hp(hp), m_depth( depth )
{
    m_isLeaf = false;
    m_nodeIndex = m_numNodes++;
    m_nodeConf.reserve(m_hp.numClasses);
}

Node::Node(const HyperParameters &hp, int depth, int reset) : m_hp(hp), m_depth( depth )
{
    m_isLeaf = false;
    m_numNodes = (reset >= 0) ? 0 : m_numNodes;
    m_nodeIndex = m_numNodes++;
    m_nodeConf.reserve(m_hp.numClasses);
}

Node::~Node()
{
}



xmlNodePtr Node::saveConfidence(const int idx, const float conf) const
{
    xmlNodePtr node = xmlNewNode( NULL, reinterpret_cast<const xmlChar*>( "confidence" ) );
    addIntProp(node, "class", idx);
    addDoubleProp(node, "conf", static_cast<double>(conf));
    return node;
}

bool Node::shouldISplit(const std::vector<int>& labels, std::vector<int>& inBagSamples)
{
    // Test depth and sample size
    if (m_depth >= m_hp.maxTreeDepth || inBagSamples.size() < 5)// || inBagSamples[0] > m_hp.numLabeled)
    {
        return false;
    }

    // Test pureness of the node
    int startLabel = labels[inBagSamples[0]];
    BOOST_FOREACH(int n, inBagSamples)
    {
        if (labels[n] != startLabel)
        {
            return true;
        }
    }

    return false;
}

bool Node::shouldISplitLU(const std::vector<int>& labels, std::vector<int>& inBagSamples)
{
    // Test depth and sample size
    if (m_depth >= m_hp.maxTreeDepth || inBagSamples.size() < 5)
    {
        return false;
    }
    else
    {
        return true;
    }
}

int Node::getDepth(const int oldDepth)
{
    if (m_isLeaf)
    {
        return (m_depth > oldDepth) ? m_depth : oldDepth;
    }
    else
    {
        int leftDepth = m_leftChildNode->getDepth(oldDepth);
        int rightDepth = m_rightChildNode->getDepth(oldDepth);

        if (leftDepth > oldDepth || rightDepth > oldDepth)
        {
            return (leftDepth > rightDepth) ? leftDepth : rightDepth;
        }
        else
        {
            return oldDepth;
        }
    }
}

void Node::getLabels(std::set<int>& nodeLabels)
{
    if (!m_isLeaf)
    {
        m_leftChildNode->getLabels(nodeLabels);
        m_rightChildNode->getLabels(nodeLabels);
    }
    else
    {
        nodeLabels.insert(m_nodeLabel);
    }

    return;
}

