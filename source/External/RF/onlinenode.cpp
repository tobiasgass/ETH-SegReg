#include "onlinenode.h"
#include "utilities.h"
#include <boost/foreach.hpp>

int OnlineNode::m_numNodes;

OnlineNode::OnlineNode(const HyperParameters &hp, int depth, int numFeatures) : m_hp(&hp), m_depth( depth ),
        m_numSamples( 0 ), m_nodeLabel( 0 ), m_numFeatures( numFeatures ),
        m_bestGini( 1e11 ), m_secondBestGini( 1e11 )
{
    m_isLeaf = true;
    m_nodeIndex = m_numNodes++;
    m_nodeConf = std::vector<float>(m_hp->numClasses,0.0);
    m_labelStatistics = std::vector<int>(m_hp->numClasses,0);
}

OnlineNode::OnlineNode(const HyperParameters &hp, int depth, int numFeatures, std::vector<int>& labelStatistics) :
        m_hp(&hp), m_depth( depth ), m_numSamples( 0 ), m_labelStatistics( labelStatistics ),
        m_numFeatures( numFeatures ),m_bestGini( 1e11 ), m_secondBestGini( 1e11 )
{
    int maxLabel = 0;
    int i = 0;
    BOOST_FOREACH(int label, m_labelStatistics) {
        if (label > maxLabel) {
            m_nodeLabel = i;
            maxLabel = label;
        }
        ++i;
    }

    m_isLeaf = true;
    m_nodeIndex = m_numNodes++;
    m_nodeConf.reserve(m_hp->numClasses);
}

OnlineNode::~OnlineNode() {

}

bool OnlineNode::shouldISplit() {
    // Test depth and sample size
    double epsilon = sqrt( pow(log(m_hp->numClasses),2.0) * log(100.0) / (2.0 * double(m_numSamples)) );
    double deltaGini = abs(m_bestGini - m_secondBestGini);
    //m_epsilons.push_back(epsilon);
    //m_deltaGinis.push_back(deltaGini);
    if (m_depth >= m_hp->maxTreeDepth || m_numSamples < m_hp->counterThreshold || deltaGini > epsilon) {
        return false;
    }

    // Test pureness of the node
    bool isPure = false;
    BOOST_FOREACH(int s, m_labelStatistics) {
        if (s == m_numSamples) {
            isPure = true;
            break;
        }
    }
    if (isPure) {
        return false;
    } else {
        return true;
    }

    return true;
}

void OnlineNode::getLabels(std::set<int>& nodeLabels) {
    if (!m_isLeaf) {
        m_leftChildNode->getLabels(nodeLabels);
        m_rightChildNode->getLabels(nodeLabels);
    } else {
        nodeLabels.insert(m_nodeLabel);
    }

    return;
}

