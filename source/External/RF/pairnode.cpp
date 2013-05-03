#include "pairnode.h"
#include <boost/foreach.hpp>

using namespace std;
int PairNode::m_numNodes;

PairNode::PairNode(const HyperParameters &hp, int depth) : m_hp( hp ), m_depth( depth ),
            m_bestFeature( 0 ), m_bestThreshold( 0.0 )
{
    m_isLeaf = false;
    m_nodeIndex = m_numNodes++;
}

PairNode::PairNode(const HyperParameters &hp, int depth, int reset) : m_hp(hp), m_depth( depth )
{
    m_isLeaf = false;
    m_numNodes = (reset >= 0) ? 0 : m_numNodes;
    m_nodeIndex = m_numNodes++;
}

PairNode::~PairNode()
{
}

void PairNode::train(const std::vector<Pair>& pairs)
{
    if ( shouldISplit(pairs) ) {
        int numFeatures = pairs[0].x1().size();
        std::vector<int> randFeatures = randPerm(numFeatures, m_hp.numRandomFeatures);
        findHypotheses(pairs,randFeatures);
        // split the data
        std::vector<Pair> leftPairs, rightPairs;
        evalNode(pairs,leftPairs, rightPairs);

        // pass them to the left and right child, respectively
        m_leftChildNode = Ptr(new PairNode(m_hp,m_depth + 1));
        m_rightChildNode = Ptr(new PairNode(m_hp,m_depth + 1));

        m_leftChildNode->train(leftPairs);
        m_rightChildNode->train(rightPairs);
    }
    else {
        m_isLeaf = true;
    }
}

void PairNode::evalNode(const std::vector<Pair>& pairs,std::vector<Pair>& leftPairs,std::vector<Pair>& rightPairs)
{
    // What shall we do with positives that are separated ?
    // -> for now, we simply discard them
    std::vector<Pair>::const_iterator it( pairs.begin() );
    std::vector<Pair>::const_iterator end( pairs.end() );
    while( it != end ) {
        float v1 = it->x1()[m_bestFeature];
        float v2 = it->x2()[m_bestFeature];

        if (v1 > m_bestThreshold && v2 > m_bestThreshold) {
            rightPairs.push_back( *it );
        }
        if (v1 < m_bestThreshold && v2 < m_bestThreshold) {
            leftPairs.push_back( *it );
        }
        ++it;
    }
}

void PairNode::findHypotheses(const std::vector<Pair>& pairs, const std::vector<int> randFeatures)
{
    std::vector<int>::const_iterator featureIt(randFeatures.begin());
    std::vector<int>::const_iterator featureEnd(randFeatures.end());

    double bestError = 1.0, bestPosError = 1.0, bestNegError = 1.0, bestThreshold = 0.0;
    int numPairs = pairs.size();
    int numPosPairs = 0, numNegPairs = 0;
    int bestFeature = *featureIt;
    float curError = 1.0, curPosError = 1.0, curNegError = 1.0;
    float curThreshold = 0.0;
    int NUM_THRESHOLDS = 10;
    std::vector<Pair>::const_iterator pairIt;
    std::vector<Pair>::const_iterator pairEnd( pairs.end() );

    while ( featureIt != featureEnd )
    {
        for(int t = 0; t < NUM_THRESHOLDS; t++) {
            curThreshold = _rand();
            int numWrongPos = 0, numWrongNeg = 0;
            pairIt = pairs.begin();
            while( pairIt != pairEnd ) {
                std::vector<float> x1 = pairIt->x1();
                std::vector<float> x2 = pairIt->x2();
                float v1 = x1.at( *featureIt );
                float v2 = x2.at ( *featureIt );

                if( pairIt->label() == 1 ) {
                    if ( (v1 < curThreshold && v2 >= curThreshold) || (v1 >= curThreshold && v2 < curThreshold) ) {
                        numWrongPos++;
                    }
                    numPosPairs++;
                }
                else {
                    if ( (v1 < curThreshold && v2 < curThreshold) || (v1 >= curThreshold && v2 >= curThreshold) ) {
                        numWrongNeg++;
                    }
                    numNegPairs++;
                }
                ++pairIt;
            }
            curError = (double)(numWrongPos+numWrongNeg)/(double)numPairs;
            curPosError = (double)numWrongPos/(double)numPosPairs;
            curNegError = (double)numWrongNeg/(double)numNegPairs;

            //if (curPosError < bestPosError && curNegError <= bestNegError)
            if (curPosError < bestPosError)
            {
                bestPosError = curPosError;
                bestNegError = curNegError;
                bestError = curError;
                bestFeature = *featureIt;
                bestThreshold = curThreshold;
            }
            else if (curPosError == bestPosError && curNegError < bestNegError)
            {
                bestPosError = curPosError;
                bestNegError = curNegError;
                bestError = curError;
                bestFeature = *featureIt;
                bestThreshold = curThreshold;
            }

        }
        ++featureIt;
    }

    //cout << "bestError: " << bestError << " bestThr: " << bestThreshold << endl; exit(0);

    m_bestFeature = bestFeature;
    m_bestThreshold = (float) bestThreshold;
}

void PairNode::getPath(const std::vector<float>& x, std::vector<int>& path)
{
    path.push_back(m_nodeIndex);

    if (!m_isLeaf)
    {
        float val = x[m_bestFeature];
        if (val >= m_bestThreshold) {
            m_rightChildNode->getPath(x,path);
        }
        else {
            m_leftChildNode->getPath(x,path);
        }
    }
}

bool PairNode::shouldISplit(const std::vector<Pair>& pairs)
{
    // Test depth and sample size
    if (m_depth >= m_hp.maxTreeDepth || pairs.size() < 3)// || inBagSamples[0] > m_hp.numLabeled)
    {
        return false;
    }

    // Test pureness of the node
    std::vector<Pair>::const_iterator pairIt( pairs.begin() );
    std::vector<Pair>::const_iterator pairEnd( pairs.end() );
    int startLabel = pairIt->label();
    while( pairIt != pairEnd)
    {
        if (pairIt->label() != startLabel)
        {
            return true;
        }
        pairIt++;
    }

    return false;
}

int PairNode::getDepth(const int oldDepth)
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

std::vector<int> PairNode::getLeafNodeIds()
{
    std::vector<int> leafIds;
    if(m_isLeaf) {
        leafIds.push_back(m_nodeIndex);
        return leafIds;
    }
    else {
        std::vector<int> rightIds;
        std::vector<int> leftIds;
        rightIds = m_rightChildNode->getLeafNodeIds();
        leftIds = m_leftChildNode->getLeafNodeIds();
        leafIds.insert(leafIds.end(),rightIds.begin(),rightIds.end());
        leafIds.insert(leafIds.end(),leftIds.begin(),leftIds.end());
        return leafIds;
    }
}
