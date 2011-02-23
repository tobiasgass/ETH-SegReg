#include "onlinenodehyperplane.h"
#include "nodeinfogain.h"
#include "utilities.h"
#include <boost/foreach.hpp>

OnlineNodeHyperplane::OnlineNodeHyperplane(const HyperParameters &hp, int depth, int numFeatures,
        std::vector<int>& labelStatistics) : OnlineNode(hp, depth, numFeatures, labelStatistics), m_bestFeature( -1 ),
        m_bestThreshold( 0.0 ) {
    m_features.reserve(hp.numRandomFeatures);
    m_nodeConf.resize(m_hp->numClasses, 0.0);
    // select random features and thresholds
    for ( int feat = 0; feat < hp.numRandomFeatures; feat++) {
        Feature f;
        f.indeces = randPerm(numFeatures, m_hp->numProjFeatures );
        f.weights = std::vector<float>(m_hp->numProjFeatures, 0.0 );
        fillWithRandomNumbers(f.weights);
        f.bestStatistics = 0;
        f.bestGini = 1e10;
        // We assume having the same number of random thresholds as random features
        std::vector<float> thresholds(/*hp.numRandomFeatures*/ 1,0.0);
        fillWithRandomNumbers(thresholds);
        std::vector<float>::iterator thrIt(thresholds.begin());
        std::vector<float>::iterator thrEnd(thresholds.end());
        while (thrIt != thrEnd) {
            *thrIt = *thrIt; // * (float)hp.numRandomFeatures;
            ++thrIt;
        }
        BOOST_FOREACH(float thr,thresholds) {
            Statistics stat;
            stat.lTotal = 0;
            stat.rTotal = 0;
            stat.lCount.resize(m_hp->numLabeled,0);
            stat.rCount.resize(m_hp->numLabeled,0);
            stat.threshold = thr;
            f.statistics.push_back(stat);
        }
        m_features.push_back(f);
    }
}

OnlineNodeHyperplane::OnlineNodeHyperplane(const HyperParameters &hp, int depth, int numFeatures) :
        OnlineNode(hp, depth, numFeatures), m_bestFeature( -1 ),
        m_bestThreshold( 0.0 ) {
    m_features.reserve(hp.numRandomFeatures);
    m_nodeConf.resize(m_hp->numClasses, 0.0);
    // select random features and thresholds
    for ( int feat = 0; feat < hp.numRandomFeatures; feat++) {
        Feature f;
        f.indeces = randPerm(numFeatures, m_hp->numProjFeatures );
        f.weights = std::vector<float>(m_hp->numProjFeatures, 0.0 );
        fillWithRandomNumbers(f.weights);
        f.bestStatistics = 0;
        f.bestGini = 1e10;
        // We assume having the same number of random thresholds as random features
        std::vector<float> thresholds(1/*hp.numRandomFeatures*/,0.0);
        fillWithRandomNumbers(thresholds);
        std::vector<float>::iterator thrIt(thresholds.begin());
        std::vector<float>::iterator thrEnd(thresholds.end());
        while (thrIt != thrEnd) {
            *thrIt = *thrIt;// * (float)hp.numRandomFeatures;
            ++thrIt;
        }

        BOOST_FOREACH(float thr,thresholds) {
            Statistics stat;
            stat.lTotal = 0;
            stat.rTotal = 0;
            stat.lCount.resize(m_hp->numLabeled,0);
            stat.rCount.resize(m_hp->numLabeled,0);
            stat.threshold = thr;
            f.statistics.push_back(stat);
        }
        m_features.push_back(f);
    }
}


void OnlineNodeHyperplane::updateHypotheses(const std::vector<float>& sample, const int label) {
    // Go over each feature and each threshold and update the statistics
    m_numSamples++;
    double bestGini = 1e10;

    std::vector<Feature>::iterator featureIt(m_features.begin());
    std::vector<Feature>::iterator featureEnd(m_features.end());
    double tmpGini = 1e10;
    int currentFeature = 0;

    while ( featureIt != featureEnd ) {
        std::vector<Statistics>::iterator statIt(featureIt->statistics.begin());
        std::vector<Statistics>::iterator statEnd(featureIt->statistics.end());
        float val =  0.0; //sample[featureIt->index];
        int counter = 0;
        BOOST_FOREACH(int feat, featureIt->indeces) {
            val += sample[feat] * featureIt->weights[counter];
            counter++;
        }

        int currentStatistics = 0;

        while ( statIt != statEnd ) {
            if ( val < statIt->threshold ) {
                statIt->lTotal++;
                statIt->lCount[label]++;
            } else {
                statIt->rTotal++;
                statIt->rCount[label]++;
            }
            double lTotal = static_cast<double>(statIt->lTotal) + 1e-10;
            double rTotal = static_cast<double>(statIt->rTotal) + 1e-10;
            double lGini = 0.0, rGini = 0.0;
            for ( int c = 0; c < m_hp->numClasses; c++ ) {
                lGini += static_cast<double>(statIt->lCount[c])*(1.0 - static_cast<double>(statIt->lCount[c]/lTotal));
                rGini += static_cast<double>(statIt->rCount[c])*(1.0 - static_cast<double>(statIt->rCount[c]/rTotal));
            }
            tmpGini = (lGini + rGini)/(lTotal + rTotal);

            if (tmpGini < bestGini) {
                if (tmpGini < m_bestGini) {
                    m_secondBestGini = m_bestGini;
                    m_bestGini = tmpGini;
                }
                bestGini = tmpGini;
                featureIt->bestStatistics = currentStatistics;
                m_bestThreshold = statIt->threshold;
                m_bestFeature = currentFeature;
            }
            ++statIt;
            ++currentStatistics;
        }


        ++featureIt;
        ++currentFeature;
    }

}

void OnlineNodeHyperplane::evalNode(const matrix<float>& data, const std::vector<int>& inBagSamples,
                                    std::vector<int>& leftNodeSamples, std::vector<int>& rightNodeSamples) {
    BOOST_FOREACH(int n, inBagSamples) {
        if (data(n, m_bestFeature) > m_bestThreshold) {
            rightNodeSamples.push_back(n);
        } else {
            leftNodeSamples.push_back(n);
        }
    }
}

double OnlineNodeHyperplane::eval(const std::vector<float>& sample, const int label) {
    if (m_isLeaf) {
        return (double)m_nodeLabel;
    } else {
        float val =  0.0;
        int counter = 0;
        BOOST_FOREACH(int feat, m_bestIndeces) {
            val += sample[feat] * m_bestWeights[counter];
            counter++;
        }

        return (val < m_bestThreshold) ? m_leftChildNode->eval(sample,label) :
               m_rightChildNode->eval(sample,label);
    }
}

void OnlineNodeHyperplane::cleanUpAndFreeze() {
    // Do some cleaning here
    m_bestIndeces = m_features[m_bestFeature].indeces;
    m_bestWeights = m_features[m_bestFeature].weights;
    m_features.clear();
}

void OnlineNodeHyperplane::train(const std::vector<float>& sample, const int label) {
    if (m_isLeaf) {
        updateHypotheses(sample, label);

        bool doSplit = shouldISplit();

        if ( doSplit ) {
            /*
            cout << "Split occured! numSamples " << m_numSamples << endl;
            dispVector(m_epsilons);
            cout << endl;
            dispVector(m_deltaGinis);
            exit(0);
            */
            m_isLeaf = false;

            if (m_hp->verbose) {
                cout << "Node #: " << m_nodeIndex << " selected feature #: " << m_bestFeature;
                cout << " and the threshold is: " << m_bestThreshold << " at depth " << m_depth << endl;
            }

            // split the data

            // forward statistics to child nodes
            std::vector<int> lLabelStatistics(m_hp->numClasses,0), rLabelStatistics(m_hp->numClasses,0);
            Statistics bestStatistics = m_features[ m_bestFeature ].statistics[ m_features[m_bestFeature].bestStatistics ];
            std::vector<int>::const_iterator it(bestStatistics.lCount.begin());
            std::vector<int>::const_iterator end(bestStatistics.lCount.end());
            for (int c = 0; c < m_hp->numClasses && it != end; c++ ) {
                lLabelStatistics[c] = *it;
                ++it;
            }
            it = bestStatistics.rCount.begin();
            end = bestStatistics.rCount.end();
            for (int c = 0; c < m_hp->numClasses && it != end; c++ ) {
                rLabelStatistics[c] = *it;
                ++it;
            }

            // clean up the current node and freeze its state
            cleanUpAndFreeze();

            // pass them to the left and right child, respectively
            m_leftChildNode = Ptr(new OnlineNodeHyperplane(*m_hp,m_depth + 1,m_numFeatures,lLabelStatistics));
            m_rightChildNode = Ptr(new OnlineNodeHyperplane(*m_hp,m_depth + 1,m_numFeatures,rLabelStatistics));

            m_leftChildNode->train(sample,label);
            m_rightChildNode->train(sample,label);

        } else { // Update Statistics
            if (m_hp->verbose) {
                cout << "Node #: " << m_nodeIndex << " is terminal, at depth " << m_depth << endl;
            }

            // calc confidence, labels, etc
            m_isLeaf = true;

            m_labelStatistics[label]++;

            int bestClass = 0;
            float bestLabel = 0;
            std::vector<int>::iterator labelIt = m_labelStatistics.begin(), labelEnd = m_labelStatistics.end();

            int tmpN = 0;
            while ( labelIt != labelEnd) {
                if (*labelIt > bestLabel) {
                    bestLabel = *labelIt;
                    bestClass = tmpN;
                }
                ++labelIt;
                ++tmpN;
            }
            //cout << "NodeLabel: " << m_nodeLabel << " Real Label: " << label << " Best Class: " << bestClass << endl;
            m_nodeLabel = bestClass;
        }
    } else {
        float val =  0.0;
        int counter = 0;
        BOOST_FOREACH(int feat, m_bestIndeces) {
            val += sample[feat] * m_bestWeights[counter];
            counter++;
        }
        if (val < m_bestThreshold) {
            m_leftChildNode->train(sample,label);
        } else {
            m_rightChildNode->train(sample,label);
        }
    }
}


void OnlineNodeHyperplane::eval(const matrix<float>& data, const std::vector<int>& sampleIndeces,
                                matrix<float>& confidences, std::vector<int>& predictions) {
    if (m_isLeaf) {
        // Make predictions and confidences
        int tmpN;
        BOOST_FOREACH( int n, sampleIndeces) {
            predictions[n] = m_nodeLabel;
            tmpN = 0;
            BOOST_FOREACH(float conf, m_nodeConf) {
                confidences(n, tmpN) = conf;
                tmpN++;
            }
        }
    } else {
        // split the data
        std::vector<int> leftNodeSamples, rightNodeSamples;
        evalNode(data,sampleIndeces,leftNodeSamples,rightNodeSamples);

        m_leftChildNode->eval(data,leftNodeSamples,confidences,predictions);
        m_rightChildNode->eval(data,rightNodeSamples,confidences,predictions);
    }
}


