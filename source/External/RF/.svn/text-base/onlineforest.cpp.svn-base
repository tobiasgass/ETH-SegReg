#include "onlineforest.h"

#include "forest.h"
#include <string.h>
#include <fstream>
#include <boost/foreach.hpp>

OnlineForest::OnlineForest(const HyperParameters &hp)
{
  m_hp = &hp;
}

void OnlineForest::initialize(const int numSamples)
{

    m_confidences.resize(numSamples, m_hp->numClasses);
    m_predictions.resize(numSamples);

    for (int n = 0; n < numSamples; n++)
    {
        for (int m = 0; m < m_hp->numClasses; m++)
        {
            m_confidences(n, m) = 0;
        }
    }
}

void OnlineForest::train(const matrix<float>& data, const std::vector<int>& labels, bool use_gpu)
{
    // Initialize
    initialize(data.size1());

    if (m_hp->useGPU || use_gpu)
    {
        cerr << "ERROR: No GPU Implementation available for online forests!" << endl;
        //std::vector<double> weights(labels.size(),1);
        //trainByGPU(data,labels, weights);
    }
    else
    {
        trainByCPU(data,labels);
    }
}

OnlineForest::~OnlineForest()
{
    m_trees.clear();
}

void OnlineForest::eval(const matrix<float>& data, const std::vector<int>& labels, bool use_gpu)
{
    if (m_hp->useGPU || use_gpu)
    {
        cerr << "ERROR: No GPU Implementation available for online forests!" << endl;
        //evalByGPU(data, labels);
    }
    else
    {
        evalByCPU(data, labels);
    }
}

void OnlineForest::load(const std::string &name)
{
    std::string loadName;
    loadName = (name == "default") ? m_hp->loadName : name;
    // now load that stuff
}

// CPU penne code only below this line

int poisson(double A)
{
    int k = 0;
    int maxK = 10;
    while(1) {
        double U_k = _rand();
        A *= U_k;
        if( k > maxK || A < exp(-1.0) ) {
            break;
        }
        k++;
    }
    return k;
}

void OnlineForest::trainByCPU(const matrix<float>& data, const std::vector<int>& labels)
{
    HyperParameters tmpHP = *m_hp;
    tmpHP.verbose = false;

    m_trees.clear();
    // Create trees
    for (int tree = 0; tree < m_hp->numTrees; tree++)
    {
        m_trees.push_back(OnlineTree(tmpHP,(int)data.size2()));
    }

    if (m_hp->verbose)
    {
        cout << "Training a random forest with " << m_hp->numTrees << " , grab a coffee ... " << endl;
        cout << "\tEpoch #: ";
    }

    // Train Forest
    std::vector<int> randperm;
    for (int round = 0; round < m_hp->numOnlineEpochs; round++)
    {
        if (m_hp->verbose)
        {
            cout << round + 1 << " " << flush;
        }
        //randperm = randPerm(data.size1());
        randperm = randPerm(m_hp->numLabeled);
        for ( int s = 0; s < m_hp->numLabeled; s++)
        {
            std::vector<float> sample(data.size2(),0);
            for ( int feature = 0; feature < (int)data.size2(); feature++)
            {
                sample[feature] = data(randperm[s],feature);
            }
            std::vector<OnlineTree>::iterator treeIt(m_trees.begin());
            std::vector<OnlineTree>::iterator treeEnd(m_trees.end());
            while ( treeIt != treeEnd )
            {
                double k = poisson(1.0);
                for(int curK = 0; curK < k; curK++) {
                    treeIt->train(sample,labels[randperm[s]]);
                }
                ++treeIt;
            }
        }
        //cout << "bla" << endl;
    }
    if (m_hp->verbose)
    {
        cout << " Done." << endl;
    }

    // Eval Forest
    double error = 0.0;
    for ( int s = 0; s < m_hp->numLabeled;s++)
    {
        std::vector<float> sample(data.size2(),0);
        for ( int feature = 0; feature < (int)data.size2(); feature++)
        {
            sample[feature] = data(s,feature);
        }

        std::vector<int> predictions(m_hp->numClasses,0.0);
        std::vector<OnlineTree>::iterator treeIt(m_trees.begin());
        std::vector<OnlineTree>::iterator treeEnd(m_trees.end());
        while ( treeIt != treeEnd )
        {
            int prediction = (int)treeIt->eval(sample,labels[s]);
            predictions[prediction]++;
            ++treeIt;
        }

        int maxPrediction = 0;
        int bestClass = 0;
        int c = 0;
        BOOST_FOREACH(int pred,predictions){
            if(pred > maxPrediction){
                maxPrediction = pred;
                bestClass = c;
            }
            c++;
        }
        if (bestClass != labels[s])
        {
            error++;
        }
    }

    if (m_hp->verbose)
    {
        cout << "\tTraining Error: " << error/(double)m_hp->numLabeled << endl;
    }

}

void OnlineForest::evalByCPU(const matrix<float>& data, const std::vector<int>& labels)
{
   // Eval Forest
    double error = 0.0;
    for ( int s = 0; s < (int) data.size1();s++)
    {
        std::vector<float> sample(data.size2(),0);
        for ( int feature = 0; feature < (int)data.size2(); feature++)
        {
            sample[feature] = data(s,feature);
        }

        std::vector<int> predictions(m_hp->numClasses,0.0);
        std::vector<OnlineTree>::iterator treeIt(m_trees.begin());
        std::vector<OnlineTree>::iterator treeEnd(m_trees.end());
        while ( treeIt != treeEnd )
        {
            int prediction = (int)treeIt->eval(sample,labels[s]);
            predictions[prediction]++;
            ++treeIt;
        }

        int maxPrediction = 0;
        int bestClass = 0;
        int c = 0;
        BOOST_FOREACH(int pred,predictions){
            if(pred > maxPrediction){
                maxPrediction = pred;
                bestClass = c;
            }
            c++;
        }
        if (bestClass != labels[s])
        {
            error++;
        }
    }
    if(m_hp->verbose) {
        cout << "\tTest Error: " << error/(double) data.size1() << endl;
    }
}

double OnlineForest::computeError(const std::vector<int>& labels)
{
    int bestClass, nSamp = 0;
    float bestConf;
    double error = 0;
    BOOST_FOREACH(int pre, m_predictions)
    {
        bestClass = 0;
        bestConf = 0;
        for (int nClass = 0; nClass < (int) m_hp->numClasses; nClass++)
        {
            if (m_confidences(nSamp, nClass) > bestConf)
            {
                bestClass = nClass;
                bestConf = m_confidences(nSamp, nClass);
            }
        }

        pre = bestClass;
        if (bestClass != labels[nSamp])
        {
            error++;
        }

        nSamp++;
    }
    error /= (double) m_predictions.size();

    return error;
}

double OnlineForest::computeError(const std::vector<int>& labels, const matrix<float>& confidences,
                                  const std::vector<int>& voteNum)
{
    double error = 0, bestConf;
    int sampleNum = 0, bestClass;
    for (int nSamp = 0; nSamp < (int) confidences.size1(); nSamp++)
    {
        bestClass = 0;
        bestConf = 0;
        if (voteNum[nSamp])
        {
            sampleNum++;
            for (int nClass = 0; nClass < (int) m_hp->numClasses; nClass++)
            {
                if (confidences(nSamp, nClass) > bestConf)
                {
                    bestClass = nClass;
                    bestConf = confidences(nSamp, nClass);
                }
            }

            if (bestClass != labels[nSamp])
            {
                error++;
            }
        }
    }
    error /= (double) sampleNum;

    return error;
}


void OnlineForest::writeError(const std::string& fileName, double error)
{
    std::ofstream myfile;
    myfile.open(fileName.c_str(), std::ios::app);
    myfile << error << endl;
    myfile.flush();
    myfile.close();
}


