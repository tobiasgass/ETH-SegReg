#include "dasrandomnaivebayes.h"
#include "naivebayeshyperplane.h"
#include "naivebayesfeature.h"
#include <boost/foreach.hpp>

DASRandomNaiveBayes::DASRandomNaiveBayes(const HyperParameters &hp) : RandomNaiveBayes( hp ), m_sim(hp)
{
}

DASRandomNaiveBayes::~DASRandomNaiveBayes()
{
}

void DASRandomNaiveBayes::train(const matrix<float>& dataTr, const std::vector<int>& labelsTr, const matrix<float>& dataTs,
                                const std::vector<int>& labelsTs,bool use_gpu)
{
    // Initialize
    initialize(dataTr.size1());

    if (m_hp.useGPU || use_gpu)
    {
        std::vector<double> weights(labelsTr.size(),1);
        //trainByGPU(data,labels, weights);
        std::cout << "ERROR: Not implemented yet! Jakob, do it!" << std::endl;
    }
    else
        trainByCPU(dataTr,labelsTr,dataTs,labelsTs);

}

void DASRandomNaiveBayes::trainByCPU(const matrix<float>& dataTr, const std::vector<int>& labelsTr,
                                     const matrix<float>& dataTs, const std::vector<int>& labelsTs)
{
    matrix<float> trainConf(dataTr.size1(), m_hp.numClasses);
    std::vector<int> trainPre(dataTr.size1());
    // First round without DA
    bool verbose = m_hp.verbose;
    HyperParameters tmpHP = m_hp;
    tmpHP.verbose = false;
    std::vector<int> outOfBagVoteCount(dataTr.size1(), 0);
    matrix<float> outOfBagConfidences(dataTr.size1(),m_hp.numClasses);

    for ( unsigned int i = 0; i < dataTr.size1(); i++)
    {
        for ( int j = 0; j < m_hp.numClasses; j++)
        {
            outOfBagConfidences(i,j) = 0.0;
        }
    }

    m_naiveBayesClassifiers.clear(); // be sure that they are empty
    for (int i = 0; i < m_hp.numTrees; i++)
    {
        NaiveBayes::Ptr naiveClassifier;
        if (m_hp.useRandProj)
        {
            naiveClassifier = NaiveBayes::Ptr(new NaiveBayesHyperplane(tmpHP));
        }
        else
        {
            naiveClassifier = NaiveBayes::Ptr(new NaiveBayesFeature(tmpHP));
        }
        naiveClassifier->train(dataTr,labelsTr, m_confidences, outOfBagConfidences, outOfBagVoteCount);
        m_naiveBayesClassifiers.push_back(naiveClassifier);
    }

    eval(dataTs,labelsTs);
    double rnbError = computeError(labelsTs);
    cout << "Supervised Error: " << rnbError << endl;

    // Compute OOBE
    double rnbOOBE = computeOOBE(labelsTr);
    double dasrnbOOBE = rnbOOBE;
    if (m_hp.verbose)
    {
        cout << "\tNaive Bayes OOBE = " << rnbOOBE << endl;
    }

    // Make Similarity
    if (m_hp.lambda)
    {
        m_sim = Similarity(m_hp);
        m_sim.train(dataTr, labelsTr);
        m_simConf = m_sim.getSimConf(dataTr, labelsTs);
        //m_simConf = m_sim.getSimConfChi2(dataTr, labelsTs);
    }

    std::vector<std::vector<double> > tmpWeights;
    std::vector<std::vector<int> > tmpLabels;
    std::vector<int> classifiersToBeTrained;
    double numberOfSwitches = 0.0;
    bool init = true;
    int numTrainedClassifiers = 0;
    // Iterative Semi-Supervised Training
    for (int nEpoch = 0; nEpoch < m_hp.numEpochs; nEpoch++)
    {
        cout << "Epoch: " << nEpoch << endl;
        calcLabelsAndWeights(trainConf, trainPre, tmpLabels, tmpWeights, nEpoch, labelsTr, numberOfSwitches, classifiersToBeTrained);

        numTrainedClassifiers = 0;
        BOOST_FOREACH(int nClassifier, classifiersToBeTrained)
        {
            m_naiveBayesClassifiers[nClassifier]->retrain(dataTr,tmpLabels[numTrainedClassifiers], tmpWeights[numTrainedClassifiers], m_confidences, outOfBagConfidences, outOfBagVoteCount,init);
            numTrainedClassifiers++;
        }

        init = false;

        if (numTrainedClassifiers)
        {
            // Evaluate the training set
            m_hp.verbose = false;
            eval(dataTr, labelsTr);
            m_hp.verbose = verbose;
            trainConf = m_confidences;
            trainPre = m_predictions;

            // Compute the OOBE
            dasrnbOOBE = computeOOBE(labelsTr);
            if (m_hp.verbose)
            {
                cout << "\t#Retrained classifiers = " << numTrainedClassifiers << ", #switches/temperature = ";
                cout << numberOfSwitches/(double) numTrainedClassifiers << ", SRF OOBE = " << dasrnbOOBE << "\t";
            }

            eval(dataTs, labelsTs);
            /*
            // Check for the stopping condition
            if (dasrnbOOBE > rnbOOBE + m_hp.maxOOBEIncrease)
            {
                // Reset the classifiers
                m_naiveBayesClassifiers.clear();
                for (int i = 0; i < m_hp.numTrees; i++)
                {
                    Tree t(tmpHP);
                    t.train(dataTr, labelsTr);
                    m_trees.push_back(t);
                }

                // Evaluate the training set
                m_hp.verbose = false;
                eval(dataTr, labelsTr);
                m_hp.verbose = verbose;
                trainConf = m_confidences;
                trainPre = m_predictions;

                rfOOBE = computeOOBE(labelsTr);
                srfOOBE = rfOOBE;
                if (m_hp.verbose)
                {
                    cout << "\tForest OOBE = " << rfOOBE << endl;
                }

                eval(dataTs, labelsTs, false);
                rfError = computeError(labelsTs);

                if (curTrial > numTrials)   // Give back the RF
                {
                    break;
                }
                else   // Make a reset and start the DA again
                {
                    init = true;
                    nEpoch = -1;
                }
            }
            */
        }
        else
        {
            if (m_hp.verbose)
            {
                cout << "\tno naive classifiers were retrained" << endl;
            }
            break;
        }
    }

    eval(dataTs, labelsTs);
    double dasrnbError = computeError(labelsTs);

    if (m_hp.verbose)
    {
        cout << endl << "\tTraining error = " << dasrnbError << endl;
        cout << "\tSemi-supervised improvement = " << 100*(rnbError - dasrnbError)/rnbError << "%" << endl;
    }
}

void DASRandomNaiveBayes::calcLabelsAndWeights(const matrix<float>& RFConf, const std::vector<int>& RFPre,
        std::vector<std::vector<int> >& tmpLabels, std::vector<std::vector<double> >& tmpWeights, const int nEpoch,
        const std::vector<int>& labels, double& numberOfSwitchs,
        std::vector<int>& classifiersToBeTrained)
{
    double p, entropy;
    double alpha = (double) m_hp.numLabeled / ((double) (RFPre.size() - m_hp.numLabeled));
    alpha *= m_hp.alpha;
    bool useEntropy = true;
    std::vector<double> cumSum(m_hp.numClasses);
    tmpLabels.clear();
    tmpWeights.clear();
    classifiersToBeTrained.clear();
    numberOfSwitchs = 0;
    std::vector<int> Y(RFPre.size());
    std::vector<double> W(RFPre.size());
    for (int nSamp = 0; nSamp < m_hp.numLabeled; nSamp++)
    {
        Y[nSamp] = labels[nSamp];
        W[nSamp] = 1.0;
    }

    for (int nTree = 0; nTree < m_hp.numTrees; nTree++)
    {
        if (shouldITrain(nEpoch))   // We are training this tree
        {
            classifiersToBeTrained.push_back(nTree);
            for (int nSamp = m_hp.numLabeled; nSamp < (int) RFPre.size(); nSamp++)
            {
                if (shouldISwitch(nEpoch))
                {
                    numberOfSwitchs++;
                    p = _rand();
                    if (m_hp.lambda)
                    {
                        cumSum[0] = (1 - m_hp.lambda)*RFConf(nSamp, 0) + m_hp.lambda*m_simConf(nSamp - m_hp.numLabeled, 0);
                        for (int m = 1; m < m_hp.numClasses; m++)
                        {
                            cumSum[m] = cumSum[m - 1] + (1 - m_hp.lambda)*RFConf(nSamp, m) + m_hp.lambda*m_simConf(nSamp - m_hp.numLabeled, m);
                        }
                    }
                    else
                    {
                        cumSum[0] = RFConf(nSamp, 0);
                        for (int m = 1; m < m_hp.numClasses; m++)
                        {
                            cumSum[m] = cumSum[m - 1] + RFConf(nSamp, m);
                        }
                    }
                    for (int m = 0; m < m_hp.numClasses; m++)
                    {
                        if (cumSum[m] > p)
                        {
                            Y[nSamp] = m;
                            break;
                        }
                    }
                }
                else   // Pick the prediction of the RF
                {
                    Y[nSamp] = RFPre[nSamp];
                }

                if (useEntropy)
                {
                    entropy = 0;
                    if (m_hp.lambda)
                    {
                        for (int nClass = 0; nClass < m_hp.numClasses; nClass++)
                        {
                            if (RFConf(nSamp, nClass))
                            {
                                entropy -= ((1 - m_hp.lambda)*RFConf(nSamp, nClass) + m_hp.lambda*m_simConf(nSamp - m_hp.numLabeled, nClass))*
                                           log((1 - m_hp.lambda)*RFConf(nSamp, nClass) + m_hp.lambda*m_simConf(nSamp - m_hp.numLabeled, nClass));
                            }
                        }
                    }
                    else
                    {
                        for (int nClass = 0; nClass < m_hp.numClasses; nClass++)
                        {
                            if (RFConf(nSamp, nClass))
                            {
                                entropy -= RFConf(nSamp, nClass)*log(RFConf(nSamp, nClass));
                            }
                        }
                    }

                    W[nSamp] = alpha*(log((double) m_hp.numClasses) - entropy);
                }
                else
                {
                    W[nSamp] = alpha;
                }
            }

            tmpLabels.push_back(Y);
            tmpWeights.push_back(W);
        }
    }
}

double DASRandomNaiveBayes::temperature(const int nEpoch)
{
    double TMax = 100, coolingRate = 0.25;
    double T = pow(coolingRate, (double) nEpoch)*TMax;
    return (T < 1e-6) ? 1e-6 : T;
}

bool DASRandomNaiveBayes::shouldISwitch(const int nEpoch)
{
    double maxP = m_hp.sampMaxP, minP = 0.0, CF = m_hp.sampCF;
    double p = exp(CF*nEpoch);
    p = (p > 0.0) ? p : 0.0;
    if (p > maxP)
    {
        p = maxP;
    }
    if (p < minP)
    {
        p = minP*(1 - (double) nEpoch/(double) m_hp.numEpochs);
    }

    return (_rand() < p) ? true : false;
}

bool DASRandomNaiveBayes::shouldITrain(const int nEpoch)
{
    double maxP = m_hp.treeMaxP, minP = 0.0, CF = m_hp.treeCF;
    double p = exp(CF*nEpoch);
    p = (p > 0.0) ? p : 0.0;
    if (p > maxP)
    {
        p = maxP;
    }
    if (p < minP)
    {
        p = minP*(1 - (double) nEpoch/(double) m_hp.numEpochs);
    }

    return (_rand() < p) ? true : false;
}

double DASRandomNaiveBayes::computeOOBE(const std::vector<int>& labels)
{
    double oobe = 0;
    matrix<float> confidence(m_hp.numLabeled, m_hp.numClasses);
    std::vector<int> voteNum(m_hp.numLabeled, 0), treePre, treeOBS;
    matrix<float> treeConf;
    for (int n = 0; n < m_hp.numLabeled; n++)
    {
        for (int m = 0; m < m_hp.numClasses; m++)
        {
            confidence(n, m) = 0.0;
        }
    }

    BOOST_FOREACH(NaiveBayes::Ptr nb, m_naiveBayesClassifiers)
    {
        treePre = nb->getPredictions();
        treeOBS = nb->getOutOfBagSamples();
        treeConf = nb->getConfidences();
        BOOST_FOREACH(int m, treeOBS)
        {
            if (m < m_hp.numLabeled)
            {
                if (m_hp.useSoftVoting)
                {
                    for (int nClass = 0; nClass < m_hp.numClasses; nClass++)
                    {
                        confidence(m, nClass) += treeConf(m, nClass);
                    }
                }
                else
                {
                    confidence(m, treePre[m])++;
                }
                voteNum[m]++;
            }
            else
            {
                break;
            }
        }
    }

    int bestClass, totalNum = 0;
    double bestConf;
    std::vector<int>::const_iterator labelItr(labels.begin()), labelEnd(labels.end());
    for (int n = 0; labelItr != labelEnd; labelItr++, n++)
    {
        if (n < m_hp.numLabeled && voteNum[n])
        {
            bestClass = 0;
            bestConf = 0;
            for (int m = 0; m < m_hp.numClasses; m++)
            {
                if (confidence(n, m) > bestConf)
                {
                    bestConf = confidence(n, m);
                    bestClass = m;
                }
            }

            if (*labelItr != bestClass)
            {
                oobe++;
            }

            totalNum++;
        }
    }
    oobe /= (double) totalNum;

    return oobe;
}
