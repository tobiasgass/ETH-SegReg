#include "similarity.h"
#include "forest.h"
#include <string.h>
#include <boost/foreach.hpp>

Similarity::Similarity(const HyperParameters &hp) : m_hp( hp )
{

}

Similarity::~Similarity()
{

}

void Similarity::train(const matrix<float>& data, const std::vector<int>& labels)
{
  if (m_hp.verbose) {
    cout << "\tTraining the similarity ...";
  }
  
  HyperParameters tmpHP = m_hp;
  tmpHP.verbose = false;

  m_trees.clear();
  m_trees.reserve(m_hp.numTrees);
  bool useLU = false;
  for (int i = 0; i < m_hp.numTrees; i++) {
    Tree t(tmpHP);
    if (useLU) {
      t.trainLU(data,labels);
    }
    else {
      t.train(data,labels);
    }
    m_trees.push_back(t);
  }

  if (m_hp.verbose) {
    cout << "Done." << endl;
  }

  m_labels = labels;
}

void Similarity::train(const matrix<float>& data, const std::vector<int>& labels, const double posW)
{
  if (m_hp.verbose) {
    cout << "\tTraining the similarity ... " << flush;
  }
  
  HyperParameters tmpHP = m_hp;
  tmpHP.verbose = false;

  m_trees.clear();
  m_trees.reserve(m_hp.numTrees);
  bool useLU = false;
  std::vector<double> weights(labels.size(), 1.0);
  for (int nSamp = 0; nSamp < m_hp.numLabeled; nSamp++) {
    if (labels[nSamp]) {
      weights[nSamp] = posW;
    }
  }
  for (int i = 0; i < m_hp.numTrees; i++) {
    Tree t(tmpHP);
    if (useLU) {
      t.trainLU(data,labels);
    }
    else {
      t.train(data,labels,weights);
    }
    m_trees.push_back(t);
  }
  
  if (m_hp.verbose) {
    cout << "Done." << endl;
  }

  m_labels = labels;
}

matrix<float> Similarity::getSimConf(const matrix<float>& data) {
  // Initialize
  int numUnlabeled = (int) data.size1() - m_hp.numLabeled;
  matrix<float> simConf(numUnlabeled, m_hp.numClasses);
  for (int n = 0; n < numUnlabeled; n++) {
    for (int m = 0; m < m_hp.numClasses; m++) {
      simConf(n, m) = 0.0;
    }
  }

  // Compute the similarities: ask the forest for getting the pathes
  std::vector<std::vector<int> > path;
  std::vector<int> pathL, pathU;
  int minDepth, treeDepth;
  double sim, totalSim, depthTotalWeight;
  if (m_hp.verbose) {
    cout << "\tComputing the RF Similarity ... " << flush;
  }
  BOOST_FOREACH(Tree t, m_trees) {
    path = t.getPath(data);
    treeDepth = t.getDepth();
    depthTotalWeight = 0;
    for (int d = 1; d < treeDepth; d++) {
      depthTotalWeight += pow(2.0, (double) d);
    }
    
    for (int nU = 0; nU < numUnlabeled; nU++) {
      pathU = path[nU + m_hp.numLabeled];
      std::vector<double> classSim(m_hp.numClasses, 0.0);
      totalSim = 0;
      for (int nL = 0; nL < m_hp.numLabeled; nL++) {
        sim = 0;
        pathL = path[nL];
        minDepth = (pathL.size() > pathU.size()) ? pathU.size() : pathL.size();
        for (int d = 1; d < minDepth; d++) {
          if (pathL[d] == pathU[d]) {
            sim += pow(2.0, (double) d)/depthTotalWeight;
          }
          else {
            break;
          }
        }
        
        classSim[m_labels[nL]] += sim;
        totalSim += sim;
      }

      for (int nClass = 0; nClass < m_hp.numClasses; nClass++) {
        simConf(nU, nClass) += (float)(classSim[nClass]/totalSim);
      }
    }
  }
  simConf /= (double) m_hp.numTrees;
  if (m_hp.verbose) {
    cout << "Done." << endl;
  }
  
  return simConf;
}

matrix<float> Similarity::getSimConf(const matrix<float>& data, const std::vector<int>& labels) {
  // Initialize
  int numUnlabeled = (int) data.size1() - m_hp.numLabeled;
  matrix<float> simConf(numUnlabeled, m_hp.numClasses);
  for (int n = 0; n < numUnlabeled; n++) {
    for (int m = 0; m < m_hp.numClasses; m++) {
      simConf(n, m) = 0.0;
    }
  }

  // Compute the similarities: ask the forest for getting the pathes
  std::vector<std::vector<int> > path;
  std::vector<int> pathL, pathU;
  int minDepth, treeDepth;
  double sim, totalSim, depthTotalWeight;
  if (m_hp.verbose) {
    cout << "\tComputing the RF Similarity ... " << flush;
  }
  BOOST_FOREACH(Tree t, m_trees) {
    path = t.getPath(data);
    treeDepth = t.getDepth();
    depthTotalWeight = 0;
    for (int d = 1; d < treeDepth; d++) {
      depthTotalWeight += pow(2.0, (double) d);
    }
    
    for (int nU = 0; nU < numUnlabeled; nU++) {
      pathU = path[nU + m_hp.numLabeled];
      std::vector<double> classSim(m_hp.numClasses, 0.0);
      totalSim = 0;
      for (int nL = 0; nL < m_hp.numLabeled; nL++) {
        sim = 0;
        pathL = path[nL];
        minDepth = (pathL.size() > pathU.size()) ? pathU.size() : pathL.size();
        for (int d = 1; d < minDepth; d++) {
          if (pathL[d] == pathU[d]) {
            sim += pow(2.0, (double) d)/depthTotalWeight;
          }
          else {
            break;
          }
        }
                
        classSim[m_labels[nL]] += sim;
        totalSim += sim;
      }

      for (int nClass = 0; nClass < m_hp.numClasses; nClass++) {
        simConf(nU, nClass) += classSim[nClass]/totalSim;
      }
    }
  }
  simConf /= (double) m_hp.numTrees;
  if (m_hp.verbose) {
    cout << "Done." << endl;
  }

  int bestClass;
  double error = 0, bestConf;
  for (int n = 0; n < numUnlabeled; n++) {
    bestClass = 0;
    bestConf = 0;
    for (int m = 0; m < m_hp.numClasses; m++) {
      if (bestConf < simConf(n, m)) {
        bestClass = m;
        bestConf = simConf(n, m);
      }
    }
    if (labels[n] != bestClass) {
      error++;
    }
  }

  cout << "\tSimilarity error = " << error/((double) numUnlabeled) << endl;
  
  return simConf;
}

matrix<float> Similarity::getSimConfChi2(const matrix<float>& data, const std::vector<int>& labels) {
  // Initialize
  int numUnlabeled = (int) data.size1() - m_hp.numLabeled;
  matrix<float> simConf(numUnlabeled, m_hp.numClasses);
  for (int n = 0; n < numUnlabeled; n++) {
    for (int m = 0; m < m_hp.numClasses; m++) {
      simConf(n, m) = 0.0;
    }
  }

  // Compute Chi2 Similarity
  matrix<float> chi2Sim(numUnlabeled, m_hp.numLabeled);
  double chi2DistSum = 0, beta = m_hp.beta;
  bool usePyramidKernel = m_hp.usePyramidKernel;
  int numLevel = 4, bin = 8, numRep = 2, repNumFeat = data.size2()/numRep, offset;
  double levelWeight;
  if (beta) {
    if (m_hp.verbose) {
      cout << "\tComputing the Chi2 Kernel ... " << flush;
    }
    for (int nU = 0; nU < numUnlabeled; nU++) {
      for (int nL = 0; nL < m_hp.numLabeled; nL++) {
        chi2Sim(nU, nL) = 0;
        if (usePyramidKernel) {
          for (int nR = 0; nR < numRep; nR++) {
            offset = 0;
            for (int nLevel = 0; nLevel < numLevel; nLevel++) {
              levelWeight = 1.0/pow(2.0, numLevel - nLevel);
              for (int nF = nR*repNumFeat + offset; nF < nR*repNumFeat + offset + pow(4.0, nLevel)*bin; nF++) {
                chi2Sim(nU, nL) += levelWeight*pow((double) data(nU + m_hp.numLabeled, nF) - (double) data(nL, nF), 2.0)
                  /(data(nU + m_hp.numLabeled, nF) + data(nL, nF) + 1e-6);
              }
              offset += (int) pow(4.0, nLevel)*bin;
            }
          }
        }
        else {
          for (int nF = 0; nF < (int) data.size2(); nF++) {
            chi2Sim(nU, nL) += pow((double) data(nU + m_hp.numLabeled, nF) - (double) data(nL, nF), 2.0)
              /(data(nU + m_hp.numLabeled, nF) + data(nL, nF) + 1e-6);
          }
          chi2Sim(nU, nL) *= 0.5;
        }
        chi2DistSum += chi2Sim(nU, nL);
      }
    }
    chi2DistSum /= m_hp.numLabeled*numUnlabeled;

    for (int nU = 0; nU < numUnlabeled; nU++) {
      for (int nL = 0; nL < m_hp.numLabeled; nL++) {
        chi2Sim(nU, nL) = exp(-chi2Sim(nU, nL)/(2*chi2DistSum));
      }
    }

    if (m_hp.verbose) {
      cout << "Done." << endl;
    }
  }

  // Compute the similarities: ask the forest for getting the pathes
  std::vector<std::vector<int> > path;
  std::vector<int> pathL, pathU;
  int minDepth, treeDepth;
  double sim, totalSim, depthTotalWeight;
  bool sharpen = false;
  if (m_hp.verbose) {
    cout << "\tComputing the RF Similarity ... " << flush;
  }
  BOOST_FOREACH(Tree t, m_trees) {
    path = t.getPath(data);
    treeDepth = t.getDepth();
    depthTotalWeight = 0;
    for (int d = 1; d < treeDepth; d++) {
      depthTotalWeight += pow(2.0, (double) d);
    }
    
    for (int nU = 0; nU < numUnlabeled; nU++) {
      pathU = path[nU + m_hp.numLabeled];
      std::vector<double> classSim(m_hp.numClasses, 0.0);
      totalSim = 0;
      for (int nL = 0; nL < m_hp.numLabeled; nL++) {
        sim = 0;
        pathL = path[nL];
        minDepth = (pathL.size() > pathU.size()) ? pathU.size() : pathL.size();
        for (int d = 1; d < minDepth; d++) {
          if (pathL[d] == pathU[d]) {
            sim += pow(2.0, (double) d)/depthTotalWeight;
          }
          else {
            break;
          }
        }

        // Add Chi2 Similarity
        if (beta) {
          sim = (1 - beta)*sim + beta*chi2Sim(nU, nL);
        }

        // Sharpening
        if (sharpen) {
          sim = pow(sim, 2.0);
        }

        classSim[m_labels[nL]] += sim;
        totalSim += sim;
      }

      for (int nClass = 0; nClass < m_hp.numClasses; nClass++) {
        simConf(nU, nClass) += classSim[nClass]/totalSim;
      }
    }
  }
  simConf /= (double) m_hp.numTrees;
  if (m_hp.verbose) {
    cout << "Done." << endl;
  }

  // Make predictions by Similarity
  int bestClass;
  double error = 0, bestConf;
  for (int n = 0; n < numUnlabeled; n++) {
    bestClass = 0;
    bestConf = 0;
    for (int m = 0; m < m_hp.numClasses; m++) {
      if (bestConf < simConf(n, m)) {
        bestClass = m;
        bestConf = simConf(n, m);
      }
    }
    if (labels[n] != bestClass) {
      error++;
    }
  }

  cout << "\tSimilarity error = " << error/((double) numUnlabeled) << endl;
  
  return simConf;
}
