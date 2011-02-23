/*
 * hyperparametes.h
 *
 *  Created on: Jan 27, 2009
 *      Author: leisti
 */

#ifndef HYPERPARAMETES_H_
#define HYPERPARAMETES_H_
#include <string>

struct HyperParameters
{
  int numTrees;
  int maxTreeDepth;
  float bagRatio;
  float confThreshold;
  int numRandomFeatures;
  int numProjFeatures;
  int useRandProj;
  int useGPU;
  int useSubSamplingWithReplacement;
  int useSoftVoting;
  int useInfoGain;
  int numClasses;
  int verbose;
  int saveForest;
  int numLabeled;
  int numHistogramBins;
  int numTries;
  std::string saveName;
  std::string savePath;
  std::string loadName;
  std::string trainData;
  std::string trainLabels;
  std::string testData;
  std::string testLabels;
  std::string naiveBayesFeatureType;
  int isExtreme;
};

#endif /* HYPERPARAMETES_H_ */
