#include "icg_rf_interface.cuh"
#include "icg_rf_forest.cu"

//-----------------------------------------------------------------------------
bool iTrainForest(const Cuda::Array<float,2>& input,
                  const Cuda::DeviceMemoryLinear<int,1>& labels,
                  const Cuda::DeviceMemoryLinear<float,1>& weights,
                  Cuda::DeviceMemoryPitched<float,2>** forest,
                  float* oobe,
                  const int num_samples, const int num_classes, 
                  const int num_trees, const int max_depth, const int num_tree_cols, 
                  const int num_hypotheses, const int num_features,
                  const float bag_ratio)
{
  return cudaTrainForest(input, labels, weights, forest, oobe,
    num_samples, num_classes, num_trees, max_depth, num_tree_cols,  
    num_hypotheses, num_features, bag_ratio);
}

//-----------------------------------------------------------------------------
bool iEvaluateForest(const Cuda::Array<float,2>& forest, 
                     const Cuda::Array<float,2>& input,
                     Cuda::DeviceMemoryPitched<float,2>* confidences,
                     Cuda::DeviceMemoryLinear<float,1>* predictions,
                     const int num_samples, const int num_trees, const int max_depth, const int num_tree_cols,
                     const int num_classes, const int num_features, const bool use_soft_voting) 
{
  return cudaEvaluateForest(
    forest, input, confidences, predictions, 
    num_samples, num_trees, max_depth, num_tree_cols,  
    num_classes, num_features, use_soft_voting);
}

