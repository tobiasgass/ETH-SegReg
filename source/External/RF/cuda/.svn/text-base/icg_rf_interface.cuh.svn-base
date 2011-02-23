#ifndef ICG_RF_INTERFACE_CUH_
#define ICG_RF_INTERFACE_CUH_

#include "icg_rf_defs.cuh"

//-----------------------------------------------------------------------------
bool iTrainForest(
	const Cuda::Array<float,2>& input,
	const Cuda::DeviceMemoryLinear<int,1>& labels,
	const Cuda::DeviceMemoryLinear<float,1>& weights,
	Cuda::DeviceMemoryPitched<float,2>** forest,
	float* oobe, 
	const int num_samples, const int num_classes, 
	const int num_trees, const int max_depth, const int num_tree_cols,  
	const int num_hypotheses, const int num_features,
	const float bag_ratio
);

//-----------------------------------------------------------------------------
inline int iGetNumTreeCols(
	const int num_features, const int num_classes)
{
	if (num_features * 2 > num_classes)
		return 1 + num_features * 2 + 1;
	else
		return 1 + num_classes + 1;
} 

//-----------------------------------------------------------------------------
bool iEvaluateForest(
	const Cuda::Array<float,2>& forest, 
	const Cuda::Array<float,2>& input,
	Cuda::DeviceMemoryPitched<float,2>* confidences,
	Cuda::DeviceMemoryLinear<float,1>* predictions,
	const int num_samples, const int num_trees, const int max_depth, const int num_tree_cols, 
    const int num_classes, const int num_features, const bool use_soft_voting
);



#endif //ICG_RF_INTERFACE_CUH_