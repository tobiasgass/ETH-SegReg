#ifndef ICG_RF_FOREST_KERNELS_CU_
#define ICG_RF_FOREST_KERNELS_CU_

#define LSON_IDX 0
#define OVERHEAD_OFS 1
#define FEAT_IDX(x) OVERHEAD_OFS + x * 2 
#define WEIGHT_IDX(x) OVERHEAD_OFS + x * 2 + 1
#define THRESH_IDX OVERHEAD_OFS + tp.num_features * 2 
#define CONF_IDX(x) OVERHEAD_OFS + x
#define LABEL_IDX OVERHEAD_OFS + tp.num_classes


////////////////////////////////////////////////////////////////////////////////
//! Training Kernels
////////////////////////////////////////////////////////////////////////////////
texture<float, 2, cudaReadModeElementType> input_tex;
texture<int, 1, cudaReadModeElementType> labels_tex;
texture<float, 2, cudaReadModeElementType> r_weights_tex;
texture<int, 2, cudaReadModeElementType> r_features_tex;

struct TrainParams {
    int num_trees; 
    unsigned char tree_level; 
    unsigned char max_depth; 
    int num_samples; 
    unsigned char num_features; 
    unsigned char num_classes;
    unsigned char num_hyp;
    int num_hyp_2;
    int num_tree_cols;
};

//-----------------------------------------------------------------------------
// \param buffer    This structure is initialized with 
template <typename T>
__global__ void icg_rf_init_kernel(T* buffer, const size_t cols, 
                                   const size_t rows, const size_t buffer_p,
                                   const T value)
{
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < cols && row < rows)
        buffer[row * buffer_p + col] = value;
}

//-----------------------------------------------------------------------------
// \param bags      Bagging output, [num_trees x num_bag_samples] (w x h), holds for 
//                  every tree the indices of the samples within its bag
// \param affiliation   [num_trees x num_samples] (w x h), stores the node 
//                          index in every tree that each sample is currently
//                          falling in.
__global__ void icg_rf_bagging_kernel(int* bags, const size_t bags_p,
                                      int* affiliation, const size_t affiliation_p,
                                      const int num_trees, const int num_bag_samples)
{
    unsigned int tree_index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int sample_index = blockIdx.y*blockDim.y + threadIdx.y;
    if (sample_index < num_bag_samples && tree_index < num_trees) {
        int index = (int)bags[sample_index * bags_p + tree_index];
        affiliation[index * affiliation_p + tree_index] = 0;
    }
}

//-----------------------------------------------------------------------------
// Checks for every leaf node of every tree, whether it is 
//      1: a node, which becomes a leaf at this tree level
//      2: a node, which needs to be split at this level
//      3: a node, which has leaf node parents at any level
// \param forest    the forest structure
// \param affiliation   [num_trees x num_samples] (w x h), stores the node 
//                          index in every tree that each sample is currently
//                          falling in.
// \param node_type     [num_trees x num_leaf_nodes] (w x h), stores the type
//                      (1,2 or 3) of all leafs of all trees
__global__ void icg_rf_classify_leaf_nodes_kernel(
    const int* num_nodes_per_level, const size_t num_nodes_per_level_p,
    int* affiliation, const size_t affiliation_p,
    int* node_type, const size_t node_type_p,
    const TrainParams tp)
{
    unsigned int tree_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (tree_idx < tp.num_trees){
        unsigned int thread_leaf_index = blockIdx.y*blockDim.y + threadIdx.y;
        unsigned int leaf_idx = thread_leaf_index; 
        if (tp.tree_level > 0)
            leaf_idx = num_nodes_per_level[(tp.tree_level - 1) * num_nodes_per_level_p + tree_idx] + thread_leaf_index; 
        unsigned int leaf_idx_stop = num_nodes_per_level[tp.tree_level * num_nodes_per_level_p + tree_idx];

        if (leaf_idx < leaf_idx_stop) {
            // check whether this node is a split node
            bool is_split_node = false;
            if (tp.tree_level != tp.max_depth) {
                bool first_node_found = false;
                int first_label =  -1;                
                for (int sample = 0; sample < tp.num_samples; sample++){
                    if (affiliation[affiliation_p * sample + tree_idx] == leaf_idx) {
                        if (!first_node_found) 
                            first_label = tex1Dfetch(labels_tex,sample);
                        else 
                            if (first_label != tex1Dfetch(labels_tex,sample))
                                is_split_node = true; // this node contains samples of different labels -> split node 
                        first_node_found = true;                    
                    }
                }
            }
            int type = 1;
            if (is_split_node)
                type = 2;

            node_type[thread_leaf_index * node_type_p + tree_idx] = type;
        }
    }
}

//-----------------------------------------------------------------------------
// Assigns the feature indices to the forest structure
// \param forest    the forest structure
// \param split_subset  [num_split_nodes x 2] (w x h), stores in the first row
//                      the tree index, and in the second row the node index of
//                      a split node
__global__ void icg_rf_split_assign_feature_kernel(
    float** trees, const size_t tree_p,
    const int* split_subset, const size_t num_splits, const size_t split_subset_p, 
    const TrainParams tp)
{
    unsigned int split_idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int feat_idx = blockIdx.y*blockDim.y + threadIdx.y;

    if (split_idx < num_splits && feat_idx < tp.num_features) {
        unsigned int tree_idx = split_subset[split_idx];
        float* tree = trees[tree_idx];
        unsigned int node_idx = split_subset[split_subset_p + split_idx];
        
        // assign the feature indices directly to the forest
        int feature = tex2D(r_features_tex, split_idx, feat_idx);
        // assign feature index to forest
        tree[tree_p * node_idx + FEAT_IDX(feat_idx)] = feature;
        // assign lson index
        tree[tree_p * node_idx + LSON_IDX] = split_subset[split_subset_p * 2 + split_idx]; //FIXXME not good here
    }
}

//-----------------------------------------------------------------------------
// Computes the minimum and maximum threshold for every split node at the 
// current level.
// \param split_subset  [num_split_nodes x 2] (w x h), stores in the first row
//                      the tree index, and in the second row the node index of
//                      a split node
// \param affiliation   [num_trees x num_samples] (w x h), stores the node 
//                          index in every tree that each sample is currently
//                          falling in.
// \param thresh_limits     [num_split_nodes x num_hypotheses * 2] (w x h),
//                          stores the minimum and maximum response of all 
//                          samples in the current split node for all weight
//                          hypotheses.
__global__ void icg_rf_split_compute_thresh_kernel(
    const int* split_subset, const size_t num_splits, const size_t split_subset_p, 
    const int* affiliation, const size_t affiliation_p,
    float* thresh_limits, const size_t thresh_limits_p,
    const TrainParams tp)
{
    unsigned int split_idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int hyp_idx = blockIdx.y*blockDim.y + threadIdx.y;

    if (split_idx < num_splits && hyp_idx < tp.num_hyp) {
        unsigned int tree_idx = split_subset[split_idx];
        unsigned int node_idx = split_subset[split_subset_p + split_idx];

        // init the limits for this node 
        float thresh_min = 1e10;
        float thresh_max = -1e10;        

        // compute the limits from the responses of every affiliated sample
        for (int sample = 0; sample < tp.num_samples; sample++){ 
            if (affiliation[affiliation_p * sample + tree_idx] == node_idx) {
                // calulate threshold 
                float response = 0;
                for (unsigned char feature = 0; feature < tp.num_features; feature++){
                    int feature_index = tex2D(r_features_tex, split_idx, feature);
                    float xi = tex2D(input_tex, feature_index, sample);                               
                    float wi = tex2D(r_weights_tex, split_idx, hyp_idx * tp.num_features + feature);
                    response += xi*wi;                        
                } 
                if (response < thresh_min)
                    thresh_min = response;
                if (response > thresh_max)
                    thresh_max = response;
            }                        
        } 
        // assign threshold limits
        thresh_limits[thresh_limits_p * (hyp_idx * 2) + split_idx] = thresh_min;
        thresh_limits[thresh_limits_p * (hyp_idx * 2 + 1) + split_idx] = thresh_max;        
    }
}

//-----------------------------------------------------------------------------
// \param split_subset  [num_split_nodes x 2] (w x h), stores in the first row
//                      the tree index, and in the second row the node index of
//                      a split node
// \param weights       [num_samples], stores a sample weight
// \param affiliation   [num_trees x num_samples] (w x h), stores the node 
//                          index in every tree that each sample is currently
//                          falling in.
// \param thresh_limits     [num_split_nodes x num_hypotheses * 2] (w x h),
//                          stores the minimum and maximum response of all 
//                          samples in the current split node for all weight
//                          hypotheses.
// \param pk_l, pk_r    [num_split_nodes x num_hypotheses^2 * num_classes] (w x h),
//                      store the number of samples falling in the left and 
//                      right child node for every hypothesis and every threshold
__global__ void icg_rf_split_eval_thresh_kernel(
    const int* split_subset, const size_t num_splits, const size_t split_subset_p, 
    const float* weights,
    const int* affiliation, const size_t affiliation_p,
    float* thresh_limits, const size_t thresh_limits_p,
    float* pk_l, const size_t pk_l_p, float* pk_r, const size_t pk_r_p,
    const TrainParams tp)
{
    unsigned int split_idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int hyp_idx = blockIdx.y*blockDim.y + threadIdx.y;

    if (split_idx < num_splits && hyp_idx < tp.num_hyp) {
        unsigned int tree_idx = split_subset[split_idx];
        unsigned int node_idx = split_subset[split_subset_p + split_idx];
        
        float min_thresh = thresh_limits[thresh_limits_p * (hyp_idx * 2) + split_idx];
        float max_thresh = thresh_limits[thresh_limits_p * (hyp_idx * 2 + 1) + split_idx];
        float thresh_step = (max_thresh - min_thresh)/tp.num_hyp;
                            
        // evaluate node for each threshold to find best split                       
        for (int sample = 0; sample < tp.num_samples; sample++){  
            if (affiliation[affiliation_p * sample + tree_idx] == node_idx) {
                // compute this hypothesis
                float sum = 0;
                for (unsigned char feature = 0; feature < tp.num_features; feature++){
                    int feature_index = tex2D(r_features_tex, split_idx, feature);
                    float xi = tex2D(input_tex, feature_index, sample);
                    float wi = tex2D(r_weights_tex, split_idx, hyp_idx * tp.num_features + feature);
                    sum += xi*wi;
                }
                // evaluate all thresholds
                unsigned char label = tex1Dfetch(labels_tex,sample);
                float weight = weights[sample]; 
                for (unsigned char thresh_hyp = 0; thresh_hyp < tp.num_hyp; thresh_hyp++) {
                    float thresh = min_thresh + (thresh_hyp + 0.5f) * thresh_step; 
                    if (sum > thresh) 
                        pk_l[pk_l_p * (tp.num_hyp_2 * label + hyp_idx * tp.num_hyp + thresh_hyp) + split_idx] += weight;
                    else 
                        pk_r[pk_r_p * (tp.num_hyp_2 * label + hyp_idx * tp.num_hyp + thresh_hyp) + split_idx] += weight;

                }                       
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Examinates the structures pkl and pkr filled by icg_rf_split_eval_thresh_kernel
// and stores the best split by evaluating a Gini function over all hypotheses.
// \param forest    the forest structure
// \param split_subset  [num_split_nodes x 2] (w x h), stores in the first row
//                      the tree index, and in the second row the node index of
//                      a split node
// \param thresh_limits     [num_split_nodes x num_hypotheses * 2] (w x h),
//                          stores the minimum and maximum response of all 
//                          samples in the current split node for all weight
//                          hypotheses.
// \param pk_l, pk_r    [num_split_nodes x num_hypotheses^2 * num_classes] (w x h),
//                      store the number of samples falling in the left and 
//                      right child node for every hypothesis and every threshold
__global__ void icg_rf_split_select_best_kernel(
    float** trees, const size_t tree_p,
    const int* split_subset, const size_t num_splits, const size_t split_subset_p,
    float* thresh_limits, const size_t thresh_limits_p,
    float* pk_l, const size_t pk_l_p, float* pk_r, const size_t pk_r_p,
    const TrainParams tp)
{
    unsigned int split_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (split_idx < num_splits) {
        unsigned int tree_idx = split_subset[split_idx];
        float* tree = trees[tree_idx];
        unsigned int node_idx = split_subset[split_subset_p + split_idx];        

        // compute scores for the child nodes
        int min_index = 0;
        float min_score = 0.5f;
        for (int i = 0; i < tp.num_hyp_2; i++) {
            float elements_l = 0;
            float elements_r = 0;
            for (unsigned char label = 0; label < tp.num_classes; label++) {
                elements_l += pk_l[pk_l_p * (tp.num_hyp_2 * label + i) + split_idx];
                elements_r += pk_r[pk_r_p * (tp.num_hyp_2 * label + i) + split_idx];
            }            
            float gini_l = 0;
            float gini_r = 0;
            for (unsigned char label = 0; label < tp.num_classes; label++) {
                float pkl = pk_l[pk_l_p * (tp.num_hyp_2 * label + i) + split_idx] / elements_l;
                float pkr = pk_r[pk_r_p * (tp.num_hyp_2 * label + i) + split_idx] / elements_r;
                gini_l += pkl * (1 - pkl); 
                gini_r += pkr * (1 - pkr);
            }
            float score = (elements_l * gini_l + elements_r * gini_r) / (elements_l + elements_r);
            if (score < min_score){
                min_score = score;
                min_index = i;
            }
        }

        // find best hypothesis and best threshold
        unsigned char best_hyp_index = min_index / tp.num_hyp;
        unsigned char best_thresh_index = min_index % tp.num_hyp;

        // assign best weights to forest
        for (unsigned char i = 0; i < tp.num_features; i++){
            float wi = tex2D(r_weights_tex, split_idx, best_hyp_index * tp.num_features + i);
            tree[tree_p * node_idx + WEIGHT_IDX(i)] = wi;
        }

        // assign threshold to forest
        float min_thresh = thresh_limits[thresh_limits_p * (best_hyp_index * 2) + split_idx];
        float max_thresh = thresh_limits[thresh_limits_p * (best_hyp_index * 2 + 1) + split_idx];
        float thresh_step = (max_thresh - min_thresh)/tp.num_hyp;
        tree[tree_p * node_idx + THRESH_IDX] = min_thresh + (best_thresh_index + 0.5f ) * thresh_step;
    }
}

//-----------------------------------------------------------------------------
// Computes the confidences for each leaf node using the amount of samples of
// each class falling into that node.
// \param forest    the forest structure
// \param weights       [num_samples], stores a sample weight
// \param split_subset  [num_split_nodes x 2] (w x h), stores in the first row
//                      the tree index, and in the second row the node index of
//                      a split node
// \param affiliation   [num_trees x num_samples] (w x h), stores the node 
//                          index in every tree that each sample is currently
//                          falling in.
// \param node_type     [num_trees x num_leaf_nodes] (w x h), stores the type
//                      (1,2 or 3) of all leafs of all trees

__global__ void icg_rf_train_leaf_node_kernel(
    float** trees, const size_t tree_p,
    const float* weights, 
    const int* leaf_subset, const size_t num_leafs, const size_t leaf_subset_p, 
    const int* affiliation, const size_t affiliation_p,
    const TrainParams tp)
{

    unsigned int split_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (split_idx < num_leafs) {
        unsigned int tree_idx = leaf_subset[split_idx];
        float* tree = trees[tree_idx];
        unsigned int node_idx = leaf_subset[leaf_subset_p + split_idx];

        extern __shared__ int accu[];
        unsigned int accu_offset = threadIdx.x * tp.num_classes;
        for (int i = 0; i < tp.num_classes; i++)
          accu[accu_offset + i] = 0.0f;

        float total_weight = 0.0f;
        for (int sample = 0; sample < tp.num_samples; sample++){ 
            if (affiliation[affiliation_p * sample + tree_idx] == node_idx){
                unsigned char label = tex1Dfetch(labels_tex,sample);
                float weight = weights[sample];
                accu[accu_offset + label] += weight;
                total_weight += weight;
            }
        }
        // assign confidences and label
        if (total_weight > 0.0f) {
            int max_class = 0;
            int max_count = 0;
            for (unsigned char i = 0; i < tp.num_classes; i++) {
                tree[tree_p * node_idx + CONF_IDX(i)] = (float) accu[accu_offset + i] / total_weight;
                if (accu[accu_offset + i] > max_count) {
                    max_class = i;
                    max_count = accu[accu_offset + i]; 
                }
            }
            tree[tree_p * node_idx + LABEL_IDX] = max_class;
        }
        else {
            // due to numerical instabilities it can happen, that a sample falls directly on a threshold during 
            // score computation. Therefore it can (unlikely) happen that a node gets created, but no sample falls in
            // it during evaluation -> so we need to catch this case by inserting zero confidences...
            for (unsigned char i = 0; i < tp.num_classes; i++) 
                tree[tree_p * node_idx + CONF_IDX(i)] = 0.0f;
            tree[tree_p * node_idx + LABEL_IDX] = 0;
        }
        // assign left son's index
        tree[tree_p * node_idx + LSON_IDX] = -1;
    }
}

//-----------------------------------------------------------------------------
__global__ void icg_rf_eval_affinity_kernel(
    float** trees, const size_t tree_p,
    int *affiliation, size_t affiliation_p, TrainParams tp)
{
    unsigned int sample_index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tree_index = blockIdx.y*blockDim.y + threadIdx.y;  

    if (sample_index < tp.num_samples && tree_index < tp.num_trees) {
        if (affiliation[affiliation_p * sample_index + tree_index] != -1) {
            float* tree = trees[tree_index];
            // set initial index in tree
            unsigned int node_index = 0;
            // go through whole tree
            for (unsigned char level = 0; level < tp.max_depth; level++) {
                int left_son_index = tree[tree_p * node_index + LSON_IDX];
                // break if terminal node has been found 
                if (left_son_index < 1)
                    break;
                // evaluate at this node
                float sum = 0;
                for (unsigned char feature = 0; feature < tp.num_features; feature++) {                    
                    int feature_index = (int) tree[tree_p * node_index + FEAT_IDX(feature)];
                    float wi = tree[tree_p * node_index + WEIGHT_IDX(feature)];
                    float xi = tex2D(input_tex, feature_index, sample_index);
                    sum += xi * wi;
                }
                float theta = tree[tree_p * node_index + THRESH_IDX];
                // recompute the node_index
                if (sum < theta) 
                    node_index = left_son_index; // index of the left son                                       
                else 
                    node_index = left_son_index + 1; // index of the right son
            }
            // store only the current node index
            affiliation[affiliation_p * sample_index + tree_index] = node_index;
        }
    }
}

//-----------------------------------------------------------------------------
__global__ void icg_rf_oobe_kernel(
    float** trees, const size_t tree_p,
    int *affiliation, size_t affiliation_p, 
    int *prediction, size_t prediction_p, TrainParams tp)
{
    unsigned int sample_index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tree_index = blockIdx.y*blockDim.y + threadIdx.y;  

    if (sample_index < tp.num_samples && tree_index < tp.num_trees) {
        if (affiliation[affiliation_p * sample_index + tree_index] == -1) {
            float* tree = trees[tree_index];
            // set initial index in tree
            unsigned int node_index = 0;
            // go through whole tree
            for (unsigned char level = 0; level < tp.max_depth; level++) {
                int left_son_index = tree[tree_p * node_index + LSON_IDX];
                // break if terminal node has been found 
                if (left_son_index < 1)
                    break;
                // evaluate at this node
                float sum = 0;
                for (unsigned char feature = 0; feature < tp.num_features; feature++) {                    
                    int feature_index = (int) tree[tree_p * node_index + FEAT_IDX(feature)];
                    float wi = tree[tree_p * node_index + WEIGHT_IDX(feature)];
                    float xi = tex2D(input_tex, feature_index, sample_index);
                    sum += xi * wi;
                }
                float theta = tree[tree_p * node_index + THRESH_IDX];
                // recompute the node_index
                if (sum < theta) 
                    node_index = left_son_index; // index of the left son                                       
                else 
                    node_index = left_son_index + 1; // index of the right son
            }
            // store only the current node index
            prediction[prediction_p * sample_index + tree_index] = tree[tree_p * node_index + LABEL_IDX];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Evaluation Kernels
////////////////////////////////////////////////////////////////////////////////
texture<float, 2, cudaReadModeElementType> forest_tex;
texture<float, 2, cudaReadModeElementType> samples_tex;
texture<float, 2, cudaReadModeElementType> buffer_tex;

//-----------------------------------------------------------------------------
//DEPRECATED
/*__global__ void icg_rf_eval_hard_kernel(float* buffer, size_t buffer_p,
                                        TrainParams tp)
{
    unsigned int sample_index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tree_index = blockIdx.y*blockDim.y + threadIdx.y;

    if (sample_index < tp.num_samples && tree_index < tp.num_trees) {
        // calculate the numbers of elements in a tree
        unsigned int tree_offset = tree_index * tp.num_tree_cols;
        // set initial index in tree
        unsigned int node_index = 0;
        // go through whole tree
        for (int level = 0; level < tp.max_depth; level++) {
            // break if terminal node has been found
            if (tex2D(forest_tex, tree_offset, node_index))
                break;
            // evaluate at this node
            float sum = 0;
            for (int feature = 0; feature < tp.num_features; feature++) {
                int feature_index = (int) tex2D(forest_tex, tree_offset + 1 + feature * 2, node_index);
                float mi = tex2D(samples_tex, feature_index, sample_index);
                float wi = tex2D(forest_tex, tree_offset + 1 + feature * 2 + 1, node_index);
                sum += mi * wi;
            }
            float theta = tex2D(forest_tex, tree_offset + 1 + tp.num_features * 2, node_index);
            // recompute the node_index
            if (sum < theta) 
                node_index = node_index * 2 + 1; // index of the left son                                       
            else 
                node_index = node_index * 2 + 2; // index of the right son
        }

        // increase the voting for the predicted class
        int label = tex2D(forest_tex, tree_offset + tp.num_tree_cols - 1, node_index);  
        buffer[tree_index * buffer_p + sample_index] = label;    
    }
}
//-----------------------------------------------------------------------------
__global__ void icg_rf_hard_pred_kernel(float *confidences, size_t confidences_p,
                                        float *predictions, 
                                        TrainParams tp)
{
    unsigned int sample_index = blockIdx.x*blockDim.x + threadIdx.x; 

    if (sample_index < tp.num_samples) {
        int classes[10] = {0,0,0,0,0,0,0,0,0,0};
        // accumulate labels
        for (int tree = 0; tree < tp.num_trees; tree++){
            unsigned char label = tex2D(buffer_tex, sample_index, tree);
            classes[label]++;
        }
        // assign predictions and confidences
        float max_conf = 0;
        int max_class = 0;
        for (int label = 0; label < tp.num_classes; label++){
            float act_conf = (float)classes[label] / tp.num_trees;
            confidences[label * confidences_p + sample_index] = act_conf;
            if (act_conf > max_conf) {
                max_conf = act_conf;
                max_class = label;
            }
        }
        predictions[sample_index] = max_class;
    }  
}*/

//-----------------------------------------------------------------------------
__global__ void icg_rf_eval_soft_kernel(float *output, size_t output_p,
                                        TrainParams tp)
{
    unsigned int sample_index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tree_index = blockIdx.y*blockDim.y + threadIdx.y;  

    if (sample_index < tp.num_samples && tree_index < tp.num_trees) {  
        // calculate the numbers of elements in a tree
        unsigned int tree_offset = tree_index * tp.num_tree_cols;
        // set initial index in tree
        unsigned int node_index = 0;
        // go through whole tree
        for (int level = 0; level < tp.max_depth; level++) {
            int left_son_index = tex2D(forest_tex, tree_offset + LSON_IDX, node_index);
            // break if terminal node has been found
            if (left_son_index < 0)
                break;
            // evaluate at this node
            float sum = 0;
            for (int feature = 0; feature < tp.num_features; feature++) {
                int feature_index = (int) tex2D(forest_tex, tree_offset + FEAT_IDX(feature), node_index);
                float wi = tex2D(forest_tex, tree_offset + WEIGHT_IDX(feature), node_index);
                float mi = tex2D(samples_tex, feature_index, sample_index);
                sum += mi * wi;
            }
            float theta = tex2D(forest_tex, tree_offset + THRESH_IDX, node_index);
            // recompute the node_index
            if (sum < theta) 
                node_index = left_son_index; // index of the left son                                       
            else 
                node_index = left_son_index + 1; // index of the right son
        }

        // return the confidences for each class 
        for (int label = 0; label < tp.num_classes; label++) {
            float confidence = tex2D(forest_tex, tree_offset + CONF_IDX(label), node_index); 
            output[sample_index * output_p + tree_index * tp.num_classes + label] = confidence;        
        }
    }
}

//-----------------------------------------------------------------------------
__global__ void icg_rf_soft_pred_kernel(float *confidences, size_t confidences_p,
                                        float *predictions, 
                                        TrainParams tp)
{
    unsigned int sample_index = blockIdx.x*blockDim.x + threadIdx.x; 

    if (sample_index < tp.num_samples) {
        float max_confidence = 0;
        float max_label = 0;
        for (int label = 0; label < tp.num_classes; label++) {
            // sum over the confidences of all trees
            float sum = 0;
            for (int tree = 0; tree < tp.num_trees; tree++){
                sum += tex2D(buffer_tex, tree * tp.num_classes + label, sample_index);      
            }
            float confidence = sum / tp.num_trees;
            confidences[label * confidences_p + sample_index] = confidence;
            if (confidence > max_confidence){
                max_confidence = confidence;
                max_label = label;
            }
        }
        predictions[sample_index] = max_label;
    }  
}

#endif //ICG_RF_FOREST_KERNELS_CU_
