#ifndef ICG_RF_FOREST_CU_
#define ICG_RF_FOREST_CU_

#include "icg_rf_forest_kernels.cu"
#include "icg_rf_rng.cu"

#include "cudatemplates/hostmemoryheap.hpp" 

//#define DEBUG 

//-----------------------------------------------------------------------------
template <typename T>
void printMemUsage(const Cuda::DeviceMemoryLinear<T, 1> &mat, const char *name = ""){
    printf("DeviceMemoryPitched %s (%d x 1) needs %d KB of device memory\n", name, mat.size[0], mat.size[0] * sizeof(T)/1024);
}
template <typename T>
void printMemUsage(const Cuda::DeviceMemoryPitched<T, 2> &mat, const char *name = ""){
    printf("DeviceMemoryPitched %s (%d x %d) needs %d KB of device memory\n", name, mat.size[0], mat.size[1], mat.stride[0] * mat.size[1] * sizeof(T)/1024);
}
template <typename T>
void printMemUsage(const Cuda::Array<T, 1> &mat, const char *name = ""){
    printf("Array %s (%d x %d) needs %d KB of device memory\n", name, mat.size[0], mat.size[1], mat.size[0] * mat.size[1] * sizeof(T)/1024);
}
template <typename T>
void printMemUsage(const Cuda::Array<T, 2> &mat, const char *name = ""){
    printf("Array %s (%d x %d) needs %d KB of device memory\n", name, mat.size[0], mat.size[1], mat.size[0] * mat.size[1] * sizeof(T)/1024);
}


//-----------------------------------------------------------------------------
inline bool printToScreen(const Cuda::HostMemoryHeap<float, 1> &data, const char *name = ""){
    printf("\nprinting Matrix: %s (%d)\n", name, (int)data.size[0]);
    for (unsigned int x = 0; x < data.size[0]; x++)
        printf("%3.3f ", data.getBuffer()[x]);
    printf("\n");
    return true;
}
bool printToScreen(const Cuda::HostMemoryHeap<float, 2> &data, const char *name = ""){
    printf("\nprinting Matrix: %s (%d x %d)\n", name, (int)data.size[0], (int)data.size[1]);
    for (unsigned int y = 0; y < data.size[1]; y++){
        for (unsigned int x = 0; x < data.size[0]; x++){
            printf("%3.3f ", data.getBuffer()[y * data.stride[0] + x]);
        }
        printf("\n");
    }
    return true;
} 
bool printToScreen(const Cuda::HostMemoryHeap<int, 2> &data, const char *name = ""){
    printf("\nprinting Matrix: %s (%d x %d)\n", name, (int)data.size[0], (int)data.size[1]);
    for (unsigned int y = 0; y < data.size[1]; y++){
        for (unsigned int x = 0; x < data.size[0]; x++){
            printf("%d ", data.getBuffer()[y * data.stride[0] + x]);
        }
        printf("\n");
    }
    return true;
} 
bool printToScreen(const Cuda::HostMemoryHeap<unsigned char, 2> &data, const char *name = ""){
    printf("\nprinting Matrix: %s (%d x %d)\n", name, (int)data.size[0], (int)data.size[1]);
    for (unsigned int y = 0; y < data.size[1]; y++){
        for (unsigned int x = 0; x < data.size[0]; x++){
            printf("%d ", data.getBuffer()[y * data.stride[0] + x]);
        }
        printf("\n");
    }
    return true;
} 
bool printToScreen(const Cuda::DeviceMemoryPitched<int, 2> &data, const char *name = ""){
    Cuda::HostMemoryHeap<int, 2> data_copy(data);
    return printToScreen(data_copy, name);
}
bool printToScreen(const Cuda::DeviceMemoryPitched<unsigned char, 2> &data, const char *name = ""){
    Cuda::HostMemoryHeap<unsigned char, 2> data_copy(data);
    return printToScreen(data_copy, name);
}
bool printToScreen(const Cuda::DeviceMemoryPitched<float, 2> &data, const char *name = ""){
    Cuda::HostMemoryHeap<float, 2> data_copy(data);
    return printToScreen(data_copy, name);
}
//-----------------------------------------------------------------------------
bool printToScreen(const float* devptr, const int width, const int height, const int stride, const char *name = ""){
    Cuda::HostMemoryHeap<float,2> data(Cuda::Size<2>(width, height));
    //printf("\ncopying %d bytes in %d rows, stride = %d\n", width * sizeof(float), height, stride * sizeof(float));
    cudaMemcpy2D(data.getBuffer(), width * sizeof(float),
        devptr, stride * sizeof(float),
        width * sizeof(float), height,
        cudaMemcpyDeviceToHost);
    ICG_RF_CHECK_CUDA_ERROR();
    return printToScreen(data, name);
}

//-----------------------------------------------------------------------------
template <typename T>
inline bool cudaInit(Cuda::DeviceMemoryPitched<T,2>* data, T value){
    int bs = 16; 
    icg_rf_init_kernel<<<dim3(divUp(data->size[0], bs), divUp(data->size[1], bs),1), dim3(bs, bs, 1)>>>(
        data->getBuffer(),data->size[0], data->size[1],data->stride[0], value);
    ICG_RF_CHECK_CUDA_ERROR();
    return true;
}
//-----------------------------------------------------------------------------
template <typename T>
inline bool cudaInit(Cuda::HostMemoryHeap<T,2>* data, T value){
    for (size_t xy = 0; xy < data->size[0] * data->size[1]; xy++)
        data->getBuffer()[xy] = value;
    return true;
}

//-----------------------------------------------------------------------------
template <typename T>
inline T cudaGet(const Cuda::HostMemoryHeap<T,2> data, int x, int y){
#ifdef DEBUG
    if (x < 0 || x >= data.size[0] || y < 0 || y >= data.size[1])
        printf("\n !!! trying to access %d|%d in %d|%d\n", x,y,data.size[0], data.size[1]);
#endif
    return data.getBuffer()[y * data.stride[0] + x];          
}

//-----------------------------------------------------------------------------
template <typename T>
inline void cudaSet(Cuda::HostMemoryHeap<T,2> *data, int x, int y, T value){
#ifdef DEBUG
    if (x < 0 || x >= data->size[0] || y < 0 || y >= data->size[1])
        printf("\n !!! trying to write to %d|%d in %d|%d\n", x,y,data->size[0], data->size[1]);
#endif
    data->getBuffer()[y * data->stride[0] + x] = value;
}


//-----------------------------------------------------------------------------
bool cudaTrainForest(const Cuda::Array<float,2>& input,
                     const Cuda::DeviceMemoryLinear<int,1>& labels,
                     const Cuda::DeviceMemoryLinear<float,1>& weights,
                     Cuda::DeviceMemoryPitched<float,2>** forest,
                     float* oobe,
                     const int num_samples, const int num_classes, 
                     const int num_trees, const int max_depth, const int num_tree_cols,
                     const int num_hypotheses, const int num_features,
                     const float bag_ratio)
{ 
    // new random seed...
#ifdef WIN32
    unsigned int seedNum = (unsigned int) time(NULL);
    //    srand(seedNum); //FIXXME, only for reproducability
#else
    unsigned int seedNum;
    struct timeval TV;
    unsigned int curTime;
    gettimeofday(&TV, NULL);
    curTime = (unsigned int) TV.tv_usec;
    seedNum = (unsigned int) time(NULL) + curTime + getpid() + getDevRandom();
    srand(seedNum);        
#endif

    // Copy Training Parameters
    TrainParams tp;
    tp.num_trees = num_trees;
    tp.tree_level = 0;
    tp.max_depth = max_depth;
    tp.num_samples = num_samples;
    tp.num_features = num_features;
    tp.num_classes = num_classes;
    tp.num_hyp = num_hypotheses;
    tp.num_hyp_2 = num_hypotheses * num_hypotheses;
    tp.num_tree_cols = num_tree_cols;

    // Init textures
    cudaChannelFormatDesc channelDescInt = cudaCreateChannelDesc<int>();
    cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();

    input_tex.filterMode = cudaFilterModePoint;
    input_tex.normalized = false;
    cudaBindTextureToArray(input_tex, input.getArray(), channelDescFloat);

    cudaBindTexture(0, labels_tex, labels.getBuffer(), labels.size[0] * sizeof(int));

    r_weights_tex.filterMode = cudaFilterModePoint;
    r_weights_tex.normalized = false;

    r_features_tex.filterMode = cudaFilterModePoint;
    r_features_tex.normalized = false;

    // perform bagging
    int num_bag_samples = (int)(input.size[1] * bag_ratio);
    Cuda::DeviceMemoryPitched<int,2>bags_dmp(Cuda::Size<2>(num_trees, num_bag_samples));
    cudaRandomNumbers(&bags_dmp, 0, labels.size[0]); 

    // structure to store the node index in every tree that each sample is voting for currently
    Cuda::DeviceMemoryPitched<int,2> affiliation_dmp(Cuda::Size<2>(num_trees, num_samples));
    cudaInit(&affiliation_dmp, -1);

    // store 0 in affiliation, when sample is in bag
    int bs = 16;
    icg_rf_bagging_kernel<<<dim3(divUp(num_trees, bs), divUp(num_bag_samples, bs),1), dim3(bs, bs, 1)>>>(
        bags_dmp.getBuffer(),bags_dmp.stride[0],
        affiliation_dmp.getBuffer(), affiliation_dmp.stride[0],
        num_trees, num_bag_samples); 

    // Init trees 
    Cuda::HostMemoryHeap<float*, 2> tree_adress_hmh(Cuda::Size<2>(num_trees,1));
    size_t single_tree_pitch;
    for (int tree = 0; tree < num_trees; tree++) {
        cudaMallocPitch((void**)&tree_adress_hmh.getBuffer()[tree], 
            &single_tree_pitch, num_tree_cols*sizeof(float), 1);        
        ICG_RF_CHECK_CUDA_ERROR();
        // init mem with zeros
        cudaMemset2D(tree_adress_hmh.getBuffer()[tree], single_tree_pitch, 0, num_tree_cols*sizeof(float), 1);
        ICG_RF_CHECK_CUDA_ERROR();
    }
    single_tree_pitch /= sizeof(float);
    // Copy tree adresses to device
    Cuda::DeviceMemoryPitched<float*, 2> tree_adress_dmp(tree_adress_hmh);

    // Init structure that stores the number nodes in each level of a tree
    Cuda::DeviceMemoryPitched<int,2> num_nodes_per_level_dmp(Cuda::Size<2>(num_trees,max_depth + 1)); 
    cudaInit(&num_nodes_per_level_dmp, 1);
    Cuda::HostMemoryHeap<int,2> nodes_per_level_hmh(num_nodes_per_level_dmp);

    // store the maximum number of new nodes over all trees
    int num_leaf_nodes_max = 1;

    for (int tree_level = 0; tree_level <= max_depth; tree_level++) {
        tp.tree_level = tree_level;

        // classify nodes to leafs and splits  
        Cuda::DeviceMemoryPitched<int,2>node_type_dmp(Cuda::Size<2>(num_trees, num_leaf_nodes_max));
        if (num_leaf_nodes_max) {
            cudaInit(&node_type_dmp, -1);
            bs = 8;
            icg_rf_classify_leaf_nodes_kernel<<<dim3(divUp(num_trees, bs), divUp(num_leaf_nodes_max, bs),1), dim3(bs,bs,1)>>>(
                num_nodes_per_level_dmp.getBuffer(), num_nodes_per_level_dmp.stride[0],
                affiliation_dmp.getBuffer(), affiliation_dmp.stride[0],
                node_type_dmp.getBuffer(), node_type_dmp.stride[0],tp);
            ICG_RF_CHECK_CUDA_ERROR();
        }

        // find out the number of leaf and split nodes per tree and in total
        Cuda::HostMemoryHeap<int,2>node_type_hmh(node_type_dmp);
        Cuda::HostMemoryHeap<int,2>num_splits_hmh(Cuda::Size<2>(num_trees,1));
        Cuda::HostMemoryHeap<int,2>num_leafs_hmh(Cuda::Size<2>(num_trees,1));
        cudaInit(&num_splits_hmh, 0);
        cudaInit(&num_leafs_hmh, 0);
        int num_splits = 0;
        int num_leafs = 0;
        for (int tree = 0; tree < num_trees; tree++) {
            for (int node = 0; node < num_leaf_nodes_max; node++){
                if (cudaGet(node_type_hmh, tree, node) == 1){
                    num_leafs_hmh.getBuffer()[tree]++;
                    num_leafs++;
                }
                if (cudaGet(node_type_hmh, tree, node) == 2){
                    num_splits_hmh.getBuffer()[tree]++;
                    num_splits++;
                }
            }
        }

        // create subsets for the split and leaf nodes
        Cuda::HostMemoryHeap<int,2>split_subset_hmh(Cuda::Size<2>(num_splits, 3));
        Cuda::HostMemoryHeap<int,2>leaf_subset_hmh(Cuda::Size<2>(num_leafs, 2));
        cudaInit(&split_subset_hmh, 0);
        cudaInit(&leaf_subset_hmh, 0);
        int split_index = 0;
        int leaf_index = 0;
        for (int tree = 0; tree < num_trees; tree++) {
            int num_leaf_nodes = 1;
            int leaf_row_offset = 0;
            int left_son = cudaGet(nodes_per_level_hmh, tree, tree_level); 
            if (tree_level > 0) {
                leaf_row_offset = cudaGet(nodes_per_level_hmh, tree, tree_level-1);
                num_leaf_nodes = cudaGet(nodes_per_level_hmh, tree, tree_level)- leaf_row_offset; 
            }
            for (int node = 0; node < num_leaf_nodes; node++){
                if (cudaGet(node_type_hmh, tree, node) == 1){
                    cudaSet(&leaf_subset_hmh, leaf_index, 0, tree);
                    cudaSet(&leaf_subset_hmh, leaf_index++, 1, node + leaf_row_offset);
                }
                if (cudaGet(node_type_hmh, tree, node) == 2){
                    cudaSet(&split_subset_hmh, split_index, 0, tree);
                    cudaSet(&split_subset_hmh, split_index, 1, node + leaf_row_offset);
                    cudaSet(&split_subset_hmh, split_index++, 2, left_son);
                    left_son += 2;
                }
            }
        }
        Cuda::DeviceMemoryPitched<int,2>split_subset_dmp(split_subset_hmh);
        Cuda::DeviceMemoryPitched<int,2>leaf_subset_dmp(leaf_subset_hmh);

        // train the split nodes
        if (num_splits) {
            // create random numbers
            Cuda::DeviceMemoryPitched<int,2> rand_features_dmp(Cuda::Size<2>(num_splits, num_features));
            Cuda::DeviceMemoryPitched<float,2> rand_weights_dmp(Cuda::Size<2>(num_splits, num_features * num_hypotheses)); 
            cudaRandomNumbers(&rand_features_dmp, 0, input.size[0] - 1);
            cudaRandomNumbers(&rand_weights_dmp);

            // copy them to arrays            
            Cuda::Array<float,2> rand_weights_arr(rand_weights_dmp);
            Cuda::Array<int,2> rand_features_arr(rand_features_dmp);
            cudaBindTextureToArray(r_weights_tex, rand_weights_arr.getArray(), channelDescFloat);
            cudaBindTextureToArray(r_features_tex, rand_features_arr.getArray(), channelDescInt);
            ICG_RF_CHECK_CUDA_ERROR(); 

            // assign the features to the trees
            bs = 8;
            icg_rf_split_assign_feature_kernel<<<dim3(divUp(num_splits, bs), divUp(num_features, bs), 1), dim3(bs,bs,1)>>>(
                tree_adress_dmp.getBuffer(), single_tree_pitch,
                split_subset_dmp.getBuffer(), num_splits, split_subset_dmp.stride[0], tp);
            ICG_RF_CHECK_CUDA_ERROR();  

            // create buffer structures for training
            Cuda::DeviceMemoryPitched<float,2> thresh_limits_dmp(Cuda::Size<2>(num_splits, num_hypotheses * 2));
            Cuda::DeviceMemoryPitched<float,2> pk_l_dmp(Cuda::Size<2>(num_splits, tp.num_hyp_2 * num_classes));
            Cuda::DeviceMemoryPitched<float,2> pk_r_dmp(Cuda::Size<2>(num_splits, tp.num_hyp_2 * num_classes));
            cudaInit(&pk_l_dmp, 0.0f);
            cudaInit(&pk_r_dmp, 0.0f);

            // compute the minimum and maximum response for each node
            bs = 8;
            icg_rf_split_compute_thresh_kernel<<<dim3(divUp(num_splits, bs), divUp(num_hypotheses, bs), 1), dim3(bs,bs,1)>>>(
                split_subset_dmp.getBuffer(), num_splits, split_subset_dmp.stride[0],
                affiliation_dmp.getBuffer(), affiliation_dmp.stride[0],
                thresh_limits_dmp.getBuffer(), thresh_limits_dmp.stride[0],tp);

            // evaluate all hypotheses
            bs = 8;   
            icg_rf_split_eval_thresh_kernel<<<dim3(divUp(num_splits, bs), divUp(num_hypotheses, bs), 1), dim3(bs, bs, 1)>>>(
                split_subset_dmp.getBuffer(), num_splits, split_subset_dmp.stride[0],
                weights.getBuffer(), affiliation_dmp.getBuffer(), affiliation_dmp.stride[0],
                thresh_limits_dmp.getBuffer(), thresh_limits_dmp.stride[0],
                pk_l_dmp.getBuffer(), pk_l_dmp.stride[0], pk_r_dmp.getBuffer(), pk_r_dmp.stride[0],tp);
            ICG_RF_CHECK_CUDA_ERROR();

            // select best hypothesis
            bs = 64;
            icg_rf_split_select_best_kernel<<<dim3(divUp(num_splits, bs), 1, 1), dim3(bs, 1, 1)>>>(
                tree_adress_dmp.getBuffer(), single_tree_pitch,
                split_subset_dmp.getBuffer(), num_splits, split_subset_dmp.stride[0],
                thresh_limits_dmp.getBuffer(), thresh_limits_dmp.stride[0],
                pk_l_dmp.getBuffer(), pk_l_dmp.stride[0], pk_r_dmp.getBuffer(), pk_r_dmp.stride[0], tp);
            ICG_RF_CHECK_CUDA_ERROR();

            cudaUnbindTexture(r_weights_tex);
            cudaUnbindTexture(r_features_tex);
        }
        // train leaf nodes
        if (num_leafs) {
            bs = 32;
            size_t shared_mem = bs * num_classes * sizeof(float);
            if (shared_mem > 16000)
                ICG_RF_THROW_ERROR("icg_rf_train_leaf_node_kernel: too much shared memory needed...");
            icg_rf_train_leaf_node_kernel<<<dim3(divUp(num_leafs, bs), 1, 1), dim3(bs, 1, 1), shared_mem>>>(
                tree_adress_dmp.getBuffer(), single_tree_pitch, weights.getBuffer(),
                leaf_subset_dmp.getBuffer(), num_leafs, leaf_subset_dmp.stride[0],
                affiliation_dmp.getBuffer(), affiliation_dmp.stride[0], tp);
            ICG_RF_CHECK_CUDA_ERROR(); 
        }  

        // update trees for the next iteration
        num_leaf_nodes_max = 0;
        if (tree_level < max_depth) {
            // reset the number of rows per tree
            for (int tree = 0; tree < num_trees; tree++) {
                int num_new_rows = num_splits_hmh.getBuffer()[tree] * 2;
                cudaSet(&nodes_per_level_hmh, tree, tree_level+1, cudaGet(nodes_per_level_hmh, tree, tree_level) + num_new_rows);
                if (num_new_rows > num_leaf_nodes_max)
                    num_leaf_nodes_max = num_new_rows; 
            }
            Cuda::copy(num_nodes_per_level_dmp, nodes_per_level_hmh);   

            // create new trees and copy the old ones        
            for (int tree = 0; tree < num_trees; tree++) {
                int num_new_rows = num_splits_hmh.getBuffer()[tree] * 2;  
                if (num_new_rows) {
                    float* new_tree;
                    size_t new_pitch;
                    cudaMallocPitch((void**)&new_tree, &new_pitch, num_tree_cols*sizeof(float), 
                        cudaGet(nodes_per_level_hmh, tree, tree_level+1));        
                    // init mem with zeros
                    cudaMemset2D(new_tree, new_pitch, 0, num_tree_cols*sizeof(float), 
                        cudaGet(nodes_per_level_hmh, tree, tree_level+1));
                    // copy old tree inside
                    cudaMemcpy2D(new_tree, new_pitch, 
                        tree_adress_hmh.getBuffer()[tree], single_tree_pitch*sizeof(float),
                        num_tree_cols*sizeof(float), cudaGet(nodes_per_level_hmh, tree, tree_level), 
                        cudaMemcpyDeviceToDevice);
                    ICG_RF_CHECK_CUDA_ERROR();
                    cudaFree(tree_adress_hmh.getBuffer()[tree]);
                    tree_adress_hmh.getBuffer()[tree] = new_tree;
                }
            }
            Cuda::copy(tree_adress_dmp, tree_adress_hmh);
        }

        // update affiliation (node index that each sample gets stuck to)
        bs = 8;
        icg_rf_eval_affinity_kernel<<<dim3(divUp(num_samples, bs), divUp(num_trees, bs),1), dim3(bs, bs, 1)>>>( 
            tree_adress_dmp.getBuffer(), single_tree_pitch,
            affiliation_dmp.getBuffer(), affiliation_dmp.stride[0],tp);
        ICG_RF_CHECK_CUDA_ERROR();
    } 

    // find out the maximum number of rows in the forest
    int max_rows = 0;
    for (int tree = 0; tree < num_trees; tree++) {
        int num_rows = cudaGet(nodes_per_level_hmh, tree, max_depth);
        if (num_rows > max_rows)
            max_rows = num_rows;
    }
    // create forest structure
    *forest = new Cuda::DeviceMemoryPitched<float,2>(Cuda::Size<2>(num_tree_cols * num_trees, max_rows));
    ICG_RF_CHECK_CUDA_ERROR();

    // compute oobe
    /*Cuda::DeviceMemoryPitched<int, 2> oobe_prediction_dmp(Cuda::Size<2>(num_trees, num_samples));
    cudaInit(&oobe_prediction_dmp, -1);
    ICG_RF_CHECK_CUDA_ERROR();
    bs = 8;
    icg_rf_oobe_kernel<<<dim3(divUp(num_samples, bs), divUp(num_trees, bs),1), dim3(bs, bs, 1)>>>( 
        tree_adress_dmp.getBuffer(), single_tree_pitch,
        affiliation_dmp.getBuffer(), affiliation_dmp.stride[0],
        oobe_prediction_dmp.getBuffer(), oobe_prediction_dmp.stride[0], tp);
    ICG_RF_CHECK_CUDA_ERROR();
    Cuda::HostMemoryHeap<int,2> oobe_prediction_hmh(oobe_prediction_dmp);
    Cuda::HostMemoryHeap<int,1> labels_hmh(labels);
    int num_oob_samples = 0;
    int num_oob_valids = 0;
    for (int tree = 0; tree < num_trees; tree++) {
        for (int sample = 0; sample < num_samples; sample++) {
            int prediction = cudaGet(oobe_prediction_hmh, tree, sample);
            if (prediction != -1){
                num_oob_samples++;
                if (prediction == labels_hmh.getBuffer()[sample])
                    num_oob_valids++;
            }
        }
    }
    (*oobe) = 1.0f - (float) num_oob_valids / num_oob_samples;*/

 
    // copy trees to forest and free ressources
    for (int tree = 0; tree < num_trees; tree++) {
        float *forest_buffer = &(*forest)->getBuffer()[tree * num_tree_cols];
        cudaMemcpy2D(forest_buffer, (*forest)->stride[0] * sizeof(float), 
            tree_adress_hmh.getBuffer()[tree], single_tree_pitch * sizeof(float),
            num_tree_cols * sizeof(float), cudaGet(nodes_per_level_hmh, tree, max_depth),
            cudaMemcpyDeviceToDevice);
        cudaFree(tree_adress_hmh.getBuffer()[tree]);
        ICG_RF_CHECK_CUDA_ERROR();
    }  

    cudaUnbindTexture(input_tex);
    cudaUnbindTexture(labels_tex);
    return true;
}


//-----------------------------------------------------------------------------
//! Evaluate the result of a random forest for a given data input. 
//! @param[in] forest The random forest to evaluate for. Its structure:
//!
//!      tree_idx | node_idx | is_terminal | m_i | w_i | theta | conf_j | label
//!
//!       tree_idx        Index of the tree in the forest, starting with '0'
//!       node_idx        Index of the node within a tree, starting with '0' 
//!       is_terminal     Specifies (0/1) whether a node or its parent is terminal
//!       m_i,w_i, theta  Specify the node hypotheses: m_1*w_1 + m_2* w_2 ... > theta
//!       conf_j          The confidence for the jth class in this node
//!       label           The label of the class with the highest confidence
//!
//! @param[in] input The input data to be classified (rows: datapoints, cols: features)
//! @param[out] confidences Confidences for each class, sized [num_classes x num_samples]
//! @param[out] predictions Class index with highest confidence, sized [1 x num_samples]
//! @param[in] num_classes The number of different classes (j)
//! @param[in] num_trees The number of trees in the forest 
//! @param[in] max_depth The depth of each tree in the forest
//! @param[in] num_features The number of features from which a hypothesis is generated (i)
//! @param[in] use_soft_voting If true, the confidences of all trees are used to predict labels.
//!            Otherwise, only the predictions of all trees are taken into account 
//! @return True on successful termination
bool cudaEvaluateForest(const Cuda::Array<float,2>& forest, 
                        const Cuda::Array<float,2>& input,
                        Cuda::DeviceMemoryPitched<float,2>* confidences,
                        Cuda::DeviceMemoryLinear<float,1>* predictions,
                        const int num_samples, const int num_trees, const int max_depth, const int num_tree_cols,
                        const int num_classes, const int num_features, const bool use_soft_voting)
{    
    if (num_samples != input.size[1])
        ICG_RF_THROW_ERROR("The input dimensions do not fit"); 

    if (confidences->size[0] != num_samples || confidences->size[1] != num_classes) 
        ICG_RF_THROW_ERROR("The confidences output dimensions do not fit"); 

    if (predictions->size[0] != num_samples) 
        ICG_RF_THROW_ERROR("The predictions output dimensions do not fit");

    TrainParams tp;
    tp.num_trees = num_trees;
    tp.tree_level = 0;
    tp.max_depth = max_depth;
    tp.num_samples = num_samples;
    tp.num_features = num_features;
    tp.num_classes = num_classes;
    tp.num_tree_cols = num_tree_cols;

    // Init forest texture
    forest_tex.filterMode = cudaFilterModePoint;
    forest_tex.normalized = false;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaBindTextureToArray(forest_tex, forest.getArray(), channelDesc);

    // Init samples texture
    samples_tex.filterMode = cudaFilterModePoint;
    samples_tex.normalized = false;
    cudaBindTextureToArray(samples_tex, input.getArray(), channelDesc);

    if (use_soft_voting) {
        int eval_block_size = 16;
        dim3 dim_block_eval(eval_block_size, eval_block_size, 1);
        dim3 dim_grid_eval(divUp(num_samples, eval_block_size), divUp(num_trees, eval_block_size),1);

        // compute confidences
        Cuda::DeviceMemoryPitched<float,2> buffer(Cuda::Size<2>(num_trees * num_classes, num_samples));
        icg_rf_eval_soft_kernel<<<dim_grid_eval, dim_block_eval>>>(            
            buffer.getBuffer(), buffer.stride[0], tp);
        ICG_RF_CHECK_CUDA_ERROR();

        buffer_tex.filterMode = cudaFilterModePoint;
        buffer_tex.normalized = false;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

        // mean over confidences and compute predictions
        int pred_block_size = 128;
        dim3 dim_block_pred(pred_block_size, 1, 1);
        dim3 dim_grid_pred(divUp(num_samples, pred_block_size), 1, 1);

        Cuda::Array<float,2> buffer_arr(buffer); 
        cudaBindTextureToArray(buffer_tex, buffer_arr.getArray(), channelDesc);       

        icg_rf_soft_pred_kernel<<<dim_grid_pred, dim_block_pred>>>(
            confidences->getBuffer(), confidences->stride[0],
            predictions->getBuffer(), tp);
        ICG_RF_CHECK_CUDA_ERROR();        

        cudaUnbindTexture(buffer_tex);              
        cudaThreadSynchronize();      
    }
    else { // use hard voting
        // DEPRECATED
        ICG_RF_THROW_ERROR("Hard evaluation is deprecated. Call Jakob\n");  
        /*
        if (num_classes > 10) 
        ICG_RF_THROW_ERROR("The maximum number of classes is exceeded");    
        // get votings 
        int eval_block_size = 16; 
        dim3 dim_block_eval(eval_block_size, eval_block_size, 1);
        dim3 dim_grid_eval(divUp(num_samples, eval_block_size), divUp(num_trees, eval_block_size), 1);

        Cuda::DeviceMemoryPitched<float,2> buffer(Cuda::Size<2>(num_samples, num_trees));

        icg_rf_eval_hard_kernel<<<dim_grid_eval, dim_block_eval>>>(            
        buffer.getBuffer(), buffer.stride[0], tp);
        ICG_RF_CHECK_CUDA_ERROR();

        // mean over votings and compute predictions
        int pred_block_size = 64; 
        dim3 dim_block_pred(pred_block_size, 1, 1);
        dim3 dim_grid_pred(divUp(num_samples, pred_block_size), 1, 1);

        Cuda::Array<float,2> buffer_arr(buffer); 
        cudaBindTextureToArray(buffer_tex, buffer_arr.getArray(), channelDesc);      

        icg_rf_hard_pred_kernel<<<dim_grid_pred, dim_block_pred>>>(
        confidences->getBuffer(), confidences->stride[0],
        predictions->getBuffer(), tp);
        ICG_RF_CHECK_CUDA_ERROR();

        cudaUnbindTexture(buffer_tex);              
        cudaThreadSynchronize();  
        */
    }
    cudaUnbindTexture(forest_tex);
    cudaUnbindTexture(samples_tex);
    return true;
}

#endif //ICG_RF_FOREST_CU_
