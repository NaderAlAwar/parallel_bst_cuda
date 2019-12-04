#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "bst.cuh"

__global__ void build_bst(int* device_array, int* device_bst, int array_size, int bst_size);

__global__ void bst_make(
    const int8_t* operations,
    const int32_t* values,
    const int input_size,
    const int individual_array_size,
    int32_t* output_in_order);

 // returns -1 if value is not found
__global__ void bst_search_for_value(int* device_bst, int value, int bst_size, int* value_index);
__global__ void bst_remove_value(int* device_bst, int value, int bst_size);
__global__ void bst_insert_value(int* device_bst, int value, int bst_size);

// fills the array subtree_indices with the indices of the subtree of with root at root_index 
// this is meant to be called from the device, but was marked global to use it for dynamic parallelism
__global__ void get_all_subtree_indices(int* subtree_indices, int root_index, int subtree_size);


// to prevent redundant code across kernels. Locates value if it exists in the bst
__device__ int locate_value_in_bst(int* device_bst, int value, int bst_size); 
__device__ int get_index_for_value(int* device_bst, int value, int root_index, int array_size);
__device__ int get_left_child_index(int index);
__device__ int get_right_child_index(int index);
__device__ int get_parent_index(int index);

// Gets the size of the subtree with root at subtree_root_index
__device__ int get_subtree_size(int subtree_root_index, int bst_size); 

// Removes a node with only one child (prevents redundant code in bst_remove_value)
__device__ void remove_value_single_child(int* device_bst, int value_index, int bst_size);


#endif
