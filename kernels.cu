
#include <stdio.h>
#include <cmath>

#include "constants.h"
#include "kernels.cuh"
#include "bst.cuh"
#include "kernel_wrappers.h"
#include "kernel_timer.h"

static void check_cuda_error(const cudaError_t e, const char *file, const int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s:%d: %s (%d)\n", file, line, cudaGetErrorString(e), e);
        exit(1);
    }
}

#define check_cuda(x) check_cuda_error(x, __FILE__, __LINE__)

__global__ void build_bst(int* device_array, int* device_bst, int array_size, int bst_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < array_size) {
        int root_index = 0; // the first value of device_array
        bool found_index = false; // indicates that the proper index of the value has not been found yet

        while (found_index == false) {
            int value = device_array[index];
            int device_bst_location = get_index_for_value(device_bst, value, root_index, bst_size);

            if (device_bst_location == -1) {
                return; // invalid index
            }

            // Check if index device_bst_location is already taken. If it is, atomicCAS will return -1, and set
            // the memory address to value. If not, atomicCAS  will return the value that is already there,
            // and the root index is set to device_bst_location
            if (atomicCAS(&device_bst[device_bst_location], -1, value) == -1) {
                found_index = true;
            } else {
                root_index = device_bst_location;
            }
        }
    }
}

__global__ void bst_make(
    const int8_t* operations,
    const int32_t* values,
    const int input_size,
    const int individual_array_size,
    int32_t* output_in_order
) {
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * individual_array_size;
    if (offset >= input_size) {
        return;
    }

    bst device_bst;
    const int upper_limit = offset + individual_array_size;
    for (int i = offset; i < upper_limit; i++) {
        if (operations[i] == 0) {
            device_bst.insert_value(values[i]);
        } else if (operations[i] == 1) {
            device_bst.remove_value(values[i]);
        }
    }

    // TODO: The output is being sorted here in order to check whether the test has passed.
    // Later on, this should be removed.
    int32_t* partial_output = new int32_t[individual_array_size]; // partial because it needs to be combined with the results of other threads
    device_bst.in_order(partial_output);

    for (int i = 0; i < device_bst.get_size(); i++) {
        output_in_order[offset + i] = partial_output[i];
    }
}

__device__ int get_index_for_value(int* device_bst, int value, int root_index, int bst_size) {
    int next_node = device_bst[root_index];
    int index = root_index;

    while (next_node != -1) {
        if (value < next_node && index < bst_size) {
            index = get_left_child_index(index); // go left
        } else if (value > next_node && index < bst_size) {
            index = get_right_child_index(index); // go right
        } else {
            return -1; // indicates that the value already exists or index is out of bounds
        }

        next_node = device_bst[index];
    }

    return index;
}

__global__ void bst_search_for_value(int* device_bst, int value, int bst_size, int* value_index) {
    if (value < 0) {
        *value_index = -1;
        return; // negative values are not allowed in the bst
    }

    *value_index = locate_value_in_bst(device_bst, value, bst_size);
}

__global__ void bst_remove_value(int* device_bst, int value, int bst_size) {
    if (value < 0) {
        return; // negative values are not allowed in the bst
    }
    int value_index = locate_value_in_bst(device_bst, value, bst_size);

    int left_child_index = get_left_child_index(value_index);
    int right_child_index = get_right_child_index(value_index);
    
    if (value_index == -1) { // value does not exist
        return;
    }
    // Check if the indices of the node are out of bounds or if the node is a leaf node.
    // If the index of the left child is out of bounds then the same can be assumed of the right child.
    else if (left_child_index >= bst_size || 
        (device_bst[left_child_index] == -1 && device_bst[right_child_index] == -1)) {
        device_bst[value_index] = -1; // equivalent to deleting
    } else if ((device_bst[left_child_index] != -1 && device_bst[right_child_index] == -1) ||
        (device_bst[left_child_index] == -1 && device_bst[right_child_index] != -1)) { // only one child
        remove_value_single_child(device_bst, value_index, bst_size);
    } else { // two children
        int successor_index = get_right_child_index(value_index);
        int next_index = get_left_child_index(successor_index);

        while (device_bst[next_index] != -1)  { // find the successor
            successor_index = next_index;
            next_index = get_left_child_index(next_index);
        }
        
        int successor_value = device_bst[successor_index];
        if (device_bst[get_left_child_index(successor_index)] == -1 && device_bst[get_right_child_index(successor_index)] == -1) { // no children
            device_bst[successor_index] = -1;
        } else {
            remove_value_single_child(device_bst, successor_index, bst_size); // Remove the successor from its old location.
        }

        device_bst[value_index] = successor_value; // replace the value to be removed with the sucessor
    }
}

__global__ void bst_insert_value(int* device_bst, int value, int bst_size) {
    int value_index = get_index_for_value(device_bst, value, 0, bst_size);

    if (value_index == -1) {
        return; // indicates that the value already exists or the index is out of bounds
    }

    device_bst[value_index] = value;
}

__device__ int locate_value_in_bst(int* device_bst, int value, int bst_size) {
    int next_node = device_bst[0];
    int index = 0;

    while (index < bst_size) {
        if (value < next_node) {
            index = get_left_child_index(index); // go left
        } else if (value > next_node) {
            index = get_right_child_index(index); // go right
        } else {
            return index; // the index of value has been found
        }

        next_node = device_bst[index];
    }

    return -1;
}

__device__ int get_left_child_index(int index) {
    return (2 * index) + 1;
}

__device__ int get_right_child_index(int index) {
    return (2 * index) + 2;
}

__device__ int get_parent_index(int index) {
    if (index % 2 == 1) { // odd indices are left children
        return (index - 1) / 2;
    } else if (index % 2 == 0) { // even indices are right children
        return (index - 2) / 2;
    } else { // all indices <= 0
        return -1;
    }
}

__device__ int get_subtree_size(int subtree_root_index, int bst_size) {
    int subtree_size = 1; // only root

    int right_child_index = get_right_child_index(subtree_root_index); // maximum possible index
    int current_level = 0; // helps determine the number of nodes at this level
    int current_level_size = 1; // only root

    while (right_child_index < bst_size) {
        current_level++; // indicates that the next level exists in the bst
        current_level_size = 2 * current_level_size; // each level contains twice the number of nodes of the previous level
        subtree_size += current_level_size;

        right_child_index = get_right_child_index(right_child_index);
    }

    return subtree_size;
}

__global__ void get_all_subtree_indices(int* subtree_indices, int root_index, int subtree_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < subtree_size && index > 0) {
        int level = static_cast<int>(log2(index + 1)) - 1; // TODO: could be more efficient to use a lookup table to map from index to level
        subtree_indices[index] = ((2 << level) * root_index) + index;
    }
}

__device__ void remove_value_single_child(int* device_bst, int value_index, int bst_size) {
    int left_child_index = get_left_child_index(value_index);
    int right_child_index = get_right_child_index(value_index);

    int subtree_root_index;

    if (device_bst[left_child_index] != -1) { // the left subtree is not empty
        subtree_root_index = left_child_index;
    } else { // the right subtree is not empty
        subtree_root_index = right_child_index;
    }

    int subtree_size = get_subtree_size(subtree_root_index, bst_size);

    int* subtree_indices = new int[subtree_size];
    subtree_indices[0] = subtree_root_index;

    // int block_size = 256;
    // int num_blocks = ((subtree_size - 1) + block_size - 1) / block_size;
    // get_all_subtree_indices<<<num_blocks, block_size>>>(subtree_indices, subtree_root_index, subtree_size);
    // cudaDeviceSynchronize();

    for (int i = 1; i < subtree_size; i++) {
        int level = static_cast<int>(log2(i + 1)) - 1; // TODO: could be more efficient to use a lookup table to map from index to level
        subtree_indices[i] = ((2 << level) * subtree_root_index) + i;
    }


    // TODO: these loops can be parallelized
    for (int i = 1; i < subtree_size; i = 2*i) { // iterate over each level of the subtree (each level is twice the size of the previous one)
        int first_index_lower_level = subtree_indices[i-1]; // this is the index of the first node at that level
        int first_index_upper_level = get_parent_index(first_index_lower_level); // this is the index of the first node at the level above

        for (int j = 0; j < i; j++) { // iterate over each node in each level
            device_bst[first_index_upper_level + j] = device_bst[first_index_lower_level + j];
        }
    }

    delete[] subtree_indices;
}

std::vector<int> build_bst_gpu(const std::vector<int>& host_vector, int bst_size) {
    cudaFree(0); // to establish cuda context

    int* host_array = new int[host_vector.size()];
    std::copy(host_vector.begin(), host_vector.end(), host_array);

    int* device_array;
    int* device_bst;
    int* host_bst = new int[bst_size];
    for (int i = 0; i < bst_size; i++) {
        host_bst[i] = -1;
    }

    cudaMallocManaged(&device_array, host_vector.size() * sizeof(int));
    cudaMallocManaged(&device_bst, bst_size * sizeof(int));

    cudaMemcpy(device_array, host_array, host_vector.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bst, host_bst, bst_size * sizeof(int), cudaMemcpyHostToDevice);

    // TODO: fix block_size and num_blocks. See bst_make_gpu below
    int block_size = 256;
    int num_blocks = ((host_vector.size() - 1) + block_size - 1) / block_size;
    build_bst<<<num_blocks, block_size>>>(device_array, device_bst, host_vector.size(), bst_size);

    cudaMemcpy(host_bst, device_bst, bst_size * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> result(bst_size);
    std::copy(host_bst, host_bst + bst_size, result.begin());

    delete[] host_bst;
    return result;
}

std::vector<int> bst_make_gpu(
    const std::vector<int8_t>& operations,
    const std::vector<int32_t>& values,
    int individual_array_size,
    kernel_timer& timer
){
    cudaFree(0); // to establish cuda context

    timer.start(); // time the setup phase
    int8_t* device_operations;
    int32_t* device_values;
    int32_t* device_output;
    int32_t* host_output = new int32_t[values.size()]; // values.size() is the size of the final output

    for (int i = 0; i < values.size(); i++) {
        host_output[i] = -1;
    }

    check_cuda(cudaMallocManaged(&device_operations, operations.size() * sizeof(int8_t)));
    check_cuda(cudaMemcpy(device_operations, operations.data(), operations.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

    check_cuda(cudaMallocManaged(&device_values, values.size() * sizeof(int32_t)));
    check_cuda(cudaMemcpy(device_values, values.data(), values.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    check_cuda(cudaMallocManaged(&device_output, values.size() * sizeof(int32_t)));
    check_cuda(cudaMemcpy(device_output, host_output, values.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    // operations.size() should always be divisible by individual_array_size
    int number_of_test_cases = operations.size() / individual_array_size;
    int number_of_blocks = std::ceil((float) number_of_test_cases / constants::MAX_THREADS);
    timer.stop();

    #ifdef DEBUG
    fprintf(stdout, "Number of blocks: %d\n", number_of_blocks);
    fprintf(stdout, "Number of tests: %d\n", number_of_test_cases);
    fprintf(stdout, "Max threads: %d\n", constants::MAX_THREADS);
    #endif

    // time the execution phase
    timer.start();
    bst_make<<<number_of_blocks, constants::MAX_THREADS>>>(device_operations,
        device_values,
        operations.size(),
        individual_array_size,
        device_output);
    // make sure that the kernel has finished before stopping the timer
    cudaDeviceSynchronize();
    timer.stop();

    // time the result phase
    timer.start();
    check_cuda(cudaMemcpy(host_output, device_output, values.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    timer.stop();

    std::vector<int32_t> result(values.size());
    std::copy(host_output, host_output + values.size(), result.begin());

    delete[] host_output;
    return result;
}
