#ifndef ICG_RF_RNG_CU
#define ICG_RF_RNG_CU

///////////////////////////////////////////////////////////////////////////////
//! RNG KERNELS
////////////////////////////////////////////////////////////////////////////////

/*
LICENSE: this code is subject to the license listed at
http://www.amolf.nl/~vanmeel/mdgpu/download.html
Among other restrictions, this code is released under the GNU Public License (GPL).

Authors:
A. Arnold (original)
Kipton Barros (modifications)

----

Generate pseudo-random numbers using a linear congruential generator. The generated
random numbers are identical to those produced by the lrand48() provided by the
C standard library.

Usage:

// From host, allocate the Rand48 structure, pass it to CUDA, and release it.
// The random sequence is persistent across CUDA kernel calls.

void hostFunction() {
rng = new Rand48();
rng->init(GRID_DIM*BLOCK_DIM, SEED);
cudaFunction1 <<<GRID_DIM, BLOCK_DIM, sharedMem>>> (*rng);
cudaFunction2 <<<GRID_DIM, BLOCK_DIM, sharedMem>>> (*rng);
rng->destroy();
delete rng;
}

// From CUDA, load the random state from device memory into local registers,
// generate random numbers, and finally store state back to device memory.
// Note that the random state, rng, is stored in registers, and is being updated
// with each device call.

__global__ void cudaFunction1(Rand48 rng) {
rand48_loadState(rng);
...
rand48_nextInt(rng);
...
rand48_storeState(rng);
}

*/

struct Rand48 {
    // strided iteration constants (48-bit, distributed on 2x 24-bit)
    uint2 A, C;
    // CUDA array -- random numbers for all threads
    uint2 *state;
    // random number for a single thread (used by CUDA device functions only)
    uint2 state0;

    // magic constants for rand48
    static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;

    void init(unsigned int nThreads, int seed) {
        uint2* seeds = new uint2[ nThreads ];

        cudaMalloc((void**) &state, sizeof(uint2)*nThreads);

        // calculate strided iteration constants
        unsigned long long A, C;
        A = 1LL; C = 0LL;
        for (unsigned int i = 0; i < nThreads; ++i) {
            C += A*c;
            A *= a;
        }
        this->A.x = (unsigned int) (A & 0xFFFFFFLL);
        this->A.y = (unsigned int) ((A >> 24) & 0xFFFFFFLL);
        this->C.x = (unsigned int) (C & 0xFFFFFFLL);
        this->C.y = (unsigned int) ((C >> 24) & 0xFFFFFFLL);

        // prepare first nThreads random numbers from seed
        unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
        for (unsigned int i = 0; i < nThreads; ++i) {
            x = a*x + c;
            seeds[i].x = (unsigned int) (x & 0xFFFFFFLL);
            seeds[i].y = (unsigned int) ((x >> 24) & 0xFFFFFFLL);
        }

        cudaMemcpy(state, seeds, sizeof(uint2)*nThreads, cudaMemcpyHostToDevice);

        delete[] seeds;
    }

    void destroy() {
        cudaFree((void*) state);
    }
};

__device__ inline void rand48_loadState(Rand48 &r) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    r.state0 = r.state[i];
}

__device__ inline void rand48_storeState(Rand48 &r) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    r.state[i] = r.state0;
}

__device__ inline void rand48_iterate(Rand48 &r) {
    // state0 is 2x 24bit to handle overflows optimally, i.e.
    // in one operation.

    // the multiplication commands however give the low and hi 32 bit,
    // which have to be converted as follows:
    // 48bit in bytes = ABCD EF (space marks 32bit boundary)
    // R0             = ABC
    // R1             =    D EF

    unsigned int R0, R1;

    // low 24-bit multiplication
    const unsigned int lo00 = __umul24(r.state0.x, r.A.x);
    const unsigned int hi00 = __umulhi(r.state0.x, r.A.x);

    // 24bit distribution of 32bit multiplication results
    R0 = (lo00 & 0xFFFFFF);
    R1 = (lo00 >> 24) | (hi00 << 8);

    R0 += r.C.x; R1 += r.C.y;

    // transfer overflows
    R1 += (R0 >> 24);
    R0 &= 0xFFFFFF;

    // cross-terms, low/hi 24-bit multiplication
    R1 += __umul24(r.state0.y, r.A.x);
    R1 += __umul24(r.state0.x, r.A.y);

    R1 &= 0xFFFFFF;

    r.state0 = make_uint2(R0, R1);
}

__device__ inline int rand48_nextInt(Rand48 &r) {
    // get upper 31 (!) bits of the 2x 24bits
    int res = ( r.state0.x >> 17 ) | ( r.state0.y << 7 );
    rand48_iterate(r);
    return res;
}

// returns a float in the range [0, 1)
__device__ inline float rand48_nextFloat(Rand48 &r) {
    // use only upper 24 bits since floating point has 24 bit significand
    // (ref: Java random documentation)
    float res = r.state0.y / (float)(1<<24);
    rand48_iterate(r);
    return res;
}

//-----------------------------------------------------------------------------
// fill data with random floats between 0 and 1
__global__ void icg_create_randoms_kernel(Rand48 rng, float* data, 
                                          size_t width, size_t height, size_t pitch)
{
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col < width) {
        rand48_loadState(rng);

        for (unsigned int row = 0; row < height; row++){
            data[row * pitch + col] =  rand48_nextFloat(rng) * 2.0f - 1.0f;                     
        }
        rand48_storeState(rng);
    }
}

//-----------------------------------------------------------------------------
// fill data with random integers between a and b 
__global__ void icg_create_randoms_kernel(Rand48 rng, int* data, 
                                          size_t width, size_t height, size_t pitch,
                                          const int a, const int b)
{
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col < width) {
        rand48_loadState(rng);

        for (unsigned int row = 0; row < height; row++){
            float elem = rand48_nextFloat(rng);
            data[row * pitch + col] =  (b-a) * elem + a;                     
        }
        rand48_storeState(rng);
    }
}

///////////////////////////////////////////////////////////////////////////////
//! RNG HOST FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// Fills a 2D - DeviceMemoryPitched structure with random floating point values
// between 0 and 1.
// Returns true on success
bool cudaRandomNumbers(Cuda::DeviceMemoryPitched<float,2>* data)
{
    int n_threads = data->size[0];
    if (n_threads > 0 && data->size[1] > 0) {
        int seed = rand();    
        
        Rand48* rng = new Rand48();
        rng->init(n_threads, seed);

        int block_size = 64;
        dim3 dim_block(block_size, 1, 1);
        dim3 dim_grid(divUp(n_threads, block_size), 1 ,1);
        icg_create_randoms_kernel<<<dim_grid, dim_block>>>(
            *rng, data->getBuffer(), data->size[0], data->size[1],
            data->stride[0]);
        ICG_RF_CHECK_CUDA_ERROR();

        rng->destroy();
        delete rng;
    }
    return true;
}

//-----------------------------------------------------------------------------
// Fills a 2D - DeviceMemoryPitched structure with random integer values
// between a and b.
// Returns true on success

bool cudaRandomNumbers(Cuda::DeviceMemoryPitched<int,2>* data, int a, int b)
{
    int n_threads = data->size[0];
    if (n_threads > 0 && data->size[1] > 0) {
        int seed = rand();    

        Rand48* rng = new Rand48();
        rng->init(n_threads, seed);

        int block_size = 64;
        dim3 dim_block(block_size, 1, 1);
        dim3 dim_grid(divUp(n_threads, block_size), 1 ,1);
        icg_create_randoms_kernel<<<dim_grid, dim_block>>>(
            *rng, data->getBuffer(), data->size[0], data->size[1],
            data->stride[0], a, b);
        ICG_RF_CHECK_CUDA_ERROR();

        rng->destroy();
        delete rng;
    }
    return true;
}





#endif // #ifndef ICG_RF_RNG_CU