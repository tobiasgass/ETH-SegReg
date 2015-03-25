#pragma once



class BlockwiseAllocater{

private:

    struct MallocBlock
	{
		static const int minBlockSizeInBytes = 10000*4096 - 3*sizeof(void*);
		MallocBlock*	m_next;
		char*			m_current; // first element of available memory in this block
		char*			m_last; // first element outside of allocated memory for this block
        long int size;
	};
    
    MallocBlock*	m_mallocBlockFirst;

public:
    BlockwiseAllocater(){
        m_mallocBlockFirst=NULL;
    }
 ~BlockwiseAllocater(){
        while (m_mallocBlockFirst)
            {
                MallocBlock* next = m_mallocBlockFirst->m_next;
                free(m_mallocBlockFirst);
                m_mallocBlockFirst = next;
            }
    }
    char * allocate(int bytesNum){
        if (!m_mallocBlockFirst || m_mallocBlockFirst->m_current+bytesNum > m_mallocBlockFirst->m_last)
            {
                int size = (bytesNum > MallocBlock::minBlockSizeInBytes) ? bytesNum : MallocBlock::minBlockSizeInBytes;

                MallocBlock* b = (MallocBlock*) malloc(sizeof(MallocBlock) + size);
                if (!b) {
                    std::cerr<<"Not enough memory"<<std::endl;;
                    exit(10);
                }
                b->m_current = (char*) b + sizeof(MallocBlock);
                b->m_last    = b->m_current + size;

                b->m_next = m_mallocBlockFirst;
                b->size=sizeof(MallocBlock) + size;
                m_mallocBlockFirst = b;
            }
        char* ptr = m_mallocBlockFirst->m_current;
        m_mallocBlockFirst->m_current += bytesNum;
        return ptr;
    }


   
    
    template<typename T>
    T*  New(){
        return (T*) allocate(sizeof(T));
    }
    
    template<typename T>
    T*  New(int n) {
        return (T*) allocate(n*sizeof(T));
    }

};
