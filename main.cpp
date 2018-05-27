#include <iostream>
#include <functional>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <immintrin.h>

using namespace std;
using namespace std::chrono;

static const int ITERATIONS = 10;

long getTimeInMillis() {
    return system_clock::now().time_since_epoch() / milliseconds(1);
}

void testAligned(const std::string& name, std::function<void(char*,int)> func) {
    std::cout << name << ";";
    for(int i = 6; i < 30; i++) {
        int dataSize = std::pow(2,i) + 32;
        char* dataRaw = new char[dataSize];
        char* data = dataRaw + (32 - (reinterpret_cast<uintptr_t>(dataRaw) % 32));
        double result = 0.0;
        for(int j = 0; j < ITERATIONS; j++) {
            long start = getTimeInMillis();
            func(data,dataSize - 32);
            long stop = getTimeInMillis();
            result += (stop - start);
        }
        std::cout << (result / ITERATIONS) << ";";
        delete[] dataRaw;
    }
    std::cout << std::endl;
}

void testUnaligned(const std::string& name,std::function<void(char*,int)> func) {
    std::cout << name << ";";
    for(int i = 6; i < 30; i++) {
        int dataSize = std::pow(2,i);
        char* data = new char[dataSize];
        double result = 0.0;
        for(int j = 0; j < ITERATIONS; j++) {
            long start = getTimeInMillis();
            func(data,dataSize);
            long stop = getTimeInMillis();
            result += (stop - start);
        }
        std::cout << (result / ITERATIONS) << ";";
        delete[] data;
    }
    std::cout << std::endl;
}


void MemZeroLoop(char* data, int dataSize) {
    for(int i = 0; i < dataSize; i++) {
        data[i] = 0;
    }
}

void MemZeroLoopShort(char* data, int dataSize) {
    for(int i = 0; i < dataSize/2; i++) {
        reinterpret_cast<short*>(data)[i] = 0;
    }
}

void MemZeroLoopInt(char* data, int dataSize) {
    for(int i = 0; i < dataSize/4; i++) {
        reinterpret_cast<int*>(data)[i] = 0;
    }
}

void MemZeroLoopLong(char* data, int dataSize) {
    for(int i = 0; i < dataSize/8; i++) {
        reinterpret_cast<long*>(data)[i] = 0;
    }
}

void MemZeroMemset(char* data, int dataSize) {
    std::memset(data,0,dataSize);
}

#if defined(__SSE__)
void MemZeroSSEAlignedStream(char* data, int dataSize) {
    __m64 zero = _mm_setzero_si64();
    for(int i = 0; i < dataSize/8; i++) {
        _mm_stream_pi(reinterpret_cast<__m64*>(data) + i , zero);
    }
}
#endif

#if defined(__SSE2__)
void MemZeroSSE2Unaligned(char* data, int dataSize) {
    __m128i zero = _mm_setzero_si128();
    for(int i = 0; i < dataSize/16; i++) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(data) + i , zero);
    }
}

void MemZeroSSE2Aligned(char* data, int dataSize) {
    __m128i zero = _mm_setzero_si128();
    for(int i = 0; i < dataSize/16; i++) {
        _mm_store_si128(reinterpret_cast<__m128i*>(data) + i , zero);
    }
}

void MemZeroSSE2AlignedStream(char* data, int dataSize) {
    __m128i zero = _mm_setzero_si128();
    for(int i = 0; i < dataSize/16; i++) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(data) + i , zero);
    }
}
#endif

#if defined(__AVX__)
void MemZeroAVXUnaligned(char* data, int dataSize) {
     _mm256_zeroupper();
    __m256i zero = _mm256_setzero_si256();
    for(int i = 0; i < dataSize/32; i++) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(data) + i , zero);
    }
    _mm256_zeroupper();
}

void MemZeroAVXAligned(char* data, int dataSize) {
     _mm256_zeroupper();
    __m256i zero = _mm256_setzero_si256();
    for(int i = 0; i < dataSize/32; i++) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(data) + i , zero);
    }
     _mm256_zeroupper();
}

void MemZeroAVXAlignedStream(char* data, int dataSize) {
     _mm256_zeroupper();
    __m256i zero = _mm256_setzero_si256();
    for(int i = 0; i < dataSize/32; i++) {
        _mm256_stream_si256(reinterpret_cast<__m256i*>(data) + i , zero);
    }
     _mm256_zeroupper();
}
#endif


int main()
{
    testUnaligned("std::memset",MemZeroMemset);
    testUnaligned("Loop(char)",MemZeroLoop);
    testUnaligned("Loop(short)",MemZeroLoopShort);
    testUnaligned("Loop(int)",MemZeroLoopInt);
    testUnaligned("Loop(long)",MemZeroLoopLong);
#if defined(__SSE__)
    testAligned("SSE(aligned_stream)",MemZeroSSEAlignedStream);
#endif
#if defined(__SSE2__)
    testUnaligned("SSE2(unlaligned)",MemZeroSSE2Unaligned);
    testAligned("SSE2(aligned)",MemZeroSSE2Aligned);
    testAligned("SSE2(aligned_stream)",MemZeroSSE2AlignedStream);
#endif
#if defined(__AVX__)
    testUnaligned("AVX(unlaligned)",MemZeroAVXUnaligned);
    testAligned("AVX(aligned)",MemZeroAVXAligned);
    testAligned("AVX(aligned_stream)",MemZeroAVXAlignedStream);
#endif
    return 0;
}
