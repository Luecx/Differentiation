//
// Created by Luecx on 02.02.2021.
//

#ifndef NNLIBRARY_UTIL_H
#define NNLIBRARY_UTIL_H


inline void print_256i_epi8(const __m256i &h){
    printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n",
           _mm256_extract_epi8(h,0),
           _mm256_extract_epi8(h,1),
           _mm256_extract_epi8(h,2),
           _mm256_extract_epi8(h,3),
           _mm256_extract_epi8(h,4),
           _mm256_extract_epi8(h,5),
           _mm256_extract_epi8(h,6),
           _mm256_extract_epi8(h,7),
           _mm256_extract_epi8(h,8),
           _mm256_extract_epi8(h,9),
           _mm256_extract_epi8(h,10),
           _mm256_extract_epi8(h,11),
           _mm256_extract_epi8(h,12),
           _mm256_extract_epi8(h,13),
           _mm256_extract_epi8(h,14),
           _mm256_extract_epi8(h,15),
           _mm256_extract_epi8(h,16),
           _mm256_extract_epi8(h,17),
           _mm256_extract_epi8(h,18),
           _mm256_extract_epi8(h,19),
           _mm256_extract_epi8(h,20),
           _mm256_extract_epi8(h,21),
           _mm256_extract_epi8(h,22),
           _mm256_extract_epi8(h,23),
           _mm256_extract_epi8(h,24),
           _mm256_extract_epi8(h,25),
           _mm256_extract_epi8(h,26),
           _mm256_extract_epi8(h,27),
           _mm256_extract_epi8(h,28),
           _mm256_extract_epi8(h,29),
           _mm256_extract_epi8(h,30),
           _mm256_extract_epi8(h,31));
}

inline void print_256i_epi16(const __m256i &h){
    printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n",
           _mm256_extract_epi16(h,0),
           _mm256_extract_epi16(h,1),
           _mm256_extract_epi16(h,2),
           _mm256_extract_epi16(h,3),
           _mm256_extract_epi16(h,4),
           _mm256_extract_epi16(h,5),
           _mm256_extract_epi16(h,6),
           _mm256_extract_epi16(h,7),
           _mm256_extract_epi16(h,8),
           _mm256_extract_epi16(h,9),
           _mm256_extract_epi16(h,10),
           _mm256_extract_epi16(h,11),
           _mm256_extract_epi16(h,12),
           _mm256_extract_epi16(h,13),
           _mm256_extract_epi16(h,14),
           _mm256_extract_epi16(h,15));
}

inline void print_256i_epi32(const __m256i &h){
    printf("%d %d %d %d %d %d %d %d \n",
           _mm256_extract_epi32(h,0),
           _mm256_extract_epi32(h,1),
           _mm256_extract_epi32(h,2),
           _mm256_extract_epi32(h,3),
           _mm256_extract_epi32(h,4),
           _mm256_extract_epi32(h,5),
           _mm256_extract_epi32(h,6),
           _mm256_extract_epi32(h,7));
}

inline void print_256_pd(const __m256 &h){
    printf("%f %f %f %f \n",
           h[0],
           h[1],
           h[2],
           h[3]);
}

inline void print_256_ps(const __m256 &h){
    printf("%f %f %f %f %f %f %f %f \n",
           h[0],
           h[1],
           h[2],
           h[3],
           h[4],
           h[5],
           h[6],
           h[7]);
}

inline void print_128i_epi8(const __m128i &h){
    printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n",
           _mm_extract_epi8(h,0),
           _mm_extract_epi8(h,1),
           _mm_extract_epi8(h,2),
           _mm_extract_epi8(h,3),
           _mm_extract_epi8(h,4),
           _mm_extract_epi8(h,5),
           _mm_extract_epi8(h,6),
           _mm_extract_epi8(h,7),
           _mm_extract_epi8(h,8),
           _mm_extract_epi8(h,9),
           _mm_extract_epi8(h,10),
           _mm_extract_epi8(h,11),
           _mm_extract_epi8(h,12),
           _mm_extract_epi8(h,13),
           _mm_extract_epi8(h,14),
           _mm_extract_epi8(h,15));
}

inline void print_128i_epi16(const __m128i &h){
    printf("%d %d %d %d %d %d %d %d \n",
           _mm_extract_epi16(h,0),
           _mm_extract_epi16(h,1),
           _mm_extract_epi16(h,2),
           _mm_extract_epi16(h,3),
           _mm_extract_epi16(h,4),
           _mm_extract_epi16(h,5),
           _mm_extract_epi16(h,6),
           _mm_extract_epi16(h,7));
}

inline void print_128i_epi32(const __m128i &h){
    printf("%d %d %d %d \n",
           _mm_extract_epi32(h,0),
           _mm_extract_epi32(h,1),
           _mm_extract_epi32(h,2),
           _mm_extract_epi32(h,3));
}

inline uint32_t hsum_epi32_avx(__m128i x) {
    __m128i hi64 = _mm_unpackhi_epi64(x,x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

inline uint32_t hsum_8x32(__m256i v) {
    __m128i sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(v),
            _mm256_extracti128_si256(v, 1)); // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
    return hsum_epi32_avx(sum128);
}


#endif //NNLIBRARY_UTIL_H
