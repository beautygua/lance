// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// 256-bit SVE fast-scan for the blocked 4-bit distance table.
//
// The arithmetic matches `sum_dist_table_32bytes_batch_neon`: each 32-byte
// step holds two independent 16-byte LUTs, low and high nibbles are looked
// up separately, even and odd bytes accumulate in `u16` lanes, and the two
// 16-byte halves are cleaned and then added. Callers must only enter this
// file when `svcntb() == 32`; the fold below addresses lanes 8..15.

#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

size_t lance_sve_cntb(void) { return svcntb(); }

static void store_folded_u16(uint16_t *dst, svuint16_t even, svuint16_t odd) {
  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg8 = svwhilelt_b16_u64(0, 8);
  const svuint16_t upper = svindex_u16(8, 1);

  even = svsub_u16_x(pg16, even, svlsl_n_u16_x(pg16, odd, 8));
  even = svadd_u16_x(pg16, even, svtbl_u16(even, upper));
  odd = svadd_u16_x(pg16, odd, svtbl_u16(odd, upper));
  svst1_u16(pg8, dst, even);
  svst1_u16(pg8, dst + 8, odd);
}

void sum_4bit_dist_table_32bytes_batch_sve(const uint8_t *codes,
                                           size_t code_length,
                                           const uint8_t *dist_table,
                                           uint16_t *dists) {
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg16 = svptrue_b16();
  const svuint8_t low_mask = svdup_n_u8(0x0f);
  // Bytes 0..15 index the first LUT. Bytes 16..31 index the second, which
  // sits in the upper half of the same 32-byte register.
  const svuint8_t lane_bias =
      svsel_u8(svwhilelt_b8_u64(0, 16), svdup_n_u8(0), svdup_n_u8(16));

  svuint16_t even_lo = svdup_n_u16(0);
  svuint16_t odd_lo = svdup_n_u16(0);
  svuint16_t even_hi = svdup_n_u16(0);
  svuint16_t odd_hi = svdup_n_u16(0);

  for (size_t i = 0; i + 32 <= code_length; i += 32) {
    const svuint8_t codes_v = svld1_u8(pg8, codes + i);
    const svuint8_t lut = svld1_u8(pg8, dist_table + i);
    svuint8_t lo = svand_u8_z(pg8, codes_v, low_mask);
    svuint8_t hi = svlsr_n_u8_z(pg8, codes_v, 4);
    lo = svadd_u8_z(pg8, lo, lane_bias);
    hi = svadd_u8_z(pg8, hi, lane_bias);

    const svuint8_t res_lo = svtbl_u8(lut, lo);
    const svuint8_t res_hi = svtbl_u8(lut, hi);
    const svuint16_t pairs_lo = svreinterpret_u16_u8(res_lo);
    const svuint16_t pairs_hi = svreinterpret_u16_u8(res_hi);

    even_lo = svadd_u16_x(pg16, even_lo, pairs_lo);
    odd_lo = svadd_u16_x(pg16, odd_lo, svlsr_n_u16_x(pg16, pairs_lo, 8));
    even_hi = svadd_u16_x(pg16, even_hi, pairs_hi);
    odd_hi = svadd_u16_x(pg16, odd_hi, svlsr_n_u16_x(pg16, pairs_hi, 8));
  }

  store_folded_u16(dists, even_lo, odd_lo);
  store_folded_u16(dists + 16, even_hi, odd_hi);
}
