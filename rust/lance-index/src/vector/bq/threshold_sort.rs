// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixed-pass threshold sorting with a NEON bucket prefix scan on AArch64.

use std::sync::LazyLock;

#[cfg(all(target_arch = "aarch64", target_pointer_width = "64"))]
use std::arch::aarch64::{
    vaddq_u64, vdupq_laneq_u64, vdupq_n_u64, vextq_u64, vld1q_u64, vst1q_u64,
};

// Read once per process so alternating builds can share one compiled extension.
//
// The NEON prefix scan is opt-in: set `LANCE_RQ_THRESHOLD_SORT=1` (or `true`)
// to enable it. It measured no end-to-end benefit on the PCA-IVF-RQ build, so
// it stays off by default. `LANCE_DISABLE_AARCH64_RQ_PREFIX_SUM=1` also forces
// it off, for symmetry with the other aarch64 switches.
static OPTIMIZED: LazyLock<bool> = LazyLock::new(|| {
    #[cfg(all(target_arch = "aarch64", target_pointer_width = "64"))]
    {
        if matches!(
            std::env::var("LANCE_DISABLE_AARCH64_RQ_PREFIX_SUM").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        ) {
            return false;
        }
        std::env::var_os("LANCE_RQ_THRESHOLD_SORT")
            .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    }
    #[cfg(not(all(target_arch = "aarch64", target_pointer_width = "64")))]
    {
        false
    }
});

/// Sort positive, finite f32 keys in the upper word, retaining equal-key order.
pub(super) fn sort(values: &mut [u64], scratch: &mut Vec<u64>) {
    sort_with_mode(values, scratch, *OPTIMIZED);
}

fn sort_with_mode(values: &mut [u64], scratch: &mut Vec<u64>, is_optimized: bool) {
    if values.len() < 2 {
        return;
    }

    scratch.resize(values.len(), 0);
    if is_optimized {
        pass::<32, true>(values, scratch);
        pass::<40, true>(scratch, values);
        pass::<48, true>(values, scratch);
        pass::<56, true>(scratch, values);
    } else {
        pass::<32, false>(values, scratch);
        pass::<40, false>(scratch, values);
        pass::<48, false>(values, scratch);
        pass::<56, false>(scratch, values);
    }
}

// Specialize the shift so histogram and scatter retain fixed-byte extraction.
#[inline]
fn pass<const SHIFT: u32, const IS_OPTIMIZED: bool>(source: &[u64], destination: &mut [u64]) {
    let mut offsets = [0usize; 256];
    for &value in source {
        offsets[((value >> SHIFT) & 0xff) as usize] += 1;
    }
    prefix_offsets(&mut offsets, IS_OPTIMIZED);
    for &value in source {
        let bucket = ((value >> SHIFT) & 0xff) as usize;
        destination[offsets[bucket]] = value;
        offsets[bucket] += 1;
    }
}

#[inline]
fn prefix_offsets(offsets: &mut [usize; 256], is_optimized: bool) {
    #[cfg(not(all(target_arch = "aarch64", target_pointer_width = "64")))]
    let _ = is_optimized;
    #[cfg(all(target_arch = "aarch64", target_pointer_width = "64"))]
    if is_optimized && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: NEON was detected and the kernel accesses only this array.
        unsafe { prefix_offsets_neon(offsets) };
        return;
    }
    let mut next = 0;
    for offset in offsets {
        let count = *offset;
        *offset = next;
        next += count;
    }
}

#[cfg(all(target_arch = "aarch64", target_pointer_width = "64"))]
#[target_feature(enable = "neon")]
unsafe fn prefix_offsets_neon(offsets: &mut [usize; 256]) {
    // SAFETY: This kernel requires 64-bit usize. Chunks contain four elements,
    // and every load/store stays inside the current chunk. Bucket counts and
    // their sum fit usize because they count entries of an existing slice.
    unsafe {
        let zero = vdupq_n_u64(0);
        let mut carry = zero;
        for chunk in offsets.chunks_exact_mut(4) {
            let ptr = chunk.as_mut_ptr().cast::<u64>();
            let a = vld1q_u64(ptr);
            let b = vld1q_u64(ptr.add(2));
            let sum_a = vaddq_u64(a, vextq_u64::<1>(zero, a));
            let sum_b = vaddq_u64(b, vextq_u64::<1>(zero, b));
            let sum_b = vaddq_u64(sum_b, vdupq_laneq_u64::<1>(sum_a));
            let first = vaddq_u64(vextq_u64::<1>(zero, sum_a), carry);
            let second = vaddq_u64(vextq_u64::<1>(sum_a, sum_b), carry);
            vst1q_u64(ptr, first);
            vst1q_u64(ptr.add(2), second);
            carry = vaddq_u64(carry, vdupq_laneq_u64::<1>(sum_b));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case(0)]
    #[case(1)]
    #[case(3)]
    #[case(31)]
    #[case(256)]
    #[case(1025)]
    fn threshold_sort_matches_baseline_for_all_key_byte_patterns(#[case] len: usize) {
        let mut scratch = Vec::new();
        for mask in 0..16 {
            let original = (0..len)
                .map(|idx| {
                    let mixed = (idx as u32).wrapping_mul(2654435761);
                    let mut key = 0x3f010101u32;
                    for byte in 0..4 {
                        if mask & (1 << byte) != 0 {
                            key ^= (mixed.rotate_left(byte * 7) & 0xf) << (byte * 8);
                        }
                    }
                    ((key as u64) << 32) | (len - idx) as u64
                })
                .collect::<Vec<_>>();
            let mut expected = original.clone();
            expected.sort_by_key(|value| value >> 32);
            for is_optimized in [false, true] {
                let mut actual = original.clone();
                sort_with_mode(&mut actual, &mut scratch, is_optimized);
                assert_eq!(actual, expected, "mask={mask}, optimized={is_optimized}");
            }
        }
    }

    #[test]
    fn threshold_sort_prefix_matches_scalar() {
        let mut expected = [0usize; 256];
        for (idx, count) in expected.iter_mut().enumerate() {
            *count = if idx % 7 == 0 { 0 } else { idx * 7919 };
            #[cfg(target_pointer_width = "64")]
            if idx % 11 == 0 {
                *count += 1usize << 33;
            }
        }
        let mut actual = expected;
        prefix_offsets(&mut expected, false);
        prefix_offsets(&mut actual, true);
        assert_eq!(actual, expected);
    }
}
