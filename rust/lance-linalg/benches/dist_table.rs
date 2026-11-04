// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of 4-bit LUT distance table summation (RaBitQ inner loop).
//!
//! On aarch64 the default build times both NEON kernels and the scalar
//! fallback in one run:
//!
//! * `neon` is the original kernel
//! * `neon_krl` is the `vsli` / `tbl` / `vpadal` trial
//!
//! Cargo features on `lance-linalg` narrow that:
//!
//! ```text
//! cargo bench -p lance-linalg --bench dist_table --features neon-dist-table-baseline
//! cargo bench -p lance-linalg --bench dist_table --features neon-dist-table-krl
//! ```
//!
//! `neon-dist-table-krl` alone also makes `sum_4bit_dist_table` use the trial
//! kernel. Enabling both features times both kernels and leaves the dispatched
//! entry on the original kernel.

use std::iter::repeat_with;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(not(target_arch = "aarch64"))]
use lance_linalg::simd::dist_table::sum_4bit_dist_table;
use lance_linalg::simd::dist_table::{BATCH_SIZE, sum_4bit_dist_table_scalar};
#[cfg(target_arch = "aarch64")]
use lance_linalg::simd::dist_table::{sum_4bit_dist_table_neon, sum_4bit_dist_table_neon_krl};
use rand::Rng;

fn bench_sum_4bit_dist_table(c: &mut Criterion) {
    let mut rng = rand::rng();

    // code_len = dim / 8 for 1-bit quantization
    for (label, n_vectors, code_len) in [
        ("32vec_dim128", 32_usize, 16_usize),
        ("32vec_dim1536", 32, 192),
        ("32vec_dim4096", 32, 512),
        ("32vec_dim65536", 32, 8192),
        ("16Kvec_dim128", 16_000, 16),
        ("16Kvec_dim1536", 16_000, 192),
    ] {
        let n = n_vectors.div_ceil(BATCH_SIZE) * BATCH_SIZE;

        let codes: Vec<u8> = repeat_with(|| rng.random::<u8>())
            .take(n * code_len)
            .collect();

        let dist_table: Vec<u8> = repeat_with(|| rng.random::<u8>())
            .take(BATCH_SIZE * code_len)
            .collect();

        let mut dists = vec![0u16; n];

        #[cfg(target_arch = "aarch64")]
        bench_aarch64_neon(c, label, n, code_len, &codes, &dist_table, &mut dists);
        #[cfg(not(target_arch = "aarch64"))]
        c.bench_function(&format!("sum_4bit_dist_table/simd/{label}"), |b| {
            b.iter(|| {
                dists.fill(0);
                sum_4bit_dist_table(n, code_len, &codes, &dist_table, &mut dists);
                black_box(&dists);
            })
        });

        // Scalar reference path
        c.bench_function(&format!("sum_4bit_dist_table/scalar/{}", label), |b| {
            b.iter(|| {
                dists.fill(0);
                sum_4bit_dist_table_scalar(code_len, &codes, &dist_table, &mut dists);
                black_box(&dists);
            })
        });
    }
}

/// Times the original NEON kernel, the KRL trial, or both.
///
/// Neither cargo feature times both. `neon-dist-table-baseline` or
/// `neon-dist-table-krl` alone times that kernel.
#[cfg(target_arch = "aarch64")]
fn bench_aarch64_neon(
    c: &mut Criterion,
    label: &str,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u16],
) {
    const BENCH_BASELINE: bool = cfg!(any(
        feature = "neon-dist-table-baseline",
        not(feature = "neon-dist-table-krl"),
    ));
    const BENCH_KRL: bool = cfg!(any(
        feature = "neon-dist-table-krl",
        not(feature = "neon-dist-table-baseline"),
    ));

    if BENCH_BASELINE {
        c.bench_function(&format!("sum_4bit_dist_table/neon/{label}"), |b| {
            b.iter(|| {
                dists.fill(0);
                sum_4bit_dist_table_neon(n, code_len, codes, dist_table, dists);
                black_box(&*dists);
            })
        });
    }
    if BENCH_KRL {
        c.bench_function(&format!("sum_4bit_dist_table/neon_krl/{label}"), |b| {
            b.iter(|| {
                dists.fill(0);
                sum_4bit_dist_table_neon_krl(n, code_len, codes, dist_table, dists);
                black_box(&*dists);
            })
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_sum_4bit_dist_table
);
criterion_main!(benches);
