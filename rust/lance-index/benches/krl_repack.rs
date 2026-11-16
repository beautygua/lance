// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Isolate the cost of rewriting Lance packed codes into the KRL layout.
//!
//! Times the current implementation (`NEW_fused`) against the previous one
//! (`OLD_inplace`) in the same process, so both see the same machine state.
//!
//! ```text
//! cargo bench -p lance-index --bench krl_repack --features neon-dist-table-krl
//! ```

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array, UInt64Array};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_index::vector::bq::RQRotationType;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::storage::{
    RABIT_CODE_COLUMN, RabitQuantizationMetadata, RabitQuantizationStorage,
    repack_lance_packed_to_krl,
};
use lance_index::vector::bq::transform::{ADD_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
use lance_index::vector::quantizer::{Quantization, QuantizerStorage};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use lance_linalg::simd::dist_table::{BATCH_SIZE, PERM0};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 1536;

fn make_codes(rows: usize, code_len: usize) -> Vec<u8> {
    let mut rng = SmallRng::seed_from_u64(0x5eed);
    (0..rows * code_len)
        .map(|_| rng.random_range(0..=u8::MAX))
        .collect()
}

/// The previous implementation: full `to_vec()` copy, then in-place rewrite.
fn old_form_impl(codes: &[u8], code_len: usize) -> Vec<u8> {
    if code_len == 0 {
        return codes.to_vec();
    }
    let batch_bytes = BATCH_SIZE * code_len;
    let full = codes.len() / batch_bytes * batch_bytes;
    let mut out = codes.to_vec();
    for batch in out[..full].chunks_exact_mut(batch_bytes) {
        for block in batch.chunks_exact_mut(BATCH_SIZE) {
            let mut sq0 = [0u8; BATCH_SIZE];
            let mut sq1 = [0u8; BATCH_SIZE];
            for lane in 0..16 {
                sq0[PERM0[lane]] = block[lane] & 0x0f;
                sq0[PERM0[lane] + 16] = block[lane] >> 4;
                sq1[PERM0[lane]] = block[lane + 16] & 0x0f;
                sq1[PERM0[lane] + 16] = block[lane + 16] >> 4;
            }
            let mut dst = [0u8; BATCH_SIZE];
            for group in (0..BATCH_SIZE).step_by(16) {
                for pair in 0..8 {
                    dst[group + 2 * pair] = sq0[group + pair] | (sq0[group + pair + 8] << 4);
                    dst[group + 2 * pair + 1] = sq1[group + pair] | (sq1[group + pair + 8] << 4);
                }
            }
            block.copy_from_slice(&dst);
        }
    }
    out
}


/// Explicit fused form, for comparison with the size-selected public entry.
fn fused_form_impl(codes: &[u8], code_len: usize) -> Vec<u8> {
    if code_len == 0 {
        return codes.to_vec();
    }
    let batch_bytes = BATCH_SIZE * code_len;
    let full = codes.len() / batch_bytes * batch_bytes;
    let mut out = Vec::with_capacity(codes.len());
    for block in codes[..full].chunks_exact(BATCH_SIZE) {
        out.extend_from_slice(&krl_block_from_lance_copy(block));
    }
    out.extend_from_slice(&codes[full..]);
    out
}

fn krl_block_from_lance_copy(block: &[u8]) -> [u8; BATCH_SIZE] {
    let mut sq0 = [0u8; BATCH_SIZE];
    let mut sq1 = [0u8; BATCH_SIZE];
    for lane in 0..16 {
        sq0[PERM0[lane]] = block[lane] & 0x0f;
        sq0[PERM0[lane] + 16] = block[lane] >> 4;
        sq1[PERM0[lane]] = block[lane + 16] & 0x0f;
        sq1[PERM0[lane] + 16] = block[lane + 16] >> 4;
    }
    let mut dst = [0u8; BATCH_SIZE];
    for group in (0..BATCH_SIZE).step_by(16) {
        for pair in 0..8 {
            dst[group + 2 * pair] = sq0[group + pair] | (sq0[group + pair + 8] << 4);
            dst[group + 2 * pair + 1] = sq1[group + pair] | (sq1[group + pair + 8] << 4);
        }
    }
    dst
}


/// Inline copy of the size-selected public entry, so it is measured under the
/// same inlining conditions as `OLD_inplace` / `NEW_fused`.
fn c2_inline_impl(codes: &[u8], code_len: usize) -> Vec<u8> {
    const FUSED_MAX_BYTES: usize = 16 * 1024 * 1024;
    if code_len == 0 {
        return codes.to_vec();
    }
    let batch_bytes = BATCH_SIZE * code_len;
    let full = codes.len() / batch_bytes * batch_bytes;
    if codes.len() <= FUSED_MAX_BYTES {
        let mut out = Vec::with_capacity(codes.len());
        for block in codes[..full].chunks_exact(BATCH_SIZE) {
            out.extend_from_slice(&krl_block_from_lance_copy(block));
        }
        out.extend_from_slice(&codes[full..]);
        out
    } else {
        let mut out = codes.to_vec();
        for batch in out[..full].chunks_exact_mut(batch_bytes) {
            for block in batch.chunks_exact_mut(BATCH_SIZE) {
                let converted = krl_block_from_lance_copy(block);
                block.copy_from_slice(&converted);
            }
        }
        out
    }
}

/// Guard: the fused repack must produce exactly the bytes of the old form.
fn assert_fused_matches_in_place() {
    for code_len in [1usize, 8, 16, 192] {
        for rows in [1usize, 31, 32, 33, 64, 100, 1024] {
            let codes = make_codes(rows, code_len);
            let want = old_form_impl(&codes, code_len);
            assert_eq!(
                repack_lance_packed_to_krl(&codes, code_len),
                want,
                "hybrid mismatch code_len={code_len} rows={rows}"
            );
            assert_eq!(
                fused_form_impl(&codes, code_len),
                want,
                "fused mismatch code_len={code_len} rows={rows}"
            );
            assert_eq!(
                c2_inline_impl(&codes, code_len),
                want,
                "c2 mismatch code_len={code_len} rows={rows}"
            );
        }
    }
    eprintln!("krl_repack: fused form matches in-place form");
}

fn metadata() -> RabitQuantizationMetadata {
    // Real rotation metadata: the query path rotates through it.
    RabitQuantizer::new_with_rotation::<Float32Type>(1, DIM as i32, RQRotationType::Fast)
        .metadata(None)
}

fn make_batch(rows: usize, code_len: usize, codes: Vec<u8>) -> RecordBatch {
    RecordBatch::try_from_iter(vec![
        (
            ROW_ID,
            Arc::new(UInt64Array::from_iter_values(0..rows as u64)) as ArrayRef,
        ),
        (
            RABIT_CODE_COLUMN,
            Arc::new(
                FixedSizeListArray::try_new_from_values(UInt8Array::from(codes), code_len as i32)
                    .unwrap(),
            ) as ArrayRef,
        ),
        (
            ADD_FACTORS_COLUMN,
            Arc::new(Float32Array::from(vec![0.0f32; rows])) as ArrayRef,
        ),
        (
            SCALE_FACTORS_COLUMN,
            Arc::new(Float32Array::from(vec![0.0f32; rows])) as ArrayRef,
        ),
    ])
    .unwrap()
}

fn bench_repack(c: &mut Criterion) {
    assert_fused_matches_in_place();
    let code_len = DIM / 8;
    let mut group = c.benchmark_group("krl_repack/raw");

    for rows in [512usize, 8192, 65_536, 262_144, 1_048_576] {
        let codes = make_codes(rows, code_len);
        group.throughput(Throughput::Bytes((rows * code_len) as u64));
        group.bench_with_input(BenchmarkId::new("OLD_inplace", rows), &codes, |b, codes| {
            b.iter(|| black_box(old_form_impl(black_box(codes), code_len)))
        });
        group.bench_with_input(BenchmarkId::new("NEW_fused", rows), &codes, |b, codes| {
            b.iter(|| black_box(fused_form_impl(black_box(codes), code_len)))
        });
        group.bench_with_input(BenchmarkId::new("C2_inline", rows), &codes, |b, codes| {
            b.iter(|| black_box(c2_inline_impl(black_box(codes), code_len)))
        });
        group.bench_with_input(BenchmarkId::new("copy_only", rows), &codes, |b, codes| {
            b.iter(|| black_box(codes.to_vec()))
        });
    }
    group.finish();
}

fn bench_storage_load(c: &mut Criterion) {
    let code_len = DIM / 8;
    let mut group = c.benchmark_group("krl_repack/storage_load");

    for rows in [512usize, 8192, 65_536, 262_144, 1_048_576] {
        let row_major = make_codes(rows, code_len);
        let packed_bytes = {
            let fsl = FixedSizeListArray::try_new_from_values(
                UInt8Array::from(row_major.clone()),
                code_len as i32,
            )
            .unwrap();
            let packed = lance_index::vector::bq::storage::pack_codes(&fsl);
            packed.values().as_primitive::<arrow_array::types::UInt8Type>().values().to_vec()
        };
        let krl_bytes = repack_lance_packed_to_krl(&packed_bytes, code_len);

        group.throughput(Throughput::Bytes((rows * code_len) as u64));

        // Old path: row-major on disk, load does `pack_codes` and then builds a
        // second KRL copy. Two code arrays stay resident.
        let mut md_old = metadata();
        md_old.packed = false;
        let batch_old = make_batch(rows, code_len, row_major);
        group.bench_with_input(
            BenchmarkId::new("old_packed_disk", rows),
            &batch_old,
            |b, batch| {
                b.iter(|| {
                    black_box(
                        RabitQuantizationStorage::try_from_batch(
                            black_box(batch.clone()),
                            &md_old,
                            DistanceType::L2,
                            None,
                        )
                        .unwrap(),
                    )
                })
            },
        );

        // New path: KRL on disk, load keeps the single copy.
        let mut md_krl = metadata();
        md_krl.packed = true;
        md_krl.krl_layout = true;
        let batch_krl = make_batch(rows, code_len, krl_bytes);
        group.bench_with_input(
            BenchmarkId::new("new_krl_disk", rows),
            &batch_krl,
            |b, batch| {
                b.iter(|| {
                    black_box(
                        RabitQuantizationStorage::try_from_batch(
                            black_box(batch.clone()),
                            &md_krl,
                            DistanceType::L2,
                            None,
                        )
                        .unwrap(),
                    )
                })
            },
        );

        // Control: packed on disk with no KRL copy at all.
        let mut md_packed = metadata();
        md_packed.packed = true;
        let batch_packed = make_batch(rows, code_len, packed_bytes);
        group.bench_with_input(
            BenchmarkId::new("packed_disk_no_krl", rows),
            &batch_packed,
            |b, batch| {
                b.iter(|| {
                    black_box(
                        RabitQuantizationStorage::try_from_batch(
                            black_box(batch.clone()),
                            &md_packed,
                            DistanceType::L2,
                            None,
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let code_len = DIM / 8;
    let mut group = c.benchmark_group("krl_repack/query");

    for rows in [4096usize, 65_536, 262_144, 1_048_576] {
        let row_major = make_codes(rows, code_len);
        let packed_bytes = {
            let fsl = FixedSizeListArray::try_new_from_values(
                UInt8Array::from(row_major.clone()),
                code_len as i32,
            )
            .unwrap();
            let packed = lance_index::vector::bq::storage::pack_codes(&fsl);
            packed
                .values()
                .as_primitive::<arrow_array::types::UInt8Type>()
                .values()
                .to_vec()
        };
        let krl_bytes = repack_lance_packed_to_krl(&packed_bytes, code_len);

        // Old: row-major on disk, pack + repack at load, two copies resident.
        let mut md_old = metadata();
        md_old.packed = false;
        let old_storage = RabitQuantizationStorage::try_from_batch(
            make_batch(rows, code_len, row_major),
            &md_old,
            DistanceType::L2,
            None,
        )
        .unwrap();

        // New: KRL on disk, one copy resident.
        let mut md_krl = metadata();
        md_krl.packed = true;
        md_krl.krl_layout = true;
        let krl_storage = RabitQuantizationStorage::try_from_batch(
            make_batch(rows, code_len, krl_bytes),
            &md_krl,
            DistanceType::L2,
            None,
        )
        .unwrap();

        let query = Arc::new(Float32Array::from_iter_values(
            (0..DIM).map(|i| (i % 13) as f32 * 0.125 - 0.75),
        )) as ArrayRef;

        group.throughput(Throughput::Bytes((rows * code_len) as u64));
        // Calculator built once per input so the timed region is the scan.
        group.bench_with_input(
            BenchmarkId::new("old_packed_disk", rows),
            &old_storage,
            |b, storage| {
                let calc = storage.dist_calculator(query.clone(), 0.0);
                b.iter(|| black_box(calc.distance_all(0)))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("new_krl_disk", rows),
            &krl_storage,
            |b, storage| {
                let calc = storage.dist_calculator(query.clone(), 0.0);
                b.iter(|| black_box(calc.distance_all(0)))
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(5));
    targets = bench_repack, bench_storage_load, bench_query
);
criterion_main!(benches);
