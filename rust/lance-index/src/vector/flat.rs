// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat Vector Index.
//!

use std::sync::{Arc, LazyLock};

use arrow::{array::AsArray, buffer::NullBuffer};
use arrow_array::{Array, ArrayRef, Float32Array, RecordBatch, make_array};
use arrow_schema::{DataType, Field as ArrowField};
use lance_arrow::*;
use lance_core::{Error, ROW_ID, Result};
use lance_linalg::distance::{DistanceType, multivec_distance};
use tracing::instrument;

use super::DIST_COL;

pub mod index;
pub(crate) mod pairwise;
pub mod storage;
pub mod transform;

fn distance_field() -> ArrowField {
    ArrowField::new(DIST_COL, DataType::Float32, true)
}

/// Get a column from a RecordBatch, supporting nested field paths.
///
/// This function handles:
/// - Simple column names: "column"
/// - Nested paths: "parent.child" or "parent.child.grandchild"
/// - Backtick-escaped field names: "parent.`field.with.dots`"
fn get_column_from_batch(batch: &RecordBatch, column: &str) -> Result<ArrayRef> {
    // Try to get the column directly first (fast path for simple columns)
    if let Some(col) = batch.column_by_name(column) {
        return Ok(col.clone());
    }

    // Parse the field path using Lance's field path parsing logic
    // This properly handles backtick-escaped field names
    let parts = lance_core::datatypes::parse_field_path(column)
        .map_err(|e| Error::schema(format!("Failed to parse field path '{}': {}", column, e)))?;

    if parts.is_empty() {
        return Err(Error::schema(format!(
            "Invalid empty field path: {}",
            column
        )));
    }

    // Get the root column
    let mut current_array: ArrayRef = batch
        .column_by_name(&parts[0])
        .ok_or_else(|| {
            Error::schema(format!(
                "Column '{}' does not exist in batch (looking for root field '{}')",
                column, parts[0]
            ))
        })?
        .clone();

    // Navigate through nested struct fields
    for part in &parts[1..] {
        let struct_array = current_array
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| {
                Error::schema(format!(
                    "Cannot access nested field '{}' in column '{}': parent is not a struct",
                    part, column
                ))
            })?;

        current_array = struct_array
            .column_by_name(part)
            .ok_or_else(|| {
                Error::schema(format!(
                    "Nested field '{}' does not exist in column '{}'",
                    part, column
                ))
            })?
            .clone();
    }

    Ok(current_array)
}

/// Environment variable overriding the element count below which
/// [`compute_distance`] runs inline instead of on the blocking pool.
/// `0` forces every batch onto the blocking pool (the pre-N17 behaviour).
pub const FLAT_DISTANCE_INLINE_THRESHOLD_ENV: &str = "LANCE_FLAT_DISTANCE_INLINE_THRESHOLD";

/// Default inline threshold, in vector elements (rows × dimension).
///
/// A refine step scores a few dozen rows; that is microseconds of work,
/// far less than the two thread hand-offs of a `spawn_blocking` round trip.
/// 256 Ki f32 elements is on the order of 50 µs of distance work, short
/// enough to hold an async worker.
pub const FLAT_DISTANCE_INLINE_THRESHOLD_DEFAULT: usize = 256 * 1024;

static FLAT_DISTANCE_INLINE_THRESHOLD: LazyLock<usize> = LazyLock::new(|| {
    std::env::var(FLAT_DISTANCE_INLINE_THRESHOLD_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(FLAT_DISTANCE_INLINE_THRESHOLD_DEFAULT)
});

/// Number of scalar elements the distance kernel will touch for `vectors`.
fn distance_work_elements(vectors: &ArrayRef) -> usize {
    match vectors.data_type() {
        DataType::FixedSizeList(_, dim) => vectors.len().saturating_mul((*dim).max(0) as usize),
        DataType::List(_) => {
            let list = vectors.as_list::<i32>();
            let offsets = list.value_offsets();
            let items = offsets
                .last()
                .copied()
                .unwrap_or(0)
                .saturating_sub(offsets.first().copied().unwrap_or(0));
            (items.max(0) as usize).saturating_mul(match list.values().data_type() {
                DataType::FixedSizeList(_, dim) => (*dim).max(0) as usize,
                _ => 1,
            })
        }
        _ => usize::MAX,
    }
}

fn compute_distance_sync(
    key: ArrayRef,
    dt: DistanceType,
    vectors: ArrayRef,
    validity_buffer: Option<NullBuffer>,
    batch: RecordBatch,
) -> Result<RecordBatch> {
    // A selection vector may have been applied to _rowid column, so we need to
    // push that onto vectors if possible.

    let vectors = vectors
        .into_data()
        .into_builder()
        .null_bit_buffer(validity_buffer.map(|b| b.buffer().clone()))
        .build()
        .map(make_array)?;
    let distances = match vectors.data_type() {
        DataType::FixedSizeList(_, _) => {
            let vectors = vectors.as_fixed_size_list();
            dt.arrow_batch_func()(key.as_ref(), vectors)? as ArrayRef
        }
        DataType::List(_) => {
            let vectors = vectors.as_list();
            let dists = multivec_distance(key.as_ref(), vectors, dt)?;
            Arc::new(Float32Array::from(dists))
        }
        _ => {
            unreachable!()
        }
    };

    batch
        .try_with_column(distance_field(), distances)
        .map_err(|e| Error::execution(format!("Failed to adding distance column: {}", e)))
}

/// Append a `_distance` column with the distance from `key` to each row's vector.
///
/// Small batches (fewer than [`FLAT_DISTANCE_INLINE_THRESHOLD_DEFAULT`]
/// elements, overridable with `LANCE_FLAT_DISTANCE_INLINE_THRESHOLD`) are
/// scored on the calling task; larger ones on the blocking pool.  The result
/// is identical either way.
#[instrument(level = "debug", skip_all)]
pub async fn compute_distance(
    key: ArrayRef,
    dt: DistanceType,
    column: &str,
    mut batch: RecordBatch,
) -> Result<RecordBatch> {
    if batch.column_by_name(DIST_COL).is_some() {
        // Ignore the distance calculated from inner vector index.
        batch = batch.drop_column(DIST_COL)?;
    }

    let vectors = get_column_from_batch(&batch, column)?;

    let validity_buffer = if let Some(rowids) = batch.column_by_name(ROW_ID) {
        NullBuffer::union(rowids.nulls(), vectors.nulls())
    } else {
        vectors.nulls().cloned()
    };

    if distance_work_elements(&vectors) < *FLAT_DISTANCE_INLINE_THRESHOLD {
        return compute_distance_sync(key, dt, vectors, validity_buffer, batch);
    }

    tokio::task::spawn_blocking(move || {
        compute_distance_sync(key, dt, vectors, validity_buffer, batch)
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{FixedSizeListArray, Float32Array, UInt64Array};
    use arrow_schema::{Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;

    fn batch_of(rows: usize, dim: i32) -> RecordBatch {
        let values = Float32Array::from_iter_values((0..rows * dim as usize).map(|i| i as f32));
        let vectors = FixedSizeListArray::try_new_from_values(values, dim).unwrap();
        let row_ids = UInt64Array::from_iter_values(0..rows as u64);
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), true),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(row_ids), Arc::new(vectors)]).unwrap()
    }

    #[test]
    fn test_distance_work_elements() {
        let batch = batch_of(30, 960);
        let vectors = batch.column_by_name("vector").unwrap().clone();
        assert_eq!(distance_work_elements(&vectors), 30 * 960);
        let empty = batch.slice(0, 0);
        let vectors = empty.column_by_name("vector").unwrap().clone();
        assert_eq!(distance_work_elements(&vectors), 0);
    }

    /// The inline and blocking-pool paths compute exactly the same bytes,
    /// so both the default threshold and a forced spawn produce equal output.
    #[tokio::test]
    async fn test_compute_distance_inline_matches_spawned() {
        let dim: i32 = 960;
        let key: ArrayRef = Arc::new(Float32Array::from_iter_values(
            (0..dim).map(|i| (i % 7) as f32 * 0.5),
        ));
        for rows in [1usize, 30, 45, 300] {
            let batch = batch_of(rows, dim);
            let vectors = batch.column_by_name("vector").unwrap().clone();
            let validity = vectors.nulls().cloned();
            let inline = compute_distance_sync(
                key.clone(),
                DistanceType::L2,
                vectors.clone(),
                validity.clone(),
                batch.clone(),
            )
            .unwrap();
            let spawned = tokio::task::spawn_blocking({
                let key = key.clone();
                let batch = batch.clone();
                move || compute_distance_sync(key, DistanceType::L2, vectors, validity, batch)
            })
            .await
            .unwrap()
            .unwrap();
            let via_api = compute_distance(key.clone(), DistanceType::L2, "vector", batch)
                .await
                .unwrap();
            assert_eq!(inline, spawned, "rows={rows}");
            assert_eq!(inline, via_api, "rows={rows}");
            let dists = via_api[DIST_COL].as_primitive::<arrow_array::types::Float32Type>();
            assert_eq!(dists.len(), rows);
            assert!(dists.values().iter().all(|d| d.is_finite()));
        }
    }
}
