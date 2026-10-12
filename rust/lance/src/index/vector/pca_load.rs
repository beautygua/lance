// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Load-time checks for a PCA pre-transform.

use std::sync::Arc;

use arrow_schema::DataType;
use lance_core::cache::{CacheCodecImpl, CacheEntryReader, CacheEntryWriter};
use lance_core::deepsize::DeepSizeOf;
use lance_index::pb::PcaParameters;
use lance_index::vector::VectorIndex;
use lance_index::vector::pca::{PcaModel, decode_pca_model, encode_pca_model};
use lance_index::vector::pretransform::PreTransformIndex;
use lance_index::vector::quantizer::Quantizer;
use lance_index::{INDEX_FILE_NAME, IndexType};
use lance_table::format::IndexMetadata;

use super::ivf::v2::IvfStateEntryBox;

use super::details::{derive_vector_index_type, pca_params_from_details};
use super::{PcaComponentMismatch, pca_component_mismatch};
use crate::{Error, Result};

pub(crate) fn format_pca_fingerprint(fingerprint: &[u8; 32]) -> String {
    fingerprint
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(crate) fn manifest_pca(index_meta: &IndexMetadata) -> Result<Option<(usize, [u8; 32])>> {
    let (pca, details_index_type) = match index_meta.index_details.as_deref() {
        Some(details) => (
            pca_params_from_details(details)?
                .map(|(params, fingerprint)| (params.dimension, fingerprint)),
            Some(derive_vector_index_type(details)),
        ),
        None => (None, None),
    };
    match (index_meta.index_version, pca) {
        (3, Some(pca)) if details_index_type.as_deref() == Some("IVF_RQ") => Ok(Some(pca)),
        (3, Some(_)) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "manifest index_version=3 details index_type actual={details_index_type:?} expected=IVF_RQ for index {}",
                index_meta.uuid
            ),
        )),
        (3, None) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "manifest index_version=3 requires index_details with PCA parameters and a 32-byte fingerprint for index {}",
                index_meta.uuid
            ),
        )),
        (version, Some(_)) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "manifest index_version={version} must not declare PCA details for index {}",
                index_meta.uuid
            ),
        )),
        (_, None) => Ok(None),
    }
}

pub(crate) fn validate_pca_manifest_and_model(
    index_meta: &IndexMetadata,
    physical_index_type: Option<&str>,
    model: Option<&Arc<PcaModel>>,
) -> Result<()> {
    let manifest = manifest_pca(index_meta)?;
    if let Some(index_type) = physical_index_type
        && (index_meta.index_version == 3 || model.is_some())
        && index_type != "IVF_RQ"
    {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={} physical index_type={index_type} contains lance:pca; expected index_type=IVF_RQ",
                index_meta.index_version
            ),
        ));
    }
    match (index_meta.index_version, manifest, model) {
        (3, Some((manifest_dimension, expected_fingerprint)), Some(model)) => {
            let actual_fingerprint = model.fingerprint();
            if actual_fingerprint != expected_fingerprint {
                return Err(Error::corrupt_file_named(
                    INDEX_FILE_NAME,
                    format!(
                        "index_version=3 lance:pca fingerprint mismatch: expected={}, actual={}",
                        format_pca_fingerprint(&expected_fingerprint),
                        format_pca_fingerprint(&actual_fingerprint)
                    ),
                ));
            }
            if model.output_dimension() != manifest_dimension {
                return Err(Error::corrupt_file_named(
                    INDEX_FILE_NAME,
                    format!(
                        "index_version=3 PCA output_dimension actual={} expected={} from manifest details",
                        model.output_dimension(),
                        manifest_dimension
                    ),
                ));
            }
            Ok(())
        }
        (3, Some(_), None) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            "index_version=3 requires physical lance:pca metadata and model buffer",
        )),
        (version, None, Some(_)) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={version} must not contain physical lance:pca metadata or model buffer"
            ),
        )),
        (_, None, None) => Ok(()),
        (version, Some(_), _) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "manifest index_version={version} has an invalid PCA declaration for index {}",
                index_meta.uuid
            ),
        )),
    }
}

pub(crate) fn validate_pca_index_dimensions(
    index_meta: &IndexMetadata,
    model: Option<&Arc<PcaModel>>,
    index: &Arc<dyn VectorIndex>,
    expected_input_dimension: usize,
) -> Result<()> {
    let Some(model) = model else {
        return Ok(());
    };
    if index.index_type() != IndexType::IvfRq {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={} PCA index type actual={:?} expected=IvfRq",
                index_meta.index_version,
                index.index_type()
            ),
        ));
    }
    let ivf = index.ivf_model();
    let centroids = ivf.centroids_array().ok_or_else(|| {
        Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={} PCA index is missing IVF centroids",
                index_meta.index_version
            ),
        )
    })?;
    let quantizer = index.quantizer();
    let Quantizer::Rabit(rq) = quantizer else {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={} PCA quantizer actual={} expected=Rabit",
                index_meta.index_version,
                quantizer.quantization_type()
            ),
        ));
    };
    validate_pca_shape(
        index_meta.index_version,
        model,
        expected_input_dimension,
        ivf.dimension(),
        &centroids.value_type(),
        rq.metadata_ref().rotated_dim(),
    )
}

pub(crate) fn validate_pca_shape(
    index_version: i32,
    model: &PcaModel,
    expected_input_dimension: usize,
    ivf_dimension: usize,
    centroid_type: &DataType,
    rq_rotated_dimension: usize,
) -> Result<()> {
    match pca_component_mismatch(
        model,
        expected_input_dimension,
        ivf_dimension,
        centroid_type,
        rq_rotated_dimension,
    ) {
        None => Ok(()),
        Some(PcaComponentMismatch::InputDimension { actual, expected }) => {
            Err(Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!(
                    "index_version={index_version} PCA input_dimension actual={actual} expected={expected} from dataset field"
                ),
            ))
        }
        Some(PcaComponentMismatch::CentroidType { actual }) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={index_version} IVF centroid dtype actual={actual} expected=Float32"
            ),
        )),
        Some(PcaComponentMismatch::OutputDimension { actual, expected }) => {
            Err(Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!(
                    "index_version={index_version} PCA output_dimension actual={actual} expected={expected} from IVF centroids"
                ),
            ))
        }
        Some(PcaComponentMismatch::RqDimension { actual, expected }) => {
            Err(Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!(
                    "index_version={index_version} RQ metadata rotated_dim actual={actual} expected={expected} from PCA output_dimension"
                ),
            ))
        }
    }
}

pub(crate) fn validate_logical_pca_fingerprints<'a>(
    index_name: &str,
    segments: impl IntoIterator<Item = &'a IndexMetadata>,
) -> Result<()> {
    let mut expected = None;
    for (position, metadata) in segments.into_iter().enumerate() {
        let actual = manifest_pca(metadata)?.map(|(_, fingerprint)| fingerprint);
        if position == 0 {
            expected = actual;
        } else if actual != expected {
            return Err(Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!(
                    "logical index {index_name} has inconsistent PCA fingerprints across segments: expected={:?}, segment={} actual={:?}",
                    expected.map(|fingerprint| format_pca_fingerprint(&fingerprint)),
                    metadata.uuid,
                    actual.map(|fingerprint| format_pca_fingerprint(&fingerprint))
                ),
            ));
        }
    }
    Ok(())
}

/// Wraps a loaded IVF index with its pre-transform and checks the pair.
pub(crate) fn attach_loaded_pre_transform(
    index: Arc<dyn VectorIndex>,
    model: Option<&Arc<PcaModel>>,
    index_meta: &IndexMetadata,
    input_dimension: usize,
) -> Result<Arc<dyn VectorIndex>> {
    let index = match model {
        Some(model) => {
            Arc::new(PreTransformIndex::new_pca(index, (*model).clone())) as Arc<dyn VectorIndex>
        }
        None => index,
    };
    validate_pca_index_dimensions(index_meta, model, &index, input_dimension)?;
    Ok(index)
}

/// Cached IVF state plus the pre-transform that must be reattached on open.
///
/// The IVF cache entry itself stays free of pre-transform bytes.
pub(crate) struct PcaIndexState {
    pub(crate) inner: IvfStateEntryBox,
    pub(crate) model: Arc<PcaModel>,
}

impl DeepSizeOf for PcaIndexState {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.inner.deep_size_of_children(context) + self.model.deep_size_of_children(context)
    }
}

impl CacheCodecImpl for PcaIndexState {
    const TYPE_ID: &'static str = "lance.vector.pretransform.PcaState";
    const CURRENT_VERSION: u32 = 1;

    fn serialize(&self, w: &mut CacheEntryWriter<'_>) -> Result<()> {
        let header = PcaParameters {
            dimension: u32::try_from(self.model.output_dimension()).map_err(|_| {
                Error::io(format!(
                    "PCA cache output_dimension={} does not fit in u32",
                    self.model.output_dimension()
                ))
            })?,
            model_fingerprint: self.model.fingerprint().to_vec(),
            ..Default::default()
        };
        let model_bytes = encode_pca_model(self.model.as_ref())?;
        let mut inner_bytes = Vec::new();
        self.inner
            .serialize(&mut CacheEntryWriter::new(&mut inner_bytes))?;
        w.write_header(&header)?;
        w.write_raw(&model_bytes)?;
        w.write_raw(&inner_bytes)?;
        Ok(())
    }

    fn deserialize(r: &mut CacheEntryReader<'_>) -> Result<Self> {
        let version = r.version();
        let header: PcaParameters = r.read_header()?;
        let model_bytes = r.read_raw().map_err(|error| {
            Error::io(format!(
                "PCA cache model blob read failed: version={version}, blob_len=unavailable, fingerprint={:?}: {error}",
                header.model_fingerprint
            ))
        })?;
        let blob_len = model_bytes.len();
        let fingerprint = header.model_fingerprint.as_slice();
        if fingerprint.is_empty() {
            return Err(Error::io(format!(
                "PCA cache metadata mismatch: version={version}, blob_len={blob_len}, fingerprint=None: non-empty blob is missing fingerprint"
            )));
        }
        if fingerprint.len() != 32 {
            return Err(Error::io(format!(
                "PCA cache fingerprint length mismatch: version={version}, blob_len={blob_len}, fingerprint_len={}, fingerprint={fingerprint:?}, expected=32",
                fingerprint.len()
            )));
        }
        if model_bytes.is_empty() {
            return Err(Error::io(format!(
                "PCA cache metadata mismatch: version={version}, blob_len={blob_len}, fingerprint_len={}, fingerprint={fingerprint:?}: empty blob has fingerprint",
                fingerprint.len()
            )));
        }
        let model = decode_pca_model(&model_bytes).map_err(|error| {
            Error::io(format!(
                "PCA cache PCA decode failed: version={version}, blob_len={blob_len}, fingerprint_len={}, fingerprint={fingerprint:?}: {error}",
                fingerprint.len()
            ))
        })?;
        let actual_fingerprint = model.fingerprint();
        if fingerprint != actual_fingerprint.as_slice() {
            return Err(Error::io(format!(
                "PCA cache fingerprint mismatch: version={version}, blob_len={blob_len}, fingerprint_len={}, expected={fingerprint:?}, actual={actual_fingerprint:?}",
                fingerprint.len()
            )));
        }
        let inner_bytes = r.read_raw().map_err(|error| {
            Error::io(format!(
                "PCA cache inner IVF blob read failed: version={version}, fingerprint={fingerprint:?}: {error}"
            ))
        })?;
        let inner = IvfStateEntryBox::deserialize(&mut CacheEntryReader::new(
            &inner_bytes,
            0,
            IvfStateEntryBox::CURRENT_VERSION,
        ))?;
        Ok(Self {
            inner,
            model: Arc::new(model),
        })
    }
}
