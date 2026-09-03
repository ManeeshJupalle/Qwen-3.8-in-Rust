//! GGUF v3 reader: header, KV metadata (every value type, nested arrays), tensor index, raw reads.
//!
//! Opening a file parses only the header. Bytes are read from the OS in small chunks for the KV
//! section and with exact-size reads for the tensor-info section, and every byte is counted, so
//! `bytes_read_on_open()` proves how far into the file `open` went (it must stay below `data_offset`).
//! Tensor bytes are read only by `read_raw` / `read_into`, which count separately.
//!
//! Format (little-endian): magic "GGUF", u32 version, u64 n_tensors, u64 n_kv, then n_kv pairs of
//! (string key, u32 value type, value), then n_tensors infos of (string name, u32 n_dims, u64 dims[n_dims],
//! u32 ggml type, u64 offset relative to the data section), then padding to `general.alignment`
//! (the GGUF specification defines 32 when the key is absent), then tensor data.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// "GGUF" as a little-endian u32.
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// Alignment used when `general.alignment` is absent (defined by the GGUF specification, not a model default).
pub const SPEC_DEFAULT_ALIGNMENT: u32 = 32;
/// Chunk size for buffered header reads. The tensor-info section is parsed with exact reads, so the
/// only way tensor bytes could be touched on open is a tensor-info section smaller than this chunk;
/// `bytes_read_on_open()` reports it if that ever happens.
const HEADER_CHUNK: usize = 4096;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (magic 0x{0:08x})")]
    BadMagic(u32),
    #[error("unsupported GGUF version {0} (this reader implements version 3)")]
    BadVersion(u32),
    #[error("unknown GGUF value type id {type_id} while reading {context}")]
    BadValueType { type_id: u32, context: String },
    #[error("unknown ggml tensor type id {type_id} for tensor {name}")]
    UnknownGgmlType { type_id: u32, name: String },
    #[error("invalid UTF-8 in {0}")]
    Utf8(String),
    #[error("header truncated while reading {0}")]
    Truncated(String),
    #[error("metadata key missing: {0}")]
    MissingKey(String),
    #[error("metadata key {key} has type {found}, expected {expected}")]
    WrongType { key: String, found: &'static str, expected: &'static str },
    #[error("metadata key {key}: array element {index} has type {found}, expected {expected}")]
    WrongElemType { key: String, index: usize, found: &'static str, expected: &'static str },
    #[error("duplicate tensor name {0}")]
    DuplicateTensor(String),
    #[error("duplicate metadata key {0}")]
    DuplicateKey(String),
    #[error("tensor {name}: element count {n_elements} is not a multiple of block size {block_size} for {ggml_type:?}")]
    BadElementCount { name: String, n_elements: u64, block_size: u64, ggml_type: GgmlType },
    #[error("tensor {name}: dimension count {0} exceeds the GGUF maximum of 4", .n_dims)]
    TooManyDims { name: String, n_dims: u32 },
    #[error("tensor not found: {0}")]
    NoTensor(String),
    #[error("tensor {name}: absolute offset {offset} is not a multiple of {align}")]
    Misaligned { name: String, offset: u64, align: u64 },
    #[error("tensor {name}: buffer has {got} bytes, tensor has {want}")]
    BufferSize { name: String, got: usize, want: u64 },
    #[error("tensor {name}: data range {start}..{end} exceeds file size {file_size}")]
    OutOfFile { name: String, start: u64, end: u64, file_size: u64 },
    #[error("alignment {0} is not a power of two")]
    BadAlignment(u32),
}

pub type Result<T> = std::result::Result<T, GgufError>;

/// GGUF metadata value type ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    Str = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ValueType {
    fn from_id(id: u32, context: &str) -> Result<Self> {
        Ok(match id {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::Str,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return Err(GgufError::BadValueType { type_id: id, context: context.to_string() }),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "UINT8",
            Self::I8 => "INT8",
            Self::U16 => "UINT16",
            Self::I16 => "INT16",
            Self::U32 => "UINT32",
            Self::I32 => "INT32",
            Self::F32 => "FLOAT32",
            Self::Bool => "BOOL",
            Self::Str => "STRING",
            Self::Array => "ARRAY",
            Self::U64 => "UINT64",
            Self::I64 => "INT64",
            Self::F64 => "FLOAT64",
        }
    }
}

/// A metadata value. Arrays keep their declared element type and may nest.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array { elem_type: ValueType, items: Vec<Value> },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::U8(_) => ValueType::U8,
            Value::I8(_) => ValueType::I8,
            Value::U16(_) => ValueType::U16,
            Value::I16(_) => ValueType::I16,
            Value::U32(_) => ValueType::U32,
            Value::I32(_) => ValueType::I32,
            Value::F32(_) => ValueType::F32,
            Value::Bool(_) => ValueType::Bool,
            Value::Str(_) => ValueType::Str,
            Value::Array { .. } => ValueType::Array,
            Value::U64(_) => ValueType::U64,
            Value::I64(_) => ValueType::I64,
            Value::F64(_) => ValueType::F64,
        }
    }

    pub fn type_name(&self) -> &'static str {
        self.value_type().name()
    }
}

/// ggml tensor data types (ids as written in GGUF tensor infos).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_id(id: u32, name: &str) -> Result<Self> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            _ => return Err(GgufError::UnknownGgmlType { type_id: id, name: name.to_string() }),
        })
    }

    pub fn id(self) -> u32 {
        self as u32
    }

    /// (elements per block, bytes per block), as in ggml's type traits.
    pub fn block_layout(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2_K => (256, 84),
            Self::Q3_K => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
            Self::IQ2_XXS => (256, 66),
            Self::IQ2_XS => (256, 74),
            Self::IQ3_XXS => (256, 98),
            Self::IQ1_S => (256, 50),
            Self::IQ4_NL => (32, 18),
            Self::IQ3_S => (256, 110),
            Self::IQ2_S => (256, 82),
            Self::IQ4_XS => (256, 136),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::F64 => (1, 8),
            Self::IQ1_M => (256, 56),
            Self::BF16 => (1, 2),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1_M => "IQ1_M",
            Self::BF16 => "BF16",
        }
    }
}

/// One entry of the tensor index. `shape` is in GGUF order (`shape[0]` is the fastest-varying
/// dimension, the reverse of numpy order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: GgmlType,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub byte_size: u64,
    /// Offset relative to the start of the data section, as stored in the file.
    pub rel_offset: u64,
    /// `data_offset + rel_offset`.
    pub absolute_file_offset: u64,
}

impl TensorInfo {
    pub fn shape_numpy_order(&self) -> Vec<u64> {
        self.shape.iter().rev().copied().collect()
    }
    pub fn end_offset(&self) -> u64 {
        self.absolute_file_offset + self.byte_size
    }
}

/// Where a layer's tensors sit in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSpan {
    pub layer: u32,
    pub tensor_count: usize,
    pub start: u64,
    pub end: u64,
    pub sum_bytes: u64,
    /// `end - start - sum_bytes`.
    pub gap_bytes: u64,
    /// Largest gap between two consecutive tensors of the layer (file order).
    pub max_single_gap: u64,
    /// Tensors of other layers or non-layer tensors whose offset lies inside `start..end`.
    pub foreign_inside: usize,
    /// The layer's tensors occupy consecutive positions in file order.
    pub adjacent_in_file_order: bool,
    /// `adjacent_in_file_order && foreign_inside == 0 && max_single_gap < alignment`, as in tools/gguf_index.py.
    pub contiguous: bool,
}

/// Counted, chunked reader for the header. Never reads past what the parser asks for in exact mode.
struct HeaderReader {
    file: File,
    buf: Vec<u8>,
    start: usize,
    end: usize,
    total_read: u64,
    consumed: u64,
    exact: bool,
}

impl HeaderReader {
    fn new(file: File) -> Self {
        HeaderReader { file, buf: Vec::new(), start: 0, end: 0, total_read: 0, consumed: 0, exact: false }
    }

    fn available(&self) -> usize {
        self.end - self.start
    }

    fn ensure(&mut self, n: usize, what: &str) -> Result<()> {
        if self.available() >= n {
            return Ok(());
        }
        if self.start > 0 {
            self.buf.copy_within(self.start..self.end, 0);
            self.end -= self.start;
            self.start = 0;
        }
        let need = n - self.end;
        let want = if self.exact { need } else { need.max(HEADER_CHUNK) };
        if self.buf.len() < self.end + want {
            self.buf.resize(self.end + want, 0);
        }
        let mut got = 0usize;
        while got < need {
            let k = self.file.read(&mut self.buf[self.end + got..self.end + want])?;
            if k == 0 {
                return Err(GgufError::Truncated(what.to_string()));
            }
            got += k;
        }
        self.end += got;
        self.total_read += got as u64;
        Ok(())
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&[u8]> {
        self.ensure(n, what)?;
        let s = &self.buf[self.start..self.start + n];
        self.start += n;
        self.consumed += n as u64;
        Ok(s)
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        Ok(self.take(1, what)?[0])
    }
    fn u16(&mut self, what: &str) -> Result<u16> {
        let b = self.take(2, what)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self, what: &str) -> Result<u32> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self, what: &str) -> Result<u64> {
        let b = self.take(8, what)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
    fn string(&mut self, what: &str) -> Result<String> {
        let len = self.u64(what)? as usize;
        let bytes = self.take(len, what)?.to_vec();
        String::from_utf8(bytes).map_err(|_| GgufError::Utf8(what.to_string()))
    }

    fn value(&mut self, vt: ValueType, what: &str) -> Result<Value> {
        Ok(match vt {
            ValueType::U8 => Value::U8(self.u8(what)?),
            ValueType::I8 => Value::I8(self.u8(what)? as i8),
            ValueType::U16 => Value::U16(self.u16(what)?),
            ValueType::I16 => Value::I16(self.u16(what)? as i16),
            ValueType::U32 => Value::U32(self.u32(what)?),
            ValueType::I32 => Value::I32(self.u32(what)? as i32),
            ValueType::F32 => Value::F32(f32::from_bits(self.u32(what)?)),
            ValueType::Bool => Value::Bool(self.u8(what)? != 0),
            ValueType::Str => Value::Str(self.string(what)?),
            ValueType::U64 => Value::U64(self.u64(what)?),
            ValueType::I64 => Value::I64(self.u64(what)? as i64),
            ValueType::F64 => Value::F64(f64::from_bits(self.u64(what)?)),
            ValueType::Array => {
                let elem_type = ValueType::from_id(self.u32(what)?, what)?;
                let count = self.u64(what)? as usize;
                let mut items = Vec::with_capacity(count.min(1 << 24));
                for _ in 0..count {
                    items.push(self.value(elem_type, what)?);
                }
                Value::Array { elem_type, items }
            }
        })
    }
}

/// An opened GGUF file: metadata and tensor index, no tensor data.
pub struct Gguf {
    path: PathBuf,
    file: File,
    file_size: u64,
    version: u32,
    alignment: u32,
    header_end: u64,
    data_offset: u64,
    bytes_read_on_open: u64,
    tensor_bytes_read: AtomicU64,
    kv: Vec<(String, Value)>,
    kv_index: HashMap<String, usize>,
    tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    layers: BTreeMap<u32, Vec<usize>>,
}

impl Gguf {
    /// Parse the header and build the tensor index. Reads no tensor data.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_size = file.metadata()?.len();
        let mut r = HeaderReader::new(file);

        let magic = r.u32("magic")?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = r.u32("version")?;
        if version != 3 {
            return Err(GgufError::BadVersion(version));
        }
        let n_tensors = r.u64("tensor_count")? as usize;
        let n_kv = r.u64("kv_count")? as usize;

        let mut kv: Vec<(String, Value)> = Vec::with_capacity(n_kv);
        let mut kv_index = HashMap::with_capacity(n_kv);
        for i in 0..n_kv {
            let key = r.string(&format!("metadata key #{i}"))?;
            let vt = ValueType::from_id(r.u32(&key)?, &key)?;
            let v = r.value(vt, &key)?;
            if kv_index.insert(key.clone(), kv.len()).is_some() {
                return Err(GgufError::DuplicateKey(key));
            }
            kv.push((key, v));
        }

        // Tensor infos: exact reads only, so the counted bytes stop at the end of the header.
        r.exact = true;
        let mut raw_infos: Vec<(String, Vec<u64>, u32, u64)> = Vec::with_capacity(n_tensors);
        for i in 0..n_tensors {
            let name = r.string(&format!("tensor info #{i}"))?;
            let n_dims = r.u32(&name)?;
            if n_dims > 4 {
                return Err(GgufError::TooManyDims { name, n_dims });
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.u64(&name)?);
            }
            let type_id = r.u32(&name)?;
            let offset = r.u64(&name)?;
            raw_infos.push((name, dims, type_id, offset));
        }
        let header_end = r.consumed;
        let bytes_read_on_open = r.total_read;
        let file = r.file;

        let alignment = match kv_index.get("general.alignment") {
            Some(&i) => match &kv[i].1 {
                Value::U32(a) => *a,
                other => {
                    return Err(GgufError::WrongType {
                        key: "general.alignment".into(),
                        found: other.type_name(),
                        expected: "UINT32",
                    })
                }
            },
            None => SPEC_DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::BadAlignment(alignment));
        }
        let data_offset = header_end.div_ceil(alignment as u64) * alignment as u64;

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut tensor_index = HashMap::with_capacity(n_tensors);
        for (name, shape, type_id, rel_offset) in raw_infos {
            let ggml_type = GgmlType::from_id(type_id, &name)?;
            let element_count: u64 = shape.iter().product();
            let (block_size, type_size) = ggml_type.block_layout();
            if !element_count.is_multiple_of(block_size) {
                return Err(GgufError::BadElementCount { name, n_elements: element_count, block_size, ggml_type });
            }
            let byte_size = element_count / block_size * type_size;
            let absolute_file_offset = data_offset + rel_offset;
            if absolute_file_offset + byte_size > file_size {
                return Err(GgufError::OutOfFile {
                    name,
                    start: absolute_file_offset,
                    end: absolute_file_offset + byte_size,
                    file_size,
                });
            }
            if tensor_index.insert(name.clone(), tensors.len()).is_some() {
                return Err(GgufError::DuplicateTensor(name));
            }
            tensors.push(TensorInfo { name, ggml_type, shape, element_count, byte_size, rel_offset, absolute_file_offset });
        }

        let mut layers: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, t) in tensors.iter().enumerate() {
            if let Some(l) = layer_of(&t.name) {
                layers.entry(l).or_default().push(i);
            }
        }

        Ok(Gguf {
            path,
            file,
            file_size,
            version,
            alignment,
            header_end,
            data_offset,
            bytes_read_on_open,
            tensor_bytes_read: AtomicU64::new(0),
            kv,
            kv_index,
            tensors,
            tensor_index,
            layers,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn file_size(&self) -> u64 {
        self.file_size
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn alignment(&self) -> u32 {
        self.alignment
    }
    /// Byte offset just past the last tensor info (before alignment padding).
    pub fn header_end(&self) -> u64 {
        self.header_end
    }
    /// Start of the tensor data section.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }
    /// Bytes read from the OS by `open` (buffered chunks included).
    pub fn bytes_read_on_open(&self) -> u64 {
        self.bytes_read_on_open
    }
    /// Bytes of tensor data read so far through `read_raw` / `read_into`.
    pub fn tensor_bytes_read(&self) -> u64 {
        self.tensor_bytes_read.load(Ordering::Relaxed)
    }

    // ---- metadata -------------------------------------------------------------------------

    pub fn kv(&self) -> &[(String, Value)] {
        &self.kv
    }

    pub fn get(&self, key: &str) -> Result<&Value> {
        self.kv_index
            .get(key)
            .map(|&i| &self.kv[i].1)
            .ok_or_else(|| GgufError::MissingKey(key.to_string()))
    }

    pub fn has(&self, key: &str) -> bool {
        self.kv_index.contains_key(key)
    }

    pub fn get_u32(&self, key: &str) -> Result<u32> {
        match self.get(key)? {
            Value::U32(v) => Ok(*v),
            other => Err(GgufError::WrongType { key: key.into(), found: other.type_name(), expected: "UINT32" }),
        }
    }
    pub fn get_i32(&self, key: &str) -> Result<i32> {
        match self.get(key)? {
            Value::I32(v) => Ok(*v),
            other => Err(GgufError::WrongType { key: key.into(), found: other.type_name(), expected: "INT32" }),
        }
    }
    pub fn get_f32(&self, key: &str) -> Result<f32> {
        match self.get(key)? {
            Value::F32(v) => Ok(*v),
            other => Err(GgufError::WrongType { key: key.into(), found: other.type_name(), expected: "FLOAT32" }),
        }
    }
    pub fn get_bool(&self, key: &str) -> Result<bool> {
        match self.get(key)? {
            Value::Bool(v) => Ok(*v),
            other => Err(GgufError::WrongType { key: key.into(), found: other.type_name(), expected: "BOOL" }),
        }
    }
    pub fn get_str(&self, key: &str) -> Result<&str> {
        match self.get(key)? {
            Value::Str(v) => Ok(v),
            other => Err(GgufError::WrongType { key: key.into(), found: other.type_name(), expected: "STRING" }),
        }
    }
    pub fn get_array(&self, key: &str) -> Result<(ValueType, &[Value])> {
        match self.get(key)? {
            Value::Array { elem_type, items } => Ok((*elem_type, items)),
            other => Err(GgufError::WrongType { key: key.into(), found: other.type_name(), expected: "ARRAY" }),
        }
    }
    pub fn get_array_i32(&self, key: &str) -> Result<Vec<i32>> {
        let (_, items) = self.get_array(key)?;
        items
            .iter()
            .enumerate()
            .map(|(i, v)| match v {
                Value::I32(x) => Ok(*x),
                other => Err(GgufError::WrongElemType {
                    key: key.into(),
                    index: i,
                    found: other.type_name(),
                    expected: "INT32",
                }),
            })
            .collect()
    }
    pub fn get_array_str(&self, key: &str) -> Result<Vec<&str>> {
        let (_, items) = self.get_array(key)?;
        items
            .iter()
            .enumerate()
            .map(|(i, v)| match v {
                Value::Str(s) => Ok(s.as_str()),
                other => Err(GgufError::WrongElemType {
                    key: key.into(),
                    index: i,
                    found: other.type_name(),
                    expected: "STRING",
                }),
            })
            .collect()
    }

    // ---- tensor index ---------------------------------------------------------------------

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor_index
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| GgufError::NoTensor(name.to_string()))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }

    /// Layer indices present in the file (from `blk.N.` names), ascending.
    pub fn layer_ids(&self) -> Vec<u32> {
        self.layers.keys().copied().collect()
    }

    /// Tensors of layer `n`, in file order.
    pub fn layer_tensors(&self, n: u32) -> Vec<&TensorInfo> {
        let mut v: Vec<&TensorInfo> = self
            .layers
            .get(&n)
            .map(|ix| ix.iter().map(|&i| &self.tensors[i]).collect())
            .unwrap_or_default();
        v.sort_by_key(|t| t.absolute_file_offset);
        v
    }

    /// Tensors that belong to no `blk.N.` layer, in file order.
    pub fn non_layer_tensors(&self) -> Vec<&TensorInfo> {
        let mut v: Vec<&TensorInfo> = self.tensors.iter().filter(|t| layer_of(&t.name).is_none()).collect();
        v.sort_by_key(|t| t.absolute_file_offset);
        v
    }

    /// Span and contiguity of layer `n` (same definition as tools/gguf_index.py).
    pub fn layer_span(&self, n: u32) -> Option<LayerSpan> {
        let ts = self.layer_tensors(n);
        if ts.is_empty() {
            return None;
        }
        let start = ts[0].absolute_file_offset;
        let end = ts.iter().map(|t| t.end_offset()).max().unwrap();
        let sum_bytes: u64 = ts.iter().map(|t| t.byte_size).sum();
        let max_single_gap = ts.windows(2).map(|w| w[1].absolute_file_offset.saturating_sub(w[0].end_offset())).max().unwrap_or(0);
        let mut by_offset: Vec<&TensorInfo> = self.tensors.iter().collect();
        by_offset.sort_by_key(|t| t.absolute_file_offset);
        let foreign_inside = by_offset
            .iter()
            .filter(|t| t.absolute_file_offset >= start && t.absolute_file_offset < end && layer_of(&t.name) != Some(n))
            .count();
        let positions: Vec<usize> = by_offset
            .iter()
            .enumerate()
            .filter(|(_, t)| layer_of(&t.name) == Some(n))
            .map(|(i, _)| i)
            .collect();
        let adjacent = positions.last().unwrap() - positions[0] + 1 == positions.len();
        let contiguous = adjacent && foreign_inside == 0 && max_single_gap < self.alignment as u64;
        Some(LayerSpan {
            layer: n,
            tensor_count: ts.len(),
            start,
            end,
            sum_bytes,
            gap_bytes: end - start - sum_bytes,
            max_single_gap,
            foreign_inside,
            adjacent_in_file_order: adjacent,
            contiguous,
        })
    }

    // ---- raw reads -----------------------------------------------------------------------

    /// Read a tensor's bytes into a new buffer.
    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>> {
        let size = self.tensor(name)?.byte_size as usize;
        let mut buf = vec![0u8; size];
        self.read_into(name, &mut buf)?;
        Ok(buf)
    }

    /// Read a tensor's bytes into `buf`, which must be exactly `byte_size` long.
    /// Asserts the absolute offset is 32-byte aligned (the layout the streaming tier relies on).
    pub fn read_into(&self, name: &str, buf: &mut [u8]) -> Result<()> {
        let t = self.tensor(name)?;
        if buf.len() as u64 != t.byte_size {
            return Err(GgufError::BufferSize { name: name.into(), got: buf.len(), want: t.byte_size });
        }
        let align = 32u64.max(self.alignment as u64);
        if t.absolute_file_offset % align != 0 {
            return Err(GgufError::Misaligned { name: name.into(), offset: t.absolute_file_offset, align });
        }
        let mut f = &self.file;
        f.seek(SeekFrom::Start(t.absolute_file_offset))?;
        f.read_exact(buf)?;
        self.tensor_bytes_read.fetch_add(buf.len() as u64, Ordering::Relaxed);
        Ok(())
    }
}

/// Layer index of a `blk.N.<suffix>` tensor name.
pub fn layer_of(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (num, _) = rest.split_once('.')?;
    num.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_names() {
        assert_eq!(layer_of("blk.0.attn_q.weight"), Some(0));
        assert_eq!(layer_of("blk.64.nextn.eh_proj.weight"), Some(64));
        assert_eq!(layer_of("token_embd.weight"), None);
        assert_eq!(layer_of("blk.x.foo"), None);
        assert_eq!(layer_of("blk.7"), None);
    }

    #[test]
    fn unknown_ggml_type_is_an_error() {
        let e = GgmlType::from_id(999, "t").unwrap_err();
        assert!(e.to_string().contains("999"));
        assert!(e.to_string().contains('t'));
    }
}
