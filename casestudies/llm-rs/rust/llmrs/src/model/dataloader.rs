use std::fs::File;
use std::path::Path;
use std::slice;
use std::slice::ChunksExact;
use std::sync::Arc;

use memmap2::Mmap;

const HEADER_LEN: usize = 256;
pub fn parse_header_data<'a, T>(
    mmap: &Mmap,
    magic_number: i32,
    version: i32,
) -> (Vec<i32>, &'a [T]) {
    // Read model header
    const HEADER_BYTES: usize = HEADER_LEN * 4;
    let model_header = mmap[0..HEADER_BYTES]
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    assert!(model_header.len() == HEADER_BYTES / 4);
    // Check magic number and version
    assert!(model_header[0] == magic_number, "Bad magic model file {}", model_header[0]);
    assert!(model_header[1] == version, "Bad version in model file {}", model_header[1]);
    let len = mmap.len();
    let bytes = &mmap[HEADER_BYTES..];
    (model_header, unsafe {
        slice::from_raw_parts(bytes.as_ptr() as *const T, (len - HEADER_BYTES) / size_of::<T>())
    })
}

pub struct DataLoader<'a> {
    // ----------------------------------------------------------------------------
    // Hyperparameters
    // ----------------------------------------------------------------------------
    /// Batch size
    pub batch_size: usize,

    /// Sequence length
    pub seq_len: usize,

    // ----------------------------------------------------------------------------
    // Input handling and its state
    // ----------------------------------------------------------------------------
    /// File for tokens
    file_mmap: Arc<Mmap>,
    tokens: Option<&'a [u16]>,
    tokens_chunks: Option<ChunksExact<'a, u16>>,
    // ----------------------------------------------------------------------------
    // Convenience variables
    // ----------------------------------------------------------------------------
    pub size_per_batch: usize,
    pub num_batches: usize,
    pub num_tokens: usize,
}

impl<'a> DataLoader<'a> {
    /// Creates a new DataLoader instance.
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the tokens file.
    /// * `batch` - Batch size.
    /// * `T` - Sequence length.
    ///
    /// # Returns
    ///
    /// A new `DataLoader` instance.
    pub fn new(filename: &Path, batch_size: usize, seq_len: usize) -> Self {
        let f = File::open(filename).expect("Error opening tokens file");
        let file_mmap = unsafe { Mmap::map(&f).expect("Error mapping tokens file") };
        let size_per_batch = batch_size * seq_len + 1;
        let mut loader = DataLoader {
            batch_size,
            seq_len,
            file_mmap: Arc::new(file_mmap),
            tokens: None,
            tokens_chunks: None,
            size_per_batch,
            num_batches: 0,
            num_tokens: 0,
        };
        let (header, tokens) = parse_header_data::<u16>(&loader.file_mmap.clone(), 20240520, 1);
        let ntoks = header[2] as usize;
        assert!(ntoks > 0);
        assert!(tokens.len() == ntoks, "{} != {}", tokens.len(), ntoks);
        loader.tokens = Some(tokens);
        loader.num_tokens = ntoks;
        let tokens_chunks = loader.tokens.as_ref().unwrap().chunks_exact(size_per_batch);
        let num_batches = tokens_chunks.len();
        loader.tokens_chunks = Some(tokens_chunks);
        loader.num_batches = num_batches;
        loader
    }

    /// Resets the DataLoader to start from the beginning of the file.
    pub fn reset(&mut self) {
        self.tokens_chunks = Some(self.tokens.as_ref().unwrap().chunks_exact(self.size_per_batch));
    }

    /// Loads the next batch of data into the DataLoader's memory.
    pub fn next_batch<'ctx: 'a>(&mut self) -> (Vec<i32>, Vec<i32>) {
        if self.tokens_chunks.as_ref().unwrap().len() == 0 {
            self.reset();
        }
        let current_tokens = self.tokens_chunks.as_mut().unwrap().next().unwrap();
        let current_tokens = if current_tokens.len() < self.size_per_batch {
            self.reset();
            self.tokens_chunks.as_mut().unwrap().next().unwrap()
        } else {
            current_tokens
        };
        // convert u16 to i32
        let inputs =
            current_tokens[..self.size_per_batch - 1].iter().map(|&x| x as i32).collect::<Vec<_>>();
        let targets = current_tokens[1..].iter().map(|&x| x as i32).collect::<Vec<_>>();
        (inputs, targets)
    }
}
