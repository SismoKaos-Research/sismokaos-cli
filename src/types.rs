/// Represents an event emitted by the background processing pipeline.
///
/// The pipeline streams, so the number of windows is not known up front. Progress is
/// reported in input records instead, which are all indexed before processing starts.
#[derive(Debug)]
pub enum PipelineEvent {
    /// Started processing the input directory
    FolderStarted(String),
    /// Input records consumed so far, out of the total indexed
    Progress {
        done: usize,
        total: usize,
    },
    /// Successfully computed and wrote a single window
    ChunkProcessed(usize),
    WindowProcessed,
    /// All windows have been written
    Completed,
    /// Non-fatal error or warning during processing
    Warning(String),
    /// Pipeline finished completely
    Finished,
}
