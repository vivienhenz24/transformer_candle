use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde_json error: {0}")]
    SerdeJson(#[from] serde_json::Error),

    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),

    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),

    #[error("validation failed: {0}")]
    Validation(String),

    #[error("artifact error: {0}")]
    Artifact(String),
}

pub(crate) fn context<S: Into<String>>(msg: S) -> Error {
    Error::Artifact(msg.into())
}
