pub mod checks;
pub mod error;

pub use checks::validate_model;
pub use error::{BuildError, ValidationError};
