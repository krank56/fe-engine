#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(target_os = "macos")]
pub mod matrix_free;

#[cfg(target_os = "macos")]
pub use metal::MetalPCG;

#[cfg(target_os = "macos")]
pub use matrix_free::MatrixFreeGPU;
