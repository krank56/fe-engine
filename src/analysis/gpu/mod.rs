#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(target_os = "macos")]
pub use metal::MetalPCG;
