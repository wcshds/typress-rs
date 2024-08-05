#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
pub mod image_processing;
pub mod model;

extern crate alloc;
