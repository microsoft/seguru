// Suppress the flurry of warnings caused by using "C" naming conventions
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub const GPU_MEMCPY_D2H: u8 = 0;
pub const GPU_MEMCPY_H2D: u8 = 1;

// This matches bindgen::Builder output
include!("./bindings.rs");
