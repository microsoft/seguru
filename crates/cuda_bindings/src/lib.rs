// Suppress the flurry of warnings caused by using "C" naming conventions
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub const GPU_MEMCPY_D2H: u8 = 0;
pub const GPU_MEMCPY_H2D: u8 = 1;

pub struct GPUConfig {
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
    pub shared_mem_bytes: u32,
}

// This matches bindgen::Builder output
include!("./bindings.rs");

// Safe wrappers
pub fn init() -> u32 {
    let ret;
    unsafe {
        ret = gpu_init() as u32;
    }
    ret
}

pub fn memalloc<'a, T>(size: usize) -> Option<&'a mut [T]> {
    let ptr;
    let ret;
    if (size % std::mem::size_of::<T>()) != 0 {
        return None;
    }
    unsafe {
        ptr = gpu_memalloc(size) as *mut T;
        ret = std::slice::from_raw_parts_mut(ptr, size / std::mem::size_of::<T>());
    }
    Some(ret)
}

pub fn memcpy<'a, T>(dst: &'a mut [T], src: &'a [T], size: usize, h_to_d: u8) -> u32 {
    let ret;
    unsafe {
        ret = gpu_memcpy(
            dst.as_mut_ptr() as *mut ::std::os::raw::c_void,
            src.as_ptr() as *const ::std::os::raw::c_void,
            size,
            h_to_d,
        ) as u32;
    }
    ret
}

pub fn free<T>(src: &mut [T]) -> u32 {
    let ret;
    unsafe {
        ret = gpu_free(src.as_mut_ptr() as *mut ::std::os::raw::c_void) as u32;
    }
    ret
}

pub fn device_sync() -> u32 {
    let ret;
    unsafe {
        ret = gpu_device_sync() as u32;
    }
    ret
}

pub fn load_module() -> u32 {
    let ret;
    unsafe {
        ret = gpu_load_module() as u32;
    }
    ret
}

pub fn unload_module() -> u32 {
    let ret;
    unsafe {
        ret = gpu_unload_module() as u32;
    }
    ret
}

// Deliberately don't export launch kernel as safe. We will generate a wrapper for it.
