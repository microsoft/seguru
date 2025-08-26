// Convert path name to a valid symbol
pub fn convert_def_path_to_gpu_sym_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c.to_string() } else { format!("_{:X}_", c as u8) })
        .collect()
}

pub fn convert_type_name_to_gpu_sym_name(name: &str) -> String {
    let path = name.replace("<", "::<");
    convert_def_path_to_gpu_sym_name(&path)
}
