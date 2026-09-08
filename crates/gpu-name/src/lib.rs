// Convert path name to a valid symbol
pub fn convert_def_path_to_gpu_sym_name(name: &str) -> String {
    // Runs on every kernel launch, so it fills one buffer rather than
    // allocating a String per character.
    let mut out = String::with_capacity(name.len() * 2);
    for c in name.chars() {
        if c.is_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            // `c as u8` truncates and `{:X}` does not pad, and the symbols the
            // code generator emits depend on both.
            let b = c as u8;
            out.push('_');
            if b >= 16 {
                out.push(HEX[(b >> 4) as usize]);
            }
            out.push(HEX[(b & 0xF) as usize]);
            out.push('_');
        }
    }
    out
}

const HEX: [char; 16] =
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'];

#[cfg(test)]
mod tests {
    fn reference(name: &str) -> String {
        name.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' {
                    c.to_string()
                } else {
                    format!("_{:X}_", c as u8)
                }
            })
            .collect()
    }

    #[test]
    fn matches_reference() {
        let mut all: String = (0u32..=0x2FF).filter_map(char::from_u32).collect();
        all.push_str("gpusorting_gpu::onesweep::onesweep_scan<a::{{closure}}::_GPU_TMP_CONFIG>");
        for s in [all.as_str(), "", "_", "abc123"] {
            assert_eq!(super::convert_def_path_to_gpu_sym_name(s), reference(s));
        }
    }
}
