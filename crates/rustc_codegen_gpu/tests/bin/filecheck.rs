// src/bin/filecheck_wrapper.rs
use std::env;
use std::path::PathBuf;
use std::process::{Command, exit};

fn extract_string_literal(line: &str) -> String {
    line[line.find("\"").unwrap() + 1..line.len() - 10].to_string()
}

fn unescape_llvm_string(s: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            // We expect two hex digits after '\'
            let hi = chars.next().expect("Expected hex digit after \\");
            if hi == '\\' {
                // If we encounter another backslash, treat it as a literal backslash
                bytes.push(b'\\');
                continue;
            }
            let lo = chars.next().expect("Expected second hex digit after \\");
            let hex_str = format!("{}{}", hi, lo);
            let byte = u8::from_str_radix(&hex_str, 16)
                .unwrap_or_else(|_| panic!("Invalid hex digits {}", hex_str));
            bytes.push(byte);
        } else {
            // Normal ASCII char
            bytes.push(ch as u8);
        }
    }
    bytes
}

fn extract_ptx_check_patterns(file_path: &str) -> Vec<String> {
    let content = std::fs::read_to_string(file_path).expect("Failed to read source file");

    // Collect all lines with `PTX_CHECK: "pattern"`
    content
        .lines()
        .filter_map(|line| {
            // Trim whitespace and check if line contains `PTX_CHECK:`
            let line = line.trim();
            if line.starts_with("// PTX_CHECK:") {
                // Extract the pattern inside quotes, e.g. ".version"
                let parts: Vec<&str> = line.splitn(2, ':').collect();
                if parts.len() == 2 {
                    // parts[1] = ` ".version"` with leading space
                    // trim and remove quotes
                    let pattern = parts[1].trim();
                    // Remove surrounding quotes if present
                    let unquoted = pattern
                        .strip_prefix('"')
                        .and_then(|s| s.strip_suffix('"'))
                        .unwrap_or(pattern);
                    return Some(unquoted.to_string());
                }
            }
            None
        })
        .collect()
}

fn check_ptx_in_llvm_file(rs_path: PathBuf, ll_path: PathBuf) {
    let content = std::fs::read_to_string(&ll_path).expect("Failed to read .ll file");

    let ptx_data = content
        .lines()
        .find(|line| line.contains("@gpu_bin_cst"))
        .map(extract_string_literal)
        .unwrap();
    let bytes = unescape_llvm_string(&ptx_data);

    assert!(ptx_data.contains(".version"));
    assert!(ptx_data.contains(".target"));
    let ptx_path = ll_path.with_extension("ptx");
    std::fs::write(&ptx_path, bytes).expect("Failed to write PTX to file");
    let output = Command::new("cuobjdump")
        .arg("-ptx")
        .arg(&ptx_path)
        .output()
        .map_err(|e| format!("Failed to run cuobjdump: {}", e))
        .unwrap();
    assert!(
        output.status.success(),
        "cuobjdump failed with status: {:?}\nError: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let ptx_str = String::from_utf8_lossy(&output.stdout).to_string();
    std::fs::write(&ptx_path, &ptx_str).expect("Failed to write PTX to file");
    let ptx_check_patterns = extract_ptx_check_patterns(rs_path.to_str().unwrap());
    for pattern in ptx_check_patterns {
        assert!(
            ptx_str.contains(&pattern),
            "PTX does not contain expected pattern `{}` in {:?}",
            pattern,
            ll_path
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    let filecheck_path = "FileCheck"; // or "/usr/bin/FileCheck"
    let status =
        Command::new(filecheck_path).args(&args).status().expect("Failed to execute FileCheck");

    let Some(code) = status.code() else {
        panic!("FileCheck terminated by signal");
    };
    if code != 0 {
        exit(code);
    }
    let mut args = args.iter();
    let mut input_file = None;
    let mut check_file = None;
    while let Some(arg) = args.next() {
        if arg == "--input-file" {
            input_file = args.next();
        } else if arg.starts_with('-') {
            // handle other flags if needed
            continue;
        } else {
            check_file = Some(arg);
        }
    }
    check_ptx_in_llvm_file(
        PathBuf::from(&check_file.unwrap()),
        PathBuf::from(&input_file.unwrap()),
    );
}
