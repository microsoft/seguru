// src/bin/filecheck_wrapper.rs
use std::env;
use std::path::PathBuf;
use std::process::{Command, exit};

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

fn check_ptx_in_llvm_file(rs_path: PathBuf, ptx_path: PathBuf) {
    let ptx_str = std::fs::read_to_string(&ptx_path).expect("Failed to read PTX file");
    let ptx_check_patterns = extract_ptx_check_patterns(rs_path.to_str().unwrap());
    for pattern in ptx_check_patterns {
        assert!(
            ptx_str.contains(&pattern),
            "PTX does not contain expected pattern `{}` in {:?}",
            pattern,
            ptx_path
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
    let ptx_file = input_file
        .as_ref()
        .and_then(|f| f.strip_suffix(".ll").map(|s| format!("{}.ptx", s)))
        .expect("Input file must have a .ll extension");
    check_ptx_in_llvm_file(PathBuf::from(&check_file.unwrap()), PathBuf::from(&ptx_file));
}
