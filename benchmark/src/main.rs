/// Run all benchmarks first:
///   CRITERION_HOME=results/criterion DISABLE_GPU_BOUND_CHECK=false cargo bench --features llvm
///   CRITERION_HOME=results/criterion DISABLE_GPU_BOUND_CHECK=true cargo bench --features seguru
///   CRITERION_HOME=results/criterion DISABLE_GPU_BOUND_CHECK=false cargo bench --features nvvm --features seguru
/// Then generate the comparison:
///   cargo r --bin benchmark -- results/criterion results/criterion-no-check compare
use std::collections::HashMap;
use std::env::args;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, serde::Deserialize)]
struct BenchmarkJson {
    mean: Mean,
}

#[derive(Debug, Deserialize)]
struct Mean {
    point_estimate: f64, // in nanoseconds
}

/// Read a criterion directory and collect results.
/// `rename_map` remaps config prefix names (e.g. "rs" -> "rs-wbound").
fn read_criterion_dir(
    criterion_dir: &Path,
    rename_map: &HashMap<&str, &str>,
    results: &mut HashMap<String, HashMap<String, HashMap<String, f64>>>,
) {
    if !criterion_dir.exists() {
        eprintln!("Criterion directory not found: {:?}", criterion_dir);
        return;
    }
    for entry in fs::read_dir(criterion_dir).unwrap() {
        let entry = entry.unwrap();
        if !entry.file_type().unwrap().is_dir() {
            continue;
        }
        let bench_name = entry.file_name().into_string().unwrap().trim().to_string();
        let bench_dir = entry.path();
        for entry in fs::read_dir(&bench_dir).unwrap() {
            let entry = entry.unwrap();
            if !entry.file_type().unwrap().is_dir() {
                continue;
            }
            let config = entry.file_name().into_string().unwrap();
            let new_json_path = entry.path().join("new/estimates.json");
            if !new_json_path.exists() {
                continue;
            }
            let content = fs::read_to_string(&new_json_path).unwrap();
            let json: BenchmarkJson = serde_json::from_str(&content).unwrap();
            let mean_ns = json.mean.point_estimate;
            let config_parts: Vec<&str> = config.split("_").collect();
            assert!(config_parts.len() >= 4);
            let (t, o, v) = (config_parts[1], config_parts[2], config_parts[3]);
            let raw_prefix = config_parts[0].to_string();
            // Apply rename map
            let prefix = if let Some(&mapped) = rename_map.get(raw_prefix.as_str()) {
                mapped.to_string()
            } else {
                continue; // skip configs not in the rename map
            };
            let thread_count = config_parts[8..12]
                .iter()
                .map(|s| s.parse::<usize>().unwrap())
                .product::<usize>();
            let t_o_v = format!("{}_{}_{}_{}", t, o, v, thread_count);

            // Clean up benchmark name
            let bench_name_clean = bench_name
                .replace("_kernel", "")
                .replace("_", "-")
                .replace("backward", "bwd")
                .replace("forward", "fwd")
                .replace("back", "bwd")
                .replace("_bench", "");

            let sub = results.entry(bench_name_clean).or_default();
            let times = sub.entry(t_o_v.clone()).or_default();
            if times.contains_key(&prefix) {
                eprintln!("Warning: duplicate entry for {} {} {}", bench_name, t_o_v, prefix);
            }
            times.insert(prefix, mean_ns);
        }
    }
}

const CARGO_MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
fn main() {
    let cargo_dir = CARGO_MANIFEST_DIR;
    let check_dir = args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/results/criterion", cargo_dir));
    let output_name = args().nth(3).unwrap_or_else(|| "compare".to_string());

    let mut results: HashMap<String, HashMap<String, HashMap<String, f64>>> = HashMap::new();

    // From criterion (with bounds check): rs -> rs-wbound, nvvm, llvm, and their empty variants
    let check_renames: HashMap<&str, &str> = [
        ("rs", "rs-wbound"),
        ("rsnobound", "rs-nobound"),
        ("nvvm", "nvvm"),
        ("llvm", "llvm"),
        ("emptyrs", "empty-rs"),
        ("emptynvvm", "empty-nvvm"),
        ("emptyllvm", "empty-llvm"),
    ]
    .into_iter()
    .collect();
    read_criterion_dir(Path::new(&check_dir), &check_renames, &mut results);

    // From criterion-no-check use only the empty baseline.
    let nocheck_renames: HashMap<&str, &str> = [("emptyrs", "empty-rs")].into_iter().collect();
    read_criterion_dir(Path::new(&check_dir), &nocheck_renames, &mut results);

    let mut file = BufWriter::new(
        std::fs::File::create(format!("{}.csv", output_name))
            .expect("failed to create csv"),
    );
    writeln!(
        file,
        "Benchmark\tT\tO\tV\tThreads\trs-wbound\trs-nobound\tnvvm\tllvm\trs-empty\tnvvm-empty\tllvm-empty\trs-wbound-n/nvvm-n\trs-nobound-n/nvvm-n\tllvm-n/nvvm-n"
    )
    .unwrap();
    println!(
        "{:<20} {:<14} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>20} {:>20} {:>16}",
        "Benchmark",
        "T_O_V_Th",
        "rs-wbound",
        "rs-nobound",
        "nvvm",
        "llvm",
        "rs-empty",
        "nvvm-empty",
        "llvm-empty",
        "rswb-n/nvvm-n",
        "rsnb-n/nvvm-n",
        "llvm-n/nvvm-n",
    );
    let mut keys: Vec<_> = results.keys().map(|k| k.to_string()).collect();
    keys.sort();
    let mut latex_data: HashMap<String, Vec<(f64, f64, f64)>> = HashMap::new();
    for bench_name in &keys {
        let sub_results = &results[bench_name];
        let mut sub_keys = sub_results.keys().collect::<Vec<_>>();
        sub_keys.sort_by(|a, b| {
            let len_cmp = a.len().cmp(&b.len());
            if len_cmp == std::cmp::Ordering::Equal { a.cmp(b) } else { len_cmp }
        });

        for t_o in sub_keys {
            let times = &sub_results[t_o];
            let get = |key: &str| times.get(key).copied().unwrap_or(0.0);

            let rs_wb_time = get("rs-wbound");
            let rs_nb_time = get("rs-nobound");
            let nvvm_time = get("nvvm");
            let llvm_time = get("llvm");
            let empty_rs = get("empty-rs");
            let empty_nvvm = get("empty-nvvm");
            let empty_llvm = get("empty-llvm");

            let t_o_parts: Vec<usize> =
                t_o.split("_").map(|s| s.parse::<usize>().unwrap()).collect();
            let (t, o, v, threads) = (t_o_parts[0], t_o_parts[1], t_o_parts[2], t_o_parts[3]);

            let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };

            let rs_wb_norm = rs_wb_time - empty_rs;
            let rs_nb_norm = rs_nb_time - empty_rs;
            let nvvm_norm = nvvm_time - empty_nvvm;
            let llvm_norm = llvm_time - empty_llvm;

            let rs_wb_norm_to_nvvm = ratio(rs_wb_norm, nvvm_norm);
            let rs_nb_norm_to_nvvm = ratio(rs_nb_norm, nvvm_norm);
            let llvm_norm_to_nvvm = ratio(llvm_norm, nvvm_norm);

            println!(
                "{:<20} {:<14} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>20.2} {:>20.2} {:>16.2}",
                &bench_name[..std::cmp::min(20, bench_name.len())],
                t_o,
                rs_wb_time,
                rs_nb_time,
                nvvm_time,
                llvm_time,
                empty_rs,
                empty_nvvm,
                empty_llvm,
                rs_wb_norm_to_nvvm,
                rs_nb_norm_to_nvvm,
                llvm_norm_to_nvvm,
            );
            writeln!(
                file,
                "{bench}\t{t}\t{o}\t{v}\t{threads}\t{rs_wb:.2}\t{rs_nb:.2}\t{nvvm:.2}\t{llvm:.2}\t{rs_empty:.2}\t{nvvm_empty:.2}\t{llvm_empty:.2}\t{rs_wb_norm_nvvm:.2}\t{rs_nb_norm_nvvm:.2}\t{llvm_norm_nvvm:.2}",
                bench = &bench_name[..std::cmp::min(18, bench_name.len())],
                t = t,
                o = o,
                v = v,
                threads = threads,
                rs_wb = rs_wb_time,
                rs_nb = rs_nb_time,
                nvvm = nvvm_time,
                llvm = llvm_time,
                rs_empty = empty_rs,
                nvvm_empty = empty_nvvm,
                llvm_empty = empty_llvm,
                rs_wb_norm_nvvm = rs_wb_norm_to_nvvm,
                rs_nb_norm_nvvm = rs_nb_norm_to_nvvm,
                llvm_norm_nvvm = llvm_norm_to_nvvm,
            )
            .unwrap();
            latex_data
                .entry(bench_name.to_string())
                .or_default()
                .push((rs_wb_norm_to_nvvm, rs_nb_norm_to_nvvm, llvm_norm_to_nvvm));
        }
    }

    drop(file);

    // Generate LaTeX data
    let mut latexfile = BufWriter::new(
        std::fs::File::create(format!("{}.tex", output_name))
            .expect("failed to create result.tex"),
    );
    let max_runs = latex_data.values().map(|v| v.len()).max().unwrap_or(0);
    let coords = keys.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(",");

    writeln!(latexfile, "\\newcommand{{\\xsymbols}}{{ symbolic x coords={{ {} }}}}", coords)
        .unwrap();

    for run_index in 0..max_runs {
        let seq = [1024, 16384, 1048576][run_index];
        for (label, idx) in [("rs-wbound", 0), ("rs-nobound", 1), ("llvm", 2)] {
            let data = latex_data
                .iter()
                .map(|(bench, values)| {
                    if run_index >= values.len() {
                        "".to_string()
                    } else {
                        let val = match idx {
                            0 => values[run_index].0,
                            1 => values[run_index].1,
                            _ => values[run_index].2,
                        };
                        format!("({},{})", bench, val)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            writeln!(
                latexfile,
                "\\pgfkeyssetvalue{{/ratio/{label}/{seq}}}{{\n{data}\n}}",
            )
            .unwrap();
        }
    }
}
