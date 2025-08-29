use std::collections::btree_map::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rustc_ast::Crate;
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::{Compiler, Config};
use rustc_middle::query::queries::registered_tools;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{CrateType, ExternEntry, ExternLocation, Externs};
use rustc_session::utils::{CanonicalizedPath, NativeLib, NativeLibKind};
use tracing::debug;

const GPU_SUFFIX: &str = "gpu";
const GPU_MACROS_CRATE: &str = "gpu_macros";
const GPU_BC_EXT: &str = "gpu.bc";
const GPU_LIB_EXT: &str = "gpu.a";
const CODEGEN_TARGET_ENV_VAR: &str = "__CODEGEN_TARGET__";

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CompilerStage {
    #[default]
    CpuOrCheckGPU = 0,
    GpuForGpu,
    GpuForCpu,
}

impl TryFrom<String> for CompilerStage {
    type Error = ();

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "cpu_or_gpu" => Ok(CompilerStage::CpuOrCheckGPU),
            "gpu2gpu" => Ok(CompilerStage::GpuForGpu),
            "gpu2cpu" => Ok(CompilerStage::GpuForCpu),
            _ => Err(()),
        }
    }
}

impl From<CompilerStage> for String {
    fn from(value: CompilerStage) -> Self {
        match value {
            CompilerStage::CpuOrCheckGPU => "gpu_or_cpu".into(),
            CompilerStage::GpuForGpu => "gpu2gpu".into(),
            CompilerStage::GpuForCpu => "gpu2cpu".into(),
        }
    }
}

pub const GPU_MAIN_RUST: &str = r#"
#[allow(dead_code)]
fn main() {}
"#;

fn parse_item_by_str(
    psess: &rustc_session::parse::ParseSess,
    input: &str,
) -> Option<rustc_ast::ptr::P<rustc_ast::Item>> {
    rustc_parse::new_parser_from_source_str(
        psess,
        rustc_span::FileName::Custom("gpu-tmp.rs".into()),
        input.into(),
    )
    .expect("failed to create parser")
    .parse_item(rustc_parse::parser::ForceCollect::No)
    .expect("cannot parse as item")
}

#[derive(Default)]
pub(crate) struct GpuOrCpuRustCallback {
    pub stage: CompilerStage,
    pub next_stage: Option<CompilerStage>,
}

fn config_env_target_for_macro(target: &str) {
    // Tell gpu_macros the target to build
    unsafe {
        std::env::set_var(CODEGEN_TARGET_ENV_VAR, target);
    }
}

fn new_gpu_dir(p: &Path) -> PathBuf {
    let p = p.join(GPU_SUFFIX);
    if !p.exists() {
        std::fs::create_dir_all(&p)
            .unwrap_or_else(|_| panic!("failed to create dir {}", p.display()));
    }
    p
}

fn new_externs_with_new_path<F: FnMut(&Path) -> Option<PathBuf>>(
    externs: &Externs,
    mut new_path: F,
) -> Externs {
    let mut new_extern_data = BTreeMap::<String, ExternEntry>::new();
    for (name, e) in externs.iter() {
        let mut new_e = e.clone();
        match &mut new_e.location {
            ExternLocation::FoundInLibrarySearchDirectories => {
                new_extern_data.insert(name.clone(), new_e);
            }
            ExternLocation::ExactPaths(paths) => {
                *paths = paths
                    .iter()
                    .filter_map(|p| {
                        let path = p.canonicalized();
                        new_path(path).map(|p| CanonicalizedPath::new(&p))
                    })
                    .collect();
                new_e.force = true;
                new_extern_data.insert(name.clone(), new_e);
            }
        }
    }
    Externs::new(new_extern_data)
}

#[allow(clippy::type_complexity)]
static REGISTERED_TOOLS: Mutex<
    Option<for<'tcx> fn(TyCtxt<'tcx>, ()) -> registered_tools::ProvidedValue<'tcx>>,
> = Mutex::new(None);

fn gpu_register_tool<'tcx>(tcx: TyCtxt<'tcx>, id: ()) -> registered_tools::ProvidedValue<'tcx> {
    let mut registered_tools = REGISTERED_TOOLS.lock().unwrap().unwrap()(tcx, id);
    registered_tools.insert(rustc_span::Ident::new(
        rustc_span::Symbol::intern("gpu_codegen"),
        rustc_span::Ident::dummy().span,
    ));
    registered_tools
}

fn config_link_gpu_code(config: &mut Config) {
    // Link to gpu code
    let mut need_link_gpu = config.opts.test;

    for crate_type in &mut config.opts.crate_types {
        if !matches!(crate_type, CrateType::Executable) {
            continue;
        }
        need_link_gpu = true;
    }

    if !need_link_gpu {
        return;
    }

    let mut bc_files: Vec<PathBuf> = Vec::new();
    let _ = new_externs_with_new_path(&config.opts.externs, |p| {
        let fname = p.file_stem().unwrap().to_str().unwrap();
        assert!(p.exists());
        let new_path =
            new_gpu_dir(p.parent().unwrap()).join(format!("{fname}{GPU_SUFFIX}.{GPU_BC_EXT}"));
        if new_path.exists() {
            bc_files.push(new_path)
        }
        None
    });

    if let Some(output_dir) = &config.output_dir {
        let crate_name = config.opts.crate_name.as_ref().unwrap().clone();
        let bc_file = new_gpu_dir(output_dir)
            .join(format!("lib{crate_name}{GPU_SUFFIX}.{GPU_BC_EXT}"))
            .with_extension(GPU_BC_EXT);
        if bc_file.exists() {
            bc_files.push(bc_file.clone());
        }
        debug!("link gpu bc_files {:?}", bc_files);
        if !bc_files.is_empty() {
            let gpu_obj_file = bc_file.with_extension(GPU_LIB_EXT);
            mlir_compile::CompileConfig::new()
                .gpu_link_and_create_static_lib(&bc_files, &gpu_obj_file)
                .expect("failed to compile gpu lib");
            assert!(gpu_obj_file.exists());
            debug!("link to {:?}", gpu_obj_file);
            config.opts.libs.push(NativeLib {
                name: gpu_obj_file.as_os_str().to_str().unwrap().to_string(),
                new_name: None,
                kind: NativeLibKind::LinkArg,
                verbatim: None,
            });
        }
    }
}

impl Callbacks for GpuOrCpuRustCallback {
    fn config(&mut self, config: &mut Config) {
        match self.stage {
            CompilerStage::CpuOrCheckGPU => {
                config_env_target_for_macro("CPU");
                return;
            }
            CompilerStage::GpuForGpu => {
                config_env_target_for_macro("GPU");
            }
            CompilerStage::GpuForCpu => {
                config_env_target_for_macro("CPU");
                config_link_gpu_code(config);
                return;
            }
        }

        //config.opts.lint_opts.push(("dead_code".into(), rustc_lint_defs::Level::Allow));

        // Change to GPU codegen backend, this does not work well
        // When using rustc_codegen_gpu here, it has a conflict with original
        // LLVM passes and cause error
        // 'Pass 'Lower Garbage Collection Instructions' is not initialized.'
        // Don't know to solve it and so use config.opts.unstable_opts.codegen_backend.
        /* config.make_codegen_backend =
        Some(Box::new(|_options| rustc_codegen_gpu::__rustc_codegen_backend()));*/

        // Change backend via codegen_backend dylib path
        //let codegen_path = std::env::var("GPU_CODEGEN").unwrap_or_default();
        config.opts.unstable_opts.codegen_backend =
            Some(crate::codegen::get_codegen_dylib(config.output_dir.as_ref().unwrap()));

        // Do not inline mir which is necessary to allow cross-crate
        // GPU kernel function monomorphization.
        // Example:
        /*
        // crate A:
        #[no_mangle]
        fn dummy_kernel_call() {
            kernel_call::<1024>();
        }
        // crate B:
        #[gpu_macros::kernel]
        fn kernel_call<const N: usize>() {
            gpu::printf("this is from gpu call with %d", N);
        }
        */
        // Without this, crate A will not see kernel_call::<1024> in
        // its mono_items, and instead will inline it in dummy_kernel_call.
        // This may change some translation behavior
        // config.opts.unstable_opts.inline_mir = Some(false);
        // inline_mir_threshold is a default threshold for all functions.
        // inline_mir_hint_threshold becomes a relaxed threshold for fn if the fn has inline hint
        // Thus, function marked as inline will still be inlined when needed.
        config.opts.unstable_opts.inline_mir_threshold = Some(0);

        // Enable tool features
        config.override_queries = Some(|_sess, providers| {
            // Save old provider if you want to call it
            REGISTERED_TOOLS.lock().unwrap().replace(providers.registered_tools);

            providers.queries.registered_tools = gpu_register_tool;
        });

        config.opts.cg.extra_filename = format!("{}{GPU_SUFFIX}", config.opts.cg.extra_filename);
        config.opts.cg.metadata.iter_mut().for_each(|m| *m = format!("{m}{GPU_SUFFIX}"));
        config.opts.cg.codegen_units = Some(1);
        config.output_dir.iter_mut().for_each(|p| *p = new_gpu_dir(p));
        config.opts.incremental.iter_mut().for_each(|p| *p = new_gpu_dir(p));
        config.opts.json_artifact_notifications = false;

        let new_extern_file = |p: &Path| {
            //assert!(p.exists(), "{} not found in {:?}", p.display(), config.opts.crate_name);
            let fname = p.file_stem().unwrap().to_str().unwrap();
            let ext = p.extension().map_or("", |e| e.to_str().unwrap());
            let new_path = new_gpu_dir(p.parent().unwrap())
                .join(format!("{fname}{GPU_SUFFIX}"))
                .with_extension(ext);

            if new_path.exists() { Some(new_path) } else { Some(p.with_extension(ext)) }
        };
        let mut gpu_search_dirs = Vec::new();
        config.opts.search_paths.iter_mut().for_each(|s| {
            if s.kind == rustc_session::search_paths::PathKind::Dependency {
                gpu_search_dirs.push(new_gpu_dir(&s.dir));
                let fkeys = s
                    .files
                    .query("", "")
                    .unwrap()
                    .filter(|(_, p)| new_extern_file(&p.path) == Some(p.path.to_path_buf()))
                    .map(|(k, _)| k)
                    .collect::<Vec<_>>();
                let retain_fkeys: Vec<&str> = fkeys.iter().map(|k| k.as_str()).collect();
                s.files.retain(&retain_fkeys);
            }
        });
        config.opts.search_paths.extend(gpu_search_dirs.iter().map(|p| {
            rustc_session::search_paths::SearchPath::new(
                rustc_session::search_paths::PathKind::Dependency,
                p.clone(),
            )
        }));
        //config.opts.output_types = rustc_session::config::OutputTypes::new();

        for crate_type in &mut config.opts.crate_types {
            if !matches!(crate_type, CrateType::Executable) {
                continue;
            }
            //config.opts.crate_name.iter_mut().for_each(|n| *n = format!("{}_gpu", n));
            *crate_type = CrateType::Rlib;
        }
        config.opts.externs = new_externs_with_new_path(&config.opts.externs, new_extern_file);
    }

    fn after_crate_root_parsing(&mut self, compiler: &Compiler, krate: &mut Crate) -> Compilation {
        if matches!(self.stage, CompilerStage::GpuForGpu) {
            krate.items.iter_mut().for_each(|item| {
                if item.ident.name == rustc_span::Symbol::intern("main") {
                    let tmp_item = parse_item_by_str(&compiler.sess.psess, GPU_MAIN_RUST)
                        .expect("failed to create parser");
                    item.attrs.extend(tmp_item.attrs.clone());
                }
            });
        }

        Compilation::Continue
    }

    fn after_expansion<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        match &self.stage {
            CompilerStage::CpuOrCheckGPU => {
                let cstore = tcx.cstore_untracked();
                for cnum in tcx.crates(()) {
                    let name = cstore.crate_name(*cnum);
                    if name.as_str() == GPU_MACROS_CRATE {
                        self.next_stage = Some(CompilerStage::GpuForGpu);
                        return Compilation::Stop;
                    }
                }
                Compilation::Continue
            }
            CompilerStage::GpuForGpu => {
                self.next_stage = Some(CompilerStage::GpuForCpu);
                Compilation::Continue
            }
            CompilerStage::GpuForCpu => Compilation::Continue,
        }
    }
}
