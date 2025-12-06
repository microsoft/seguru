## Initial setup: Add a rust/compiler/rustc_codegen_ssa to the repo.

1. Get a filtered rust repo as rustc_codegen_ssa

```bash
git init rust-tmp
cd rust-tmp
git remote add origin https://github.com/rust-lang/rust.git
git sparse-checkout init --no-cone
git sparse-checkout set compiler/rustc_codegen_ssa
git fetch origin main
git checkout 3f550237 -b 1.87
git filter-repo --path compiler/rustc_codegen_ssa --force
```

2. Add the rustc_codegen_ssa as a subtree into seguru repo.

```bash
cd ../seguru
git remote add rustc_codegen_ssa ~/rust-temp
git fetch rustc_codegen_ssa
git subtree add --prefix=rust rustc_codegen_ssa 1.87 --squash --squash-commit
```

rust/compiler/rustc_codegen_ssa is the only folder we will use. We patch rust/compiler/rustc_codegen_ssa folder in seguru repo direct.


### How to update to a new rust version

1. Update rustc_codegen_ssa repo using sparse checkout.

If 6501e64fc is the version you want to update to, do

```bash
git init rust-tmp
cd rust-tmp
git remote add origin https://github.com/rust-lang/rust.git
git sparse-checkout init --no-cone
git sparse-checkout set compiler/rustc_codegen_ssa
git fetch origin main
git checkout 6501e64fc -b new
git filter-repo --path compiler/rustc_codegen_ssa --force
```

After `git checkout 6501e64fc -b new`, we only see compiler/rustc_codegen_ssa, but other files are hidden,
and so must call `git filter-repo --path compiler/rustc_codegen_ssa --force` so thath the later subtree pull
will not get other files.

Now the rustc_codegen_ssa repo is update to date.

2. Update subtree in seguru

```bash
cd seguru
git remote add rustc_codegen_ssa ~/rust-tmp
git subtree pull --prefix=rust rustc_codegen_ssa new --squash
```