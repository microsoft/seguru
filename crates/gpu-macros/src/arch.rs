use proc_macro::TokenStream;
pub(crate) fn replace_asm(input: TokenStream) -> TokenStream {
    let asm_code = input.to_string();
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    let replaced_asm = asm_code.replace(":reg32", ":w").replace(":reg64", ":x"); // adjust float size as needed

    #[cfg(target_arch = "x86_64")]
    let replaced_asm =
        asm_code.replace(":reg16", ":h").replace(":reg32", ":e").replace(":reg64", ":r"); // adjust float size as needed
    replaced_asm.parse().unwrap()
}
