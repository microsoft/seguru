#[test]
fn test_use_volatile_load() {
    syntax_host::test_use_volatile_load(0);
    syntax_host::test_use_volatile_load(1234);
}
