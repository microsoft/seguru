#[test]
fn test_raw_eq_zero() {
    syntax_host::test_raw_eq_zero(0, 0.0, 0.0, 1);
    syntax_host::test_raw_eq_zero(1, 0.0, 0.0, 0);
    syntax_host::test_raw_eq_zero(0, 0.1, 0.0, 0);
    syntax_host::test_raw_eq_zero(0, 0.0, 0.001, 0);
}
