pub(crate) fn println_fmt_to_prinf_fmt(fmt: &str, holders: &[String]) -> Result<String, String> {
    let mut positions = Vec::new();
    let mut result = String::new();
    let mut last_idx = 0;
    let mut holder_iter = holders.iter();

    let mut chars = fmt.char_indices();
    while let Some((idx, c)) = chars.next() {
        if c == '\\' {
            chars.next();
            continue;
        } else if c == '{' {
            if let Some((next_idx, next_c)) = chars.clone().next() {
                if next_c == '}' {
                    // Record the position of "{}"
                    positions.push(idx);

                    // Add substring before "{}"
                    result.push_str(&fmt[last_idx..idx]);

                    // Replace with correct format
                    let holder = holder_iter.next().ok_or("More placeholders than inputs")?;
                    result.push_str(holder);
                    // Skip the '}'
                    chars.next();
                    last_idx = next_idx + 1;
                } else {
                    return Err("Currently we only support '{}' as placeholder".into());
                }
            }
        }
    }

    // Add the rest of the string
    result.push_str(&fmt[last_idx..]);

    if holder_iter.next().is_some() {
        return Err("More values than placeholders".into());
    }

    Ok(result)
}
