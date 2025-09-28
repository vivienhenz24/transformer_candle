use std::{fs, io::Write, path::Path};

pub fn format_markdown_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut table = String::new();
    table.push_str("| ");
    table.push_str(&headers.join(" | "));
    table.push_str(" |\n| ");
    table.push_str(&headers.iter().map(|_| "---").collect::<Vec<_>>().join(" | "));
    table.push_str(" |\n");
    for row in rows {
        table.push_str("| ");
        table.push_str(&row.join(" | "));
        table.push_str(" |\n");
    }
    table
}

pub fn update_results(section: &str, content: &str) -> std::io::Result<()> {
    let section_upper = section.to_uppercase();
    let start_marker = format!("<!-- {}_START -->", section_upper);
    let end_marker = format!("<!-- {}_END -->", section_upper);

    let header = "# Attention Benchmarks\n\n";
    let path = Path::new("crates/attention/benchmarks/RESULTS.md");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut body = if path.exists() {
        fs::read_to_string(path)?
    } else {
        header.to_string()
    };

    let section_block = format!(
        "{start}\n{content}\n{end}\n",
        start = start_marker,
        content = content,
        end = end_marker
    );

    if let (Some(start_idx), Some(end_idx)) = (
        body.find(&start_marker),
        body.find(&end_marker),
    ) {
        let end_bound = end_idx + end_marker.len();
        body.replace_range(start_idx..end_bound, &section_block);
    } else {
        if !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(&section_block);
    }

    let mut file = fs::File::create(path)?;
    file.write_all(body.as_bytes())?;
    Ok(())
}
