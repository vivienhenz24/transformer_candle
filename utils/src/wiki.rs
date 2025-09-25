use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use bzip2::read::MultiBzDecoder;
use html_escape::decode_html_entities;
use once_cell::sync::Lazy;
use quick_xml::{events::Event, reader::Reader};
use regex::Regex;

static REF_TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<ref[^>]*?>.*?</ref>").unwrap());
static REF_SELF_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<ref[^>]*/>").unwrap());
static HTML_COMMENT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<!--.*?-->").unwrap());
static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<[^>]+>").unwrap());
static URL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"https?://\S+").unwrap());
static CATEGORY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?im)^\s*\[\[(Category|File|Image):[^\]]+\]\]\s*$").unwrap());
static TABLE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)\{\|.*?\|\}").unwrap());
static MAGIC_WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"__[^_]+__").unwrap());
static LINK_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]").unwrap());
static HEADING_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*={2,}\s*(.*?)\s*={2,}\s*$").unwrap());
static MULTI_WS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s{2,}").unwrap());

const MIN_ARTICLE_CHARS: usize = 400;

pub fn preprocess_wikipedia_dump<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
) -> Result<()> {
    let input_path = input_path.as_ref().to_owned();
    let output_path = output_path.as_ref().to_owned();

    let input_meta = fs::metadata(&input_path)
        .with_context(|| format!("failed to read metadata for {}", input_path.display()))?;
    if output_path.exists() {
        let output_meta = fs::metadata(&output_path)
            .with_context(|| format!("failed to read metadata for {}", output_path.display()))?;
        if output_meta.modified()? >= input_meta.modified()? {
            return Ok(());
        }
    }

    println!(
        "Preparing Wikipedia dump: {} -> {}",
        input_path.display(),
        output_path.display()
    );

    let file = File::open(&input_path)
        .with_context(|| format!("failed to open {}", input_path.display()))?;
    let decoder = MultiBzDecoder::new(file);
    let mut reader = Reader::from_reader(BufReader::new(decoder));
    reader.trim_text(true);

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let output_file = File::create(&output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let mut writer = BufWriter::new(output_file);

    let max_articles: Option<usize> = std::env::var("WIKI_MAX_ARTICLES")
        .ok()
        .and_then(|v| v.parse().ok());

    let mut buf = Vec::new();
    let mut scratch = Vec::new();
    let mut current_title = String::new();
    let mut current_text = String::new();
    let mut in_page = false;
    let mut is_redirect_page = false;
    let mut processed_articles = 0usize;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"page" => {
                    in_page = true;
                    is_redirect_page = false;
                    current_title.clear();
                    current_text.clear();
                }
                b"title" => {
                    current_title = read_text_block(&mut reader, b"title", &mut scratch)?;
                }
                b"redirect" => {
                    is_redirect_page = true;
                }
                b"text" => {
                    current_text = read_text_block(&mut reader, b"text", &mut scratch)?;
                }
                _ => {}
            },
            Ok(Event::Empty(e)) if e.name().as_ref() == b"redirect" => {
                is_redirect_page = true;
            }
            Ok(Event::End(e)) if e.name().as_ref() == b"page" => {
                if in_page && !is_redirect_page {
                    if let Some(article) = clean_article(&current_title, &current_text) {
                        writer.write_all(article.as_bytes())?;
                        writer.write_all(b"\n\n")?;
                        processed_articles += 1;
                        if processed_articles % 1000 == 0 {
                            println!("Processed {} articles so far...", processed_articles);
                        }
                        if let Some(limit) = max_articles {
                            if processed_articles >= limit {
                                println!(
                                    "Processed {} articles (limit reached).",
                                    processed_articles
                                );
                                writer.flush()?;
                                return Ok(());
                            }
                        }
                    }
                }
                in_page = false;
            }
            Ok(Event::Eof) => {
                println!("Finished preprocessing {} articles.", processed_articles);
                writer.flush()?;
                return Ok(());
            }
            Ok(_) => {}
            Err(err) => return Err(err.into()),
        }
    }
}

fn read_text_block<R: std::io::BufRead>(
    reader: &mut Reader<R>,
    end: &[u8],
    buf: &mut Vec<u8>,
) -> quick_xml::Result<String> {
    buf.clear();
    let mut content = Vec::new();
    loop {
        match reader.read_event_into(buf)? {
            Event::Text(t) => content.extend_from_slice(&t.into_inner()),
            Event::CData(t) => content.extend_from_slice(&t.into_inner()),
            Event::End(e) if e.name().as_ref() == end => break,
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(String::from_utf8_lossy(&content).into_owned())
}

fn clean_article(title: &str, raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if raw
        .trim_start()
        .to_ascii_uppercase()
        .starts_with("#REDIRECT")
    {
        return None;
    }

    let mut text = raw.replace('\r', "");
    text = REF_TAG_RE.replace_all(&text, " ").into_owned();
    text = REF_SELF_RE.replace_all(&text, " ").into_owned();
    text = HTML_COMMENT_RE.replace_all(&text, " ").into_owned();
    text = MAGIC_WORD_RE.replace_all(&text, " ").into_owned();
    text = TABLE_RE.replace_all(&text, " ").into_owned();
    text = strip_templates(&text);
    text = simplify_links(&text);
    text = LINK_RE.replace_all(&text, "$1").into_owned();
    text = REF_TAG_RE.replace_all(&text, " ").into_owned();
    text = REF_SELF_RE.replace_all(&text, " ").into_owned();
    text = TAG_RE.replace_all(&text, " ").into_owned();
    text = URL_RE.replace_all(&text, " ").into_owned();
    text = CATEGORY_RE.replace_all(&text, " ").into_owned();
    text = decode_html_entities(&text).into_owned();
    text = text.replace("'''", "");
    text = text.replace("''", "");
    text = HEADING_RE
        .replace_all(&text, |caps: &regex::Captures| {
            format!("\n{}\n", caps[1].trim())
        })
        .into_owned();

    if let Some(idx) = lower_find(&text, "\n== references") {
        text.truncate(idx);
    }
    if let Some(idx) = lower_find(&text, "\n== external links") {
        text.truncate(idx);
    }
    if let Some(idx) = lower_find(&text, "\n== see also") {
        text.truncate(idx);
    }

    let text = text
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let text = REF_TAG_RE.replace_all(&text, " ").into_owned();
    let text = REF_SELF_RE.replace_all(&text, " ").into_owned();
    let text = LINK_RE.replace_all(&text, "$1").into_owned();
    let text = MULTI_WS_RE.replace_all(&text, " ").into_owned();
    let text = text.trim();

    if text.len() < MIN_ARTICLE_CHARS {
        return None;
    }

    if title.trim().is_empty() {
        Some(text.to_string())
    } else {
        Some(format!("{}\n\n{}", title.trim(), text))
    }
}

fn lower_find(haystack: &str, needle: &str) -> Option<usize> {
    let hay_lower = haystack.to_ascii_lowercase();
    hay_lower.find(&needle.to_ascii_lowercase())
}

fn strip_templates(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut depth = 0i32;
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '{' && chars.peek() == Some(&'{') {
            depth += 1;
            chars.next();
            continue;
        }
        if ch == '}' && chars.peek() == Some(&'}') {
            if depth > 0 {
                depth -= 1;
            }
            chars.next();
            continue;
        }
        if depth == 0 {
            result.push(ch);
        }
    }
    result
}

fn simplify_links(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '[' && chars.peek() == Some(&'[') {
            chars.next();
            let mut link_content = String::new();
            while let Some(next) = chars.next() {
                if next == ']' && chars.peek() == Some(&']') {
                    chars.next();
                    break;
                }
                link_content.push(next);
            }
            if !link_content.is_empty() {
                if let Some(pos) = link_content.rfind('|') {
                    result.push_str(link_content[pos + 1..].trim());
                } else {
                    result.push_str(link_content.trim());
                }
            }
        } else {
            result.push(ch);
        }
    }
    result
}
