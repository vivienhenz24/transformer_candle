#[derive(Debug, Clone)]
pub struct ContextAwareCorpus {
    segments: Vec<String>,
}

impl ContextAwareCorpus {
    pub fn from_text(text: &str) -> Self {
        let mut segments = Vec::new();
        for chunk in text.split('\n') {
            let chunk = chunk.trim();
            if chunk.is_empty() {
                continue;
            }
            segments.push(chunk.to_string());
        }
        if segments.is_empty() {
            segments.push(text.to_string());
        }
        Self { segments }
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.segments.iter().map(|s| s.as_str())
    }

    pub fn segments(&self) -> &[String] {
        &self.segments
    }
}
