use super::context_aware::ContextAwareCorpus;

#[derive(Debug, Clone)]
pub struct MergeHeuristics {
    pub alphabet: Vec<String>,
}

impl MergeHeuristics {
    pub fn from_corpus(corpus: &ContextAwareCorpus) -> Self {
        let mut alphabet = std::collections::BTreeSet::new();
        for segment in corpus.segments() {
            for ch in segment.chars() {
                alphabet.insert(ch.to_string());
            }
        }
        Self {
            alphabet: alphabet.into_iter().collect(),
        }
    }
}
