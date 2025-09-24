#[derive(Debug, Clone, Default)]
pub struct PreprocessorChain;

impl PreprocessorChain {
    pub fn run(&self, text: &str) -> String {
        let normalized = text.replace('\r', "");
        normalized
            .split('\n')
            .map(|line| line.trim_end())
            .collect::<Vec<_>>()
            .join("\n")
    }
}
