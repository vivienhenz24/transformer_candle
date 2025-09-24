#[derive(Debug, Default)]
pub struct MemoryProfile {
    pub bytes: Vec<u64>,
}

impl MemoryProfile {
    pub fn track(&mut self, bytes: u64) {
        self.bytes.push(bytes);
    }
}
