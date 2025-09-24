pub const SYSTEM_PROMPT: &str = "You are a helpful Shakespearean writing assistant. Continue the scene in the style of classical theatre while keeping characters consistent.";

pub fn build_prompt(user_prompt: &str) -> String {
    format!(
        "{system}\n\n[USER PROMPT]\n{prompt}\n\n[ASSISTANT RESPONSE]\n",
        system = SYSTEM_PROMPT,
        prompt = user_prompt
    )
}
