pub mod attention;
pub mod architectures;
pub mod generation;
pub mod layers;
pub mod memory;

pub use attention::adaptive_window::{AdaptiveWindowAttention, AdaptiveWindowConfig};
pub use attention::cascade_attention::{CascadeAttentionComposer, CascadeAttentionConfig, CrossLayerState};
pub use attention::hierarchical::HierarchicalAttention;
pub use attention::sparse_patterns::{PatternSpec, SparseAttentionMask};
pub use architectures::adaptive_depth::{AdaptiveDepthConfig, AdaptiveDepthController, DepthDecision, TokenDepthUsage};
pub use architectures::cascade::{
    CascadeBlock,
    CascadeTransformer,
    CascadeTransformerBuilder,
    CascadeTransformerConfig,
};
pub use architectures::progressive::{ProgressiveGenerationConfig, ProgressiveRefiner};
pub use generation::adaptive_sampling::{AdaptiveSampler, AdaptiveSamplingConfig};
pub use generation::confidence_based::{ConfidenceGuidedSampler, ConfidenceSettings};
pub use generation::creative_modes::{CreativeMode, CreativePalette};
pub use layers::embedding::{CascadeEmbeddingConfig, CascadeEmbeddings};

pub mod prelude {
    pub use crate::attention::adaptive_window::{AdaptiveWindowAttention, AdaptiveWindowConfig};
    pub use crate::attention::cascade_attention::{CascadeAttentionComposer, CascadeAttentionConfig, CrossLayerState};
    pub use crate::attention::hierarchical::HierarchicalAttention;
    pub use crate::attention::sparse_patterns::{PatternSpec, SparseAttentionMask};
    pub use crate::architectures::adaptive_depth::{AdaptiveDepthConfig, AdaptiveDepthController, DepthDecision, TokenDepthUsage};
    pub use crate::architectures::cascade::{CascadeBlock, CascadeTransformer, CascadeTransformerBuilder, CascadeTransformerConfig};
    pub use crate::architectures::progressive::{ProgressiveGenerationConfig, ProgressiveRefiner};
    pub use crate::generation::adaptive_sampling::{AdaptiveSampler, AdaptiveSamplingConfig};
    pub use crate::generation::creative_modes::{CreativeMode, CreativePalette};
    pub use crate::layers::activation::{ActivationRegistry, ActivationStrategy};
    pub use crate::layers::feed_forward::{CascadeFeedForward, FeedForwardConfig};
    pub use crate::layers::normalization::{CascadeNorm, NormStrategy};
}
