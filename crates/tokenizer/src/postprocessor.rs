use crate::config::PostCfg;
use crate::errors::{Error, Result};
use std::collections::HashMap;
use tokenizers::processors::template::{TemplateProcessing, TemplateProcessingBuilderError};

const BOS_TOKEN: &str = "<bos>";
const EOS_TOKEN: &str = "<eos>";

pub fn maybe_build_template(
    cfg: &PostCfg,
    special_ids: &HashMap<String, u32>,
) -> Result<Option<TemplateProcessing>> {
    if !cfg.add_bos && !cfg.add_eos && !cfg.pair_template {
        return Ok(None);
    }

    if cfg.add_bos && !special_ids.contains_key(BOS_TOKEN) {
        return Err(Error::Validation(
            "postprocessor requires <bos> special token when add_bos is true".into(),
        ));
    }

    if cfg.add_eos && !special_ids.contains_key(EOS_TOKEN) {
        return Err(Error::Validation(
            "postprocessor requires <eos> special token when add_eos is true".into(),
        ));
    }

    let mut builder = TemplateProcessing::builder();

    let single_template = single_sequence_template(cfg);
    builder = builder
        .try_single(single_template)
        .map_err(builder_error)?;

    let pair_template = pair_sequence_template(cfg);
    builder = builder
        .try_pair(pair_template)
        .map_err(builder_error)?;

    let special_tokens = special_ids
        .iter()
        .map(|(token, id)| (token.clone(), *id))
        .collect::<Vec<_>>();
    builder = builder.special_tokens(special_tokens);

    builder
        .build()
        .map(Some)
        .map_err(builder_error)
}

fn single_sequence_template(cfg: &PostCfg) -> Vec<String> {
    let mut pieces = Vec::new();

    if cfg.add_bos {
        pieces.push(format!("{BOS_TOKEN}:0"));
    }

    pieces.push("$A:0".to_owned());

    if cfg.add_eos {
        pieces.push(format!("{EOS_TOKEN}:0"));
    }

    pieces
}

fn pair_sequence_template(cfg: &PostCfg) -> Vec<String> {
    let mut pieces = Vec::new();

    if cfg.add_bos {
        pieces.push(format!("{BOS_TOKEN}:0"));
    }

    pieces.push("$A:0".to_owned());

    if cfg.add_eos {
        pieces.push(format!("{EOS_TOKEN}:0"));
    }

    if cfg.pair_template {
        if cfg.add_bos {
            pieces.push(format!("{BOS_TOKEN}:1"));
        }

        pieces.push("$B:1".to_owned());

        if cfg.add_eos {
            pieces.push(format!("{EOS_TOKEN}:1"));
        }
    } else {
        pieces.push("$B:1".to_owned());
    }

    pieces
}

fn builder_error(err: TemplateProcessingBuilderError) -> Error {
    Error::Validation(err.to_string())
}
