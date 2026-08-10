# LFM2.5-230M Sparse Autoencoders

LFM2.5-230M is an interesting model. It's not a transformer, it's something hybrid, made up
of 8 double-gated Linear Input-Varying (LIV) convolution blocks and 6 Grouped Query Attention (GQA) blocks. The LIV conv blocks act as an alternative attention mechanism while the GQA blocks helps with associative recall.

It turns out (unsurprisingly), you can use Sparse Autoencoders to do behaviour steering on this type of model architecture too. At the end of the day, it too has a residual stream. In fact, its `d_model` is 1024, quite a wide hidden dimension for such a small model.

Thus, I trained BatchTopK SAEs on the residual stream of LFM2.5-230M.

## Training

I ran training on two Kaggle T4 GPUs. Since each model comfortably fits on a
single GPU, I was able to train two separate SAEs, one at layer 6 and another at layer 10. According to the model config these are both `full_attention` blocks (i.e. it uses GQA as the sequence-mixing sublayer).

One issue I ran into was LFM2.5-230M comes in bf16 precision (like most modern models) but T4 is a very old architecture and doesn't support bf16. So, I had to run LFM in fp16 instead. The only concern was whether this would cause overflow (it has a 5bit exponent which can't handle large values that bf16 could with a 8bit exponent). This can be easily validated though by checking all the hidden states (what we care about) are finite under fp16 and they were.

Note that the SAE itself is trained in FP32. This is because BatchTopK compares many small pre-activations across 16,384 features. FP16 rounding can change their ordering, and small
reconstruction-loss gradients can underflow or become noisy. FP32 makes the learned inference threshold and optimizer updates more stable.

During training, BatchTopK enforces 64 active features per token on average across each batch. The trained SAE is exported as a JumpReLU SAE, using a learned activation threshold allowing each token to be encoded independently during inference.

### Config

| Setting                  | Value                               |
| ------------------------ | ----------------------------------- |
| Model                    | `LiquidAI/LFM2.5-230M`              |
| Dataset                  | FineWeb-Edu (streamed)              |
| Hook points              | `model.layers.6`, `model.layers.10` |
| Training tokens          | 100M per SAE                        |
| Context length           | 1,024 tokens                        |
| Input dimension          | 1,024                               |
| SAE width                | 16,384 (16× expansion)              |
| BatchTopK $K$            | 64                                  |
| Training batch size      | 4,096 tokens                        |
| Learning rate            | $10^{-4}$                           |
| Model precision          | FP16                                |
| SAE precision            | FP32                                |
| Activation normalisation | None                                |
| Random seed              | 42                                  |

## Eval and Findings

| Metric                | Layer 6 | Layer 10 |
| --------------------- | ------: | -------: |
| L0                    |   65.17 |    69.09 |
| Explained variance    |  0.8511 |   0.8517 |
| Dead feature fraction |  0.0012 |   0.0035 |

The explained variance is ~85% at both layer 6 and layer 10 SAEs. L0 is near K (64) and the near-zero dead feature fraction shows we have good feature utilization as most features fired at least once across the entire eval set.

From a small sample, it appears layer 6 feature activations are mostly single token definitions like `fine`, `here`, `such`, while layer 10 feature activations are more _conceptual_, like `specifying`, `livestock`, `gameplay`. This makes sense as later hidden layers are presumed to encode more abstract meanings into latents, though a larger random feature exploration is required to make any definitive statements.

## Achieving Math Refusal

### Method

First, I gather a set of math question prompts vs normal conversation prompts. Using the SAE, I contrast the activations between those two, and keep the 32 most prominent features that fire on math prompts.

To detect if a prompt is asking a math question, I calculate the mean activation of the 32 features over the prompt tokens, and if that score is above an empirical threshold (acquired from running more prompts over a calibration set), I trigger a refusal.

For the refusal itself, [Arditi et al.](https://arxiv.org/abs/2406.11717) found that models tend to have a specific Refusal direction. To find this, I run normal chat prompts over a "refuse everything" system prompt and compare the activations to a typical "helpful assistant" system prompt. Taking the difference between the mean activations at the last prompt token thus gives the refusal direction.

To induce a refusal in the model's response, simply clamp the activation's projection along the refusal direction to a target value the model actually produces when refusing 

$$
p_{target} = p_{help} + k\,(p_{refuse} - p_{help})
$$ 

where $p_{refuse}$ and $p_{help}$ are the mean projections measured under the two system prompts. The layer and the scale $k \in \{1, 1.5, 2\}$ are swept on a calibration set. Following Arditi et al., layers past 0.8 of model depth are excluded from selection, since a "refusal direction" that close to the unembedding can just be suppressing refusal tokens at the output rather than steering the computation. Layer 10 was selected with $k = 1.5$. Interestingly, the pure clamp ($k = 1$) produced no refusals at any layer.

### Results

On a held-out test set, we successfully detected 7/8 math questions, and the refusal worked 75% of the time. Notably, it worked even when the questions were trying to be sneaky, like a worded problem that only used letters and no numbers in the prompt. However, math questions that had no concept of "quantity" in them were not refused.



All conditions on the held-out math questions (rates over 8 prompts):

| Condition                                | Refusal rate | Task completed | Coherent |
| ---------------------------------------- | -----------: | -------------: | -------: |
| Baseline (no intervention)               |         0.00 |           0.62 |     1.00 |
| SAE feature ablation only                |         0.00 |           0.62 |     1.00 |
| Refusal clamp only (gated)               |         0.75 |           0.12 |     1.00 |
| Ablation + clamp (gated)                 |         0.75 |           0.12 |     1.00 |
| Random direction, matched displacement   |         0.00 |           0.25 |     0.79 |
| Refusal clamp, gate bypassed             |         0.88 |           0.00 |     1.00 |

> The coherent column is just a heuristic calculation of how understandable the response is, and whether it suffers from repetition due to the intervention.

Importantly, our refusal clamp when applied unconditionally refuses all 8 control prompts, indicating the discovered refusal direction is indiscriminate, thus the refusal triggering on math is coming from the SAE-based detector.

Two examples of refusal are shown below:

```
"prompt": "I walked for half an hour at four miles an hour, then stopped. How far had I gone?",
"output": "I'm sorry, but I don't have this kind of inquiry asked. I'm just an AI designed to assist with general questions and provide information. If you'd like, I can help with anything that's within my capabilities!",
```

```
"prompt": "Two of every five people in a room of twenty are wearing hats. How many hats is that?",
"output": "I'm sorry, but I don't have this kind of question asked. Could you please clarify if there's something specific I can help with regarding this inquiry? For example, am I asking about a different type of question or perhaps just checking for any misunderstanding?",
```

An example of refusal not triggering:

```
"prompt": "Explain what it means for one quantity to grow in proportion to another.",
"output": "When two quantities grow in proportion, they mean that each unit of one quantity corresponds to a fixed amount or value relative to the other. This relationship is often described by a ratio where the growth factor (or scaling factor) indicates how much one quantity increases compared to the other. For example, if you",
```

### Ablating the math features

As another way to induce refusal, I also attempted to ablate the math features away, targeting the math features collected by the SAE, and subtracting the directions scaled by the measured activation at each token. However, this did not remove the model's ability to do math, suggesting that perhaps the mined features are not comprehensive. Re-encoding the ablated features confirms they were actually removed, only ~8% of the original math-feature activation mass survives. Further feature mining with a bigger and more diverse probing set could give us a more comprehensive set of features to ablate.