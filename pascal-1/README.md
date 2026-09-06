# Pascal-1 Worklog

How good can a tiny model get?

More formally, what is the maximum intelligence per parameter
ratio achievable for a model that's around 124M parameters? Why that number? Well, a 12-layer (`depth=12`) GPT-2 model is 124M parameters, so that's our baseline.

I'm self-imposing the following restrictions for this project:

- All training runs on my M4 Pro Mac (24 GB VRAM)
- No frameworks, just Metal kernels, Rust, and some Python
- Minimal AI usage (I'm doing this for fun)

## Compute

According to the Chinchilla scaling laws, the ideal token to parameter ratio is 20:1.

So, for a 124M parameter model, we need around 2.5B tokens of pretraining data. Plugging this in to the full formula to get the TFLOPs:

$$

C = 6 \times 124\times10^{6} \times 2.5\times10^{9}
  = 1.86\times10^{18}\ \mathrm{FLOPs}
  = 1.86\times10^{6}\ \mathrm{TFLOPs}


$$

An Apple M4 Pro chip maxes out at around 18.4 TFLOPs in FP16 precision. Being conservative,
say we maintain 40% GPU saturation during training (accounting for the Mac's thermal throttling too).

$$

P_{\mathrm{effective}}
= \eta P_{\mathrm{peak}}
= 0.40 \times 18.4
= 7.36\ \mathrm{TFLOPs/s}


$$

So the full pretraining run should take:

$$

\frac{1.86\times10^{6}\ \mathrm{TFLOPs}}
     {18.4\ \mathrm{TFLOPs/s}\times0.40}
= 2.53\times10^{5}\ \mathrm{s}
= 70.2\ \mathrm{h}
\approx 2.93\ \mathrm{days}


$$

That's the theoretical value. I wonder if I can bring it closer to 48 hours (without degrading model quality). Additionally, the 124M parameter run will be the final (biggest) run on my Mac. Before doing such a heroic run, I'll do many smaller runs starting at `depth=4` and going up to `depth=10` in increments of two.

At the pretraining stage, I'm mainly interested in

- Getting a really good data mixture
- Exploring different model architectures (looped transformers?)
- Optimizing Metal kernels for faster training

Then, I will do an SFT warm-up to instruction-tune the model followed by some RL. At this stage, I'm interested in

- Trying different RL environments with verifiable rewards (coding, math)
- Eliciting tool-calling capabilities
- Tuning model personality

## Pretraining

### Dataset

Before we do anything, we need a dataset. The common choice here is [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), a subset the original FineWeb dataset focusing on education content, which leads to a better model in benchmarks. But, since we're running on contrained hardware, we want to optimise how much "intelligence per token" (i.e. high sample efficiency) we can juice from the dataset, and according to the findings in the [nanochat](https://github.com/karpathy/nanochat) project, the [Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) dataset leads to better performance. But, can we do even better? The model we're going to train will be _tiny_, so I'm ok with it being like a savant that's really good at one thing, while lacking in all other areas. In particular, I'm interested to see whethre mixing in agent traces in the pretraining data will lead to better agentic performance down the line after we do SFT. See the work by [Su et al.](https://arxiv.org/abs/2509.13310) showing this works. Furthermore, the original Llama paper showed that you can actually overtrain by using a greater token to parameter ratio than Chinchilla dictates, leading to better test-time performance in exchange for more FLOPs. So in this case, it may be worth to experiment with increasing our pretraining token count to around 5B (at this point though I would need to train on an actual GPU cluster).

### Architecture

The standard GPT architecture will lead to ~good results. This is already proven by nanochat. And at this point I've already [trained a GPT model in JAX](/gpt-on-tpu). So let's try something a bit different. The first thing I want to explore is a looped transformer.

Now let's try Mixture of Experts. I [already implemented this during my JAX studies](/gpt-on-tpu), however I didn't scale that to over 100M parameters.

One more thing that is of interest, is giving the model a native 1M token context window.
This would in theory enable highly efficient in-context learning and also improve retrieval capabilities. Prior research has shown that this works. To do this efficiently, we'll have to switch the attention mechanism to sliding-window attention (SWA), where the model only attends to a moving subset of the tokens in context.

#### Tokenization

Token-free would be interesting to explore. That is, byte-level embeddings, which means we only deal with a 256 dimension embedding matrix. See Meta's [BLT paper](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/).
