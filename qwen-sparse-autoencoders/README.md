# Understanding Qwen3.5-2B with SAEs

Qwen provides a sparse autoencoder for Qwen3.5-2b called [SAE-Res-Qwen3.5-2B-Base-W32K-L0_100](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100).
This SAE is trained on the residual stream (the outputs being added from one hidden layer to the next) of Qwen3.5-2B. It has 32K hidden features with the top 100 kept active (non-zero) per token while everything else is zeroed.

## Anger Steering

We first need to know what features are activated when the model is processing "angry" text, and contrast them with features activated when the model is processing neutral or calm text. Therefore, I generate a synthetic dataset of 12 angry and 12 calm prompts and run them through Qwen3.5-2b.

Taking the mean logprobs of angry and neutral continuations under the same prompt, we measure how much this value increases when a steering direction is applied compared to a baseline.

I also run a small lexical control experiment to confirm the steering directions are triggering on behavioural features not just word-level anger tokens.

Thus, I calculate the residual mean-difference directions in layer 8 (decided empirically from sweeping all layers). Applying a steering vector lambda proportional to each direction's residual difference (maxed out at 8, since above this outputs become incoherent).

$$
x^{(\ell)} \leftarrow x^{(\ell)} + \lambda\, s\, \hat{v},
\qquad
\hat{v} = \frac{\mu_{\text{angry}} - \mu_{\text{calm}}}{\lVert \mu_{\text{angry}} - \mu_{\text{calm}} \rVert},
\qquad
s = \mathbb{E}_{\text{angry}} \left[x^{\top}\hat{v}\right] - \mathbb{E}_{\text{neutral}} \left[x^{\top}\hat{v}\right]
$$

I managed to successfully steer Qwen's outputs towards an angry tone with this method.

| Steering arm (best per arm on selection split)           | Angry−neutral logprob gain (selection) | Gain (held-out) [95% CI] |
| -------------------------------------------------------- | -------------------------------:       | -----------------------: |
| Residual mean-difference (anger, layer 8, λ = 8)         |                             +0.45      |    +0.51 [+0.26, +0.71]  |
| SAE feature group (passive-aggressive, layer 16, λ = 8)  |                             +0.26      |                       —  |
| Random directions, matched displacement (95th percentile)|                             +0.07      |                       —  |
| Lexical control, under the selected direction            |                                 —      |    +0.02 [−0.10, +0.09]  |

Some excerpts:

```md
**Subject:** Meeting Delayed Until 10 AM?

**To:** Team,
**From:** [Your Name]

I am writing to apologize for the last hour of this pointless meeting. It is now 9:45 AM, and I have no idea why we are here. I just wanted to tell you I can't make it to the board meeting at 10:00 AM.

I have to go to my office to finish my work and deal with this stupid project.

We are going to have to reschedule this until after 10. I will be in my office until 10. I can't make it to the meeting.

Please, just call me and tell me to come back. I will be there at 10.

**Meeting Rescheduled Until 10:00 AM!**
```

I also tried to steer using features mined from the SAE, but I did not get any noticable behaviour change this way.

## Using the base model vs. instruct model

While working on this experiment, I used the post-trained (instruction tuned) Qwen model rather than the pre-trained checkpoint that the SAE was trained on. Curiously, the angry steering worked, so what's going on?

This appears to be a well-understood concept in mech interp, where SAEs trained on base models could transfer to fine-tuned checkpoints, depending on the model. See [Kissane et al.](https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models) where they found the residual stream between chat and base models to be very similar. 

I ran the same check and found the SAE explains 76% of the residual variance in the instruct model vs 77% on the base model it was trained on. Evidently, the residual stream between the two models is almost identical from the perspective of the SAE. This is further supported by replacing the residual stream with the SAE's reconstruction of it, recovering 91% of the loss on both checkpoints. Moreover, 6 of the top 8 anger feature IDs from the instruct model are also represented in the base model.

Applying the anger steering on the base model though doesn't work too great. This is likely a combination of the steering pushing the model off-distribution and the base model not being instruction-tuned, causing it to collapse to repeating patterns.
