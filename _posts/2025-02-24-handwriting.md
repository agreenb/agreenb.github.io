---
layout: post
title: "Scribble Trajectories: Building an Online Handwriting Transformer"
date: 2025-02-24
categories: [ml, robotics, transformers]
---

## Introduction

I see online handwriting generation — predicting pen strokes in real time — as a fun project akin to **robotic trajectory modeling**. A robot typically follows a sequence of motor commands (like Δx and Δy) plus discrete events (like lifting a pen or opening a gripper). My goal was to feel extremely comfortable with multimodal transformer modeling for a real task that required building everything from scratch: data processing, model development, training pipelines, experimentation, visualization. <!--more-->
I aim , my next project will be to map text or visual instructions to continuous robot movements.

I'll go over:

- My **data representation** with Δx, Δy, pen lift.
- Why discrete **transformers** improved over the **RNN + GMM** approach for me.
- Debugging tips: overfitting on small dataset subsets, cross-attention visualization, and more.
- How this approach ties into SOTA robotic transformers.
- Final results and next steps, including bridging to massive **foundation models**.

---

## Where Robotics Fits In

**Online handwriting** is essentially a 2D trajectory with an event signal (pen lift). A **robot manipulator** in 3D or 6D space can be described similarly: each step updates joint positions, and a discrete signal says "gripper open" or "gripper closed."

##### Connecting to SOTA robotics transformers

1. **RT-2**

   - **What it is:** [RT-2](https://arxiv.org/pdf/2307.15818) from Google DeepMind treats robot actions as text tokens. That way, it can take any vision-language model and fine-tune it into a vision-language-action model. During inference, the text tokens are decoded into continuous robot actions for closed-loop control.
   - **Why it relates:** This approach views motor commands as another language, just like how strokes could become discrete tokens in a handwriting language.

2. **RT-Trajectory**

   - **What it is:** [RT-Trajectory](https://arxiv.org/pdf/2311.01977), also from DeepMind, uses trajectories as a way to specify robot tasks. It captures similarities between motions across different tasks, improving generalization beyond language-only or image-only approaches.
   - **Why it relates:** Handwriting is already composed of 2D stroke movements, similar to these 2D trajectory sketches. By learning from 2D or 2.5D trajectory representations, RT-Trajectory shows how specifying a drawn path can yield robust policy generalization since it's easier to interpolate or adapt new motions.

3. **Pi-0**
   - **Why it relates:** [Pi-0](https://arxiv.org/pdf/2410.24164v1), from Physical Intelligence, integrates vision, language, and proprioception into a single embedding space. It uses action chunking and flow matching to model complex continuous actions at high frequencies up to 50 Hz.
   - **Why it relates:** Handwriting generation deals with trajectories at lower frequencies, but the principle is the same: refine trajectory generation in a globally coherent way.

**Online handwriting** is a smaller domain that uses similar transformer blocks you'd find above, but at a lower dimension.

---

## Model Development

#### Stage 1: Data representation and early models

##### 1) Δx, Δy, pen lift Format

I started with a stroke dataset where each timestep has three values:

- **Δx**: movement in x from the previous point
- **Δy**: movement in y
- **pen lift** ∈ {0, 1}: is the pen on paper or lifted?

##### 2) GMM approach

Alex Graves' [classic handwriting approach](https://arxiv.org/pdf/1308.0850) uses an RNN and mixture density network (MDN) to model continuous stroke distributions.

1. **MDN background**

   - **Gaussian Mixture Model:** A probabilistic approach assuming data arises from multiple Gaussian distributions, each with its own mean, variance, and mixture weight.
   - **Mixture Density Network:** A neural network that outputs the parameters of a GMM at each timestep, giving a continuous distribution over Δx and Δy.

2. **Why I started with this approach**

   - I wanted to be able to capture continuous variability in handwriting, like subtle style differences or loopy letters.
   - MDNs can represent multi-modal distributions, like multiple ways to draw the letter "a".

3. **Pitfalls**

   - **Mode collapse:** The MDN almost always converged to a single mixture component, producing monotonic diagonal movements—either up and to the right or down and to the left—regardless of text conditioning. This happened because it consistently picked the same Gaussian with a small variance range.
   - **Too many hyperparameters:** Balancing mixture weights (π), correlation (ρ), temperature, and entropy constraints required a lot of tuning. Since I was paying for my own training compute, I did not have the capacity to run so many experiments.
   - **Unstable training:** Subtle issues like exploding gradients or near-zero σ caused frequent NaNs or seemingly random outputs.

   It took time to dig into the model outputs and realize that mode collapse was causing this behavior. The model was always choosing the same Gaussian distribution, leading to repetitive, unnatural stroke patterns. This realization led me to shift toward discrete tokenization, which eliminated that unnatural diagonal stroke bias.

4. **Discrete transformers: simpler and more stable**
   - **Tokenized approach fits transformers:** Transformers excel at tokenized data, and binning Δx and Δy allowed me to use standard classification and cross-entropy.
   - **Tokenization works with the data**: Handwriting deltas span a small range, so discretizing them doesn't sacrifice much resolution or smoothness.
   - **No continuous sampling quirks:** Inference just relies on picking the next token from a softmax distribution.
   - **Easier debugging:** Digging into stroke tokens or mispredicted pen-lift is way easier than debugging multi-dimensional Gaussians.

##### 3) Unconditional generation with a transformer

First, I used **discrete binning** of Δx and Δy to turn continuous deltas into discrete tokens.
The original distribution of Δx and Δy values shows how most stroke movements are small and clustered around zero, while larger movements are significantly less frequent.

![Screenshot 6: Raw Δx, Δy Distribution](/assets/images/image6.png)

I implemented **adaptive binning**, which processed Δx and Δy into 24 bins each, allowing the transformer to autoregressively predict the next best token.

The binning strategy:

- **Fine resolution near zero**  
  Most strokes are small movements around Δx, Δy. By using uniform bins around zero, where most handwriting variations occur, the model can distinguish subtle pen movements.

- **Log-spaced tails**  
  Large movements are less frequent but can be significant (fast scribbles, big jumps). Logarithmic spacing ensures that minimal bins are assigned for these rare events while still capturing them.

- **Adaptive refinement**  
  Merge any bins with low data counts and adding extra bins where data was densest, maintaining a well-spaced distribution.

Computed bin edges after adaptive binning:

![Screenshot 7: Adaptive Binning Bin Edges](/assets/images/image7.png)

You can see that Δx and Δy are evenly distributed in their token classes rather than being concentrated in a few highly populated ones. Unlike uniform binning, which would have wasted resolution on rare large movements, small pen adjustments and larger strokes are properly represented.

![Screenshot 8: Adaptive Binning Applied to Δx, Δy](/assets/images/image8.png)

Now, it's more straightforward for the model to classify the next token with standard cross-entropy loss, rather than needing to regress to floating-point values.

###### I first tested a **transformer** that generates new strokes based on previous strokes:

- **Self-attention on stroke tokens**  
  The transformer processes each token (Δx, Δy, pen lift), learning how strokes typically flow over time.

- **Scribble-like outputs**  
  With no text to guide it, the model produces free-form strokes. These can look like letters or loops, confirming it can generate coherent movement patterns before adding text conditioning.

- **Advantages over RNN**  
  Transformers better capture long-range dependencies through self-attention, and discrete token classification avoids the complexities of continuous sampling or GMM instability.

---

#### Stage 2: Shifting to conditional generation (text-to-handwriting)

##### Alignment and sequence length

Now it was time to make the model more useful and multimodal. I wanted to input text (e.g. "Hello world") and produce realistic handwriting for it, so I:

- Used an encoder for text tokens, incorporating positional encodings to preserve character order.
- Used a decoder for stroke tokens, applying cross-attention so that each stroke prediction could reference the text.

##### Conditional generation

- **Encoder-decoder architecture**: The decoder queries the text encoder at each step, reinforcing the relationship between characters and strokes.
- **Pad masking**: Since text sequences vary in length, `[PAD]` tokens were masked so the model wouldn't attend to non-character tokens.
- **Cross-attention validation**: I monitored attention matrices to ensure stroke tokens referenced text embeddings when text was present, while still autoregressing correctly on strokes when text was absent. If attention was uniformly distributed across text tokens, it indicated the model was ignoring the text, requiring adjustments to positional encodings or cross-attention layers.

This text-conditioned generation allowed the model to map language directly onto structured, sequential handwriting motion.

```python
class TransformerStrokeModel(nn.Module):
    /* ... */
    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, stroke_batch, text_features=None, text_pad_mask=None, return_attn=False):
        b, t, _ = stroke_batch.shape

        x_x = self.embedding_x(stroke_batch[:, :, 1])
        x_y = self.embedding_y(stroke_batch[:, :, 2])
        combined = torch.cat([x_x, x_y], dim=-1)
        combined = self.fc_embed(combined)
        combined = self.position_encoding_strokes(combined)

        /* ... */
        decoded, cross_attn_list = self.decoder(
            combined, text_features,
            tgt_mask=causal_mask,
            return_attn_weights=True,
            memory_key_padding_mask=text_pad_mask,
        )

        logits_x = self.fc_out_x(decoded)
        logits_y = self.fc_out_y(decoded)
        pen_logits = self.fc_pen_lift(decoded).squeeze(-1)

        /* ... */
```

---

#### Stage 3: Debugging strategy

##### Overfitting on one example, then five, then more

1. When my model couldn't memorize one text-stroke pair, I knew I had a model architecture or data processing bug.
2. Then I'd test 5 pairs to see if it could overfit. Then 10, then 100.
3. Only after that would I expand to the full dataset.

##### Oops moments

1. **Swapped Δx and pen lift**: In one run, Δy looked great but the rest was off, and I discovered I had columns reversed in the batch. Simple fix.
2. **No positional encodings for text**: Without them, the cross-attention was relatively random. Once I added `PositionalEncoding`, the attention matrix aligned strokes with the correct characters in order.

---

#### Stage 4: Architecture tweaks

#### Less is more

- I first tried 8–12 heads and 6–8 layers, but it was overkill that slowed down training because of wasted model capacity on limited data. 2–4 heads and 1–3 layers gave quicker convergence and easier debugging.

##### Visualizing attention weights in PyTorch

- While debugging, a huge interpretability boost was subclassing `nn.TransformerDecoderLayer` to return cross-attention weights. Plotting them as heatmaps (stroke_tokens × text_tokens) let me check alignment and quickly debug a bunch of situations (e.g. text completely ignored, positional encodings ignored, everything fixated on the first letter).

---

#### Stage 5: Refining conditional generation

With the basics working, I could now focus on long-tail wins:

- **pen lift** weighting: the pen is only lifted 5% of the time in the dataset, so I used focal loss.
- **Temperature sampling**: preventing repetitive loops at inference time.
- **Occasional no-text**: forcing unconditional mode half the time to preserve general scribble ability.

---

#### Stage 6: Final results and observations

Plenty of good:

- **Human-like** cursive or printed letters, with real pen lifts.

Some not-so-good:

- **Over-smoothing**: some letters lost distinct edges.
- **Spacing**: sometimes too tight or too wide.

I spent a lot of time visualizing:

1. Plotting each epoch's output.
2. Seeding with real strokes for X timesteps, letting the model complete the rest.

---

#### Stage 7: Future directions and extensions

- **Handwriting → text**  
  The inverse model is basically handwriting recognition. This closes the loop and could enable a single multimodal, multitask model that does handwriting generation and recognition.

- **Style transfer**  
  Condition on user-specific samples to replicate personal handwriting styles or emulate various fonts. With a small style embedding or reference strokes, the model can generate text in that style.

- **Diffusion, VAEs, and EBMs**

  - **Diffusion models:**  
    Instead of discrete or GMM-based sampling, a diffusion approach could produce smoother strokes by iteratively refining a noisy sequence. Latent diffusion models include a VAE-like encoder-decoder pipeline that compresses the data before applying the denoising process.

    - [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/pdf/2205.09991) proposes a non-autoregressive method that predicts all timesteps concurrently, blurring the lines between sampling from a trajectory model and planning with it. Each denoising step focuses on local consistency (nearby timesteps in the past and future), but composing many of these steps creats global coherence. Applied to handwriting or robot motion, this approach can help ensure the entire stroke or trajectory is consistent, rather than being generated purely from past context in a causal manner.

    - Pi-0 replaces standard cross-entropy with a **flow matching** loss in a decoder-only transformer. They maintain separate weights for the diffusion tokens, effectively embedding diffusion within a vision-language-action pipeline to handle high-frequency action control. A handwriting model could do the same.

  - **Variational autoencoders:**  
    A VAE would encode Δx, Δy sequences into a latent space and then decodes them back into strokes, allowing style interpolation or manipulation in the latent space.

  - **Energy-based models:**  
    EBMs define an energy function over data configurations. I spent a number of months studying EBM math and typical architectures, which helped when considering more flexible training objectives or capturing complex multimodal distributions. Going further in this direction could produce more robust handwriting outputs while reducing mode collapse.

- **Interactive Demos**  
  A web-based interface where users type text and see real-time text generation. Also great for collecting more data.

- **Robotics**  
  Expand from 2D pen strokes to 3D or 6D manipulator movements. The model's text conditioning can guide a robot to write letters on a whiteboard or execute more complex tasks, similar to methods like RT-2 or Pi-0 that fuse language with action tokens in a transformer.

---

## Example outputs

##### Early MDN generation

![Screenshot 5: Early Transformer + MDN Generation](/assets/images/image5.png)

###### **Identifying the Issue**

- The generated strokes consistently move up and to the right, revealing that the model **always selects the same Gaussian component** instead of adapting to the handwriting context.
- Even though the transformer theoretically improved long-range dependencies, the unstable MDN head remained a bottleneck, producing repetitive trajectories.
- A shift to **discrete tokenization** was necessary to give the transformer better control over individual stroke outputs, avoiding the need for unstable mixture sampling.

After trying a lot of hyperparameter tuning and model tweaks, I switched to discrete tokens, which eliminated this issue.

##### Early discrete transformer generation

In this example, I seeded the model with some strokes from the dataset (in blue) and let it generate the remaining handwriting (in red) based on both strokes and text. Early on, the model produced fragmented and erratic strokes.

![Screenshot 1: Early model generation](/assets/images/image1.png)

###### **Identifying the issue**

- The generated strokes frequently lost coherence, failing to maintain the structure of letters beyond a few timesteps.
- This early output helped diagnose issues with:
  - **cross-attention alignment:** ensuring the model properly conditions on text instead of generating arbitrary strokes.
  - **stroke continuity:** adjusting positional encodings and training dynamics to prevent erratic jumps.
  - **autoregressive stability:** the model struggled to smoothly transition from real strokes to generated ones.

As training progressed, I refined the text-stroke relationship, improving overall performance.

---

##### Pen lift failure due to imbalanced data

After some refinements on the above, the model still **failed to lift the pen** correctly.

![Screenshot 2: pen lift imbalance](/assets/images/image2.png)

###### **Identifying the issue**

- The model struggled with pen lifts, treating them as rare occurrences and failing to separate letters properly.
- This happened because lift events were underrepresented in the dataset, making the model biased toward keeping the pen down.

###### **Fixing it with focal loss**

- I introduced **focal loss**, which increases the weight of rare events during training.

---

##### Lack of positional encoding in text

After the above, I finally got the model producing **reasonable letter-like scribbles**, but they were still jumbled and nonsensical, failing to follow the intended character sequence.

![Screenshot 3: missing positional encoding](/assets/images/image3.png)

###### **Identifying the issue**

- The model appeared to understand what letters should look like, but had no sense of where to place them in relation to the input text.
- This is because positional encodings were missing from the text encoder, meaning the model saw all text tokens as unordered symbols rather than a structured sequence.

###### **Fixing it with positional encodings**

- With this fix, the model could associate text positions with stroke positions, improving letter placement and structure.

---

##### Improved handwriting generation

This example demonstrates a **successful** text-to-handwriting generation.

![Screenshot 4: final improved result](/assets/images/image4.png)

###### **What works here**

- The generated handwriting follows the structure of the input text, properly aligning strokes with corresponding letters.
- Pen lifts are correctly placed to make coherent spacing.
- Consistent letter spacing and stroke continuity make the handwriting flow naturally, avoiding erratic movements from earlier models.

###### **Key improvements that led to this result**

This level of convergence required many iterations of **model architecture refinement, hyperparameter tuning, and targeted loss reductions**:

- **Model architecture improvements**: I reduced the number of Transformer layers and heads to **2–4 heads, 1–3 layers**, balancing expressiveness and convergence speed. Larger models took longer to learn without necessarily improving output quality.
- **Hyperparameter tuning**: Learning rate schedules, warm-up steps, and batch size adjustments helped **stabilize training and prevent overfitting**, particularly by adjusting loss scaling on pen lift events.
- **Loss function adjustments**: Introducing **focal loss** helped **pen lift balancing**, while ensuring **positional encodings** were properly applied to both stroke and text tokens helped cross-attention learn better alignments.
- **Improved cross-attention layers**: Careful inspection of attention weights helped uncover bugs.

These refinements steadily brought loss down.

---

## Conclusion and Key Takeaways

1. **Sequential data**

   The same concepts for handling pen lift and Δx, Δy in an autoregressive way can be applied to controlling a robot or simulating other continuous systems.

2. **Text conditioning**

   Leveraging cross-attention and positional encodings allows the model to align characters with specific stroke segments, which can scale to conditioning on other modalities, like images or sensor data in robotics.

3. **Transformers vs. RNNs**

   Self-attention is great for long-range dependencies, large sequence lengths, and scaling up on training. Discretizing Δx and Δy further simplifies transformer training compared to an MDN approach, which can be unstable.

4. **Debugging**

   Overfitting small subsets (1, then 5 examples), verifying shapes, and visualizing attention were critical to catching bugs.

5. **Robotics**

   Scaling from 2D pen strokes to a 6D (or 7D) manipulator: replace "pen lift" with "gripper open," and (Δx, Δy) with joint movements.

   I really liked the exercise of **online** handwriting vs. offline (static images of text). Robotics is fundamentally about continuous action sequences, and online data is essential for capturing that moment-to-moment motion, so I recommend trying this project out for yourself :)
