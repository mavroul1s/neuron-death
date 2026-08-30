# Published neuron-focused methods extension

This extension implements only the next step requested by the supervisor: add
one or two published techniques that act on neurons which are not responding,
compare them with ReDo, and inspect the results before considering a new method.
It does not introduce a method of our own and does not change the manuscript.

## Methods

### ReDo — activation magnitude

At every 1,000 optimizer steps, ReDo evaluates a fresh 64-example batch. For
neuron `i` in layer `l`, it divides mean absolute activation by the layer-wide
mean. A neuron is reset when this normalized score is at most `tau`. Resetting
resamples incoming weights from the original initialization distribution,
restores the incoming bias, zeros outgoing weights, and zeros the matching
optimizer-state slices. The comparison uses `tau=0.1`, the best established
setting in this project.

### SNR — inter-firing-time history

Self-Normalized Resets (SNR) tracks the number of examples since each neuron
last produced a positive activation. A unit's own observed inter-firing times
define its reset threshold, so frequently firing and rarely firing units are not
judged against one common inactivity duration. The released mini-batch estimator
treats a neuron as firing when it is positive for at least one example (and one
spatial position for a convolutional channel) in the batch; otherwise its age
increases by batch size.

The implementation follows the authors' official Permuted-MNIST Colab defaults:

- rejection tail probability `eta=0.08`, i.e. the 92nd percentile;
- initial/capped inactivity threshold: 20,000 examples;
- neuron-specific threshold update every 16 tasks;
- double a threshold when its new percentile does not contract it;
- minimum threshold: 100 examples.

The released code stores a dense histogram. This implementation stores the
same counts in a sparse age-indexed histogram, which is statistically identical
but keeps checkpoints practical for the 3×500-unit MLP.

Primary sources: [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/359c0c4dc53b6bf64041b4c8334f1c11-Paper-Conference.pdf),
[official implementation](https://github.com/ajozefiak/SelfNormalizedResets).

### ReGraMa — gradient magnitude

ReGraMa replaces ReDo's activation score with GraMa. For every hidden neuron,
it computes the mean absolute gradient of that neuron's incoming weights and
normalizes it by the layer mean:

`G_i^l = mean(|grad W_i^l|) / mean_k(mean(|grad W_k^l|))`.

Every 1,000 steps, neurons with `G_i^l <= 0.01` are reset using the same
incoming-resample/outgoing-zero operation as ReDo. This isolates the selection
signal: activation magnitude for ReDo versus learning capacity represented by
gradient magnitude for ReGraMa.

The paper and released code agree on the score and threshold. The paper also
specifies zeroing outgoing weights; this project follows that stated reset rule
and resets optimizer slices, matching its existing faithful ReDo pipeline.

Primary sources: [NeurIPS 2025 paper](https://papers.nips.cc/paper_files/paper/2025/file/722f3f9298a961d2639eadd3f14a2816-Paper-Conference.pdf),
[official implementation](https://github.com/torressliu/grad-based-plasticity-metrics).

## Initial experiment

`configs/neuron_methods/` contains 15 paired runs:

- 5 seeds × ReDo (`tau=0.1`);
- 5 seeds × SNR (`eta=0.08`);
- 5 seeds × ReGraMa (`tau=0.01`).

All use the established 200-task Permuted-MNIST setting, 784–500–500–500–10
ReLU MLP, batch size 128, and SGD with momentum 0.9 at the calibrated learning
rate. The existing no-intervention runs for seeds 0–4 provide the baseline.

The first comparison should report online accuracy, late-window accuracy,
number of resets, reset count by layer, and the composition of the selected set
under the project's common dead/dormant definitions. ReGraMa was introduced in
deep RL, so its result here is evidence about transfer to continual supervised
learning, not a direct reproduction of its benchmark claims.

