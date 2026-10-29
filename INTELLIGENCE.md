# AXON · Smart Inference and Safe Personalization

AXON is a digit classifier, not an autonomous general intelligence. v4 makes
predictions more robust and informative rather than claiming 99.99% accuracy.
The neural network remains implemented entirely in NumPy.

## Smart inference

With a trained 28×28 MNIST model, AXON evaluates the original centered drawing
plus four zero-padded one-pixel translations: left, right, up, and down.
The center receives 50% of the weight; each additional view receives 12.5%.
This can reduce dependence on exact centering in some cases, but can also make
some predictions worse. Validate both.

The app shows the top three digits, blended model score, prediction margin,
and fraction of views agreeing with the answer. Ambiguous inputs are flagged
for review instead of treated as reliable predictions. You can toggle smart
mode on or off.

The network explorer shows the real unshifted forward pass; it does not
pretend five separate model passes are a single computation. Reported softmax
scores are NOT calibrated probabilities of correctness.

The original 8×8 model only supports one pass. Train a 28×28 model first:

    python -m axon train-mnist
    python -m axon demo

## Measure the impact

    python -m axon audit
    python -m axon audit --limit 0
    python -m axon audit --test
    python -m axon predict digit.png
    python -m axon predict digit.png --classic

The audit reports clean accuracy, one-pixel-shift accuracy, accepted coverage,
and accuracy among accepted predictions. It defaults to the validation split.
The test split is available only by explicit request. Neither measures freehand
accuracy: evaluate that on new, independently drawn images.

## Safeguarded learning

Corrected drawings remain private in data/corrections/ and do not go to GitHub.
After saving correctly labeled examples in the app:

    python -m axon personalize

Personalization mixes saved drawings with base MNIST training examples to
reduce forgetting. The candidate checkpoint is rejected if validation
accuracy drops by more than 0.2 percentage points or accuracy on saved
drawings decreases. A rejected candidate does not overwrite the prior model.

Limits: the safeguards do not guarantee future freehand performance,
review thresholds are transparent heuristics (65% score, 20 percentage-point
margin, and 60% view agreement), and 99.99% recognition is not promised.
