

Run::

    $ python example_types.py

This trains a neural network to recognise four different types of noisy functions from a number of samples.
We then see how, for a specific example input, a small nudge (epsilon) allows us to move away from the submanifold we have learnt on.
Away from this submanifold it is easy to dramatically change the output of the model.
Hence small, carefully crafted nudges may change the classification.
Or equivalently, although a model may be stable within the submanifold it has been trained on, it may be extremely sensitive to perturbations away from the learnt submanifold.

