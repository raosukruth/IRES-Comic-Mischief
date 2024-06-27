# Dynamic Pretraining of Multimodal models for Recognition of Questionable Content in Videos
This repository contains modular implementation of HICCAP architecture. Each layer in HICCAP, such as Feature Encoding, HCA, Comic Mischief Detection tasks etc, is implemented as a composable module. These building blocks can be reused to compose an altogether different model such as a model meant for pretraining alone. In the case of pretraining, feature Encoding, HCA, and 3 MLP models (VTM, VAM, ATM) are composed to build a pretraining software stack.

# TODO
1. Implement support for Multi task heads
2. Implement support for evaluation
3. Strengthen the training loop

# Running the tests
Use the test as follows:
### python ComicMischiefTest.py <binary|multi|pretrain> <train|eval>

