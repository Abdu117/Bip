## Use case 2: Results

Training the weak pretrained model on more data raises the test accuracy from **83.9 %** to up
to **97.5 %**, but EWC does not help here: the accuracy is highest without EWC (λ = 0) and
drops as λ grows.

### Setup

| Setting | Value |
| --- | --- |
| Pretraining | 100 images per class (1,000 images) |
| Continued training | 1,000 other images per class (10,000 images) |
| Test set | 200 images per class (2,000 images) |
| Epochs | 5 |
| Learning rate | 0.001 |
| λ values | 0, 0.1, 1, 10, 100 |
| Seed | 42 |

### Results

![Test accuracy after training for each λ](images/usecase2_accuracy_vs_lambda.png)

| λ | Pretrained model (%) | After training (%) | Gain (percentage points) |
| --- | --- | --- | --- |
| 0 (no EWC) | 83.90 | 97.50 | +13.60 |
| 0.1 | 83.90 | 96.80 | +12.90 |
| 1 | 83.90 | 96.75 | +12.85 |
| 10 | 83.90 | 96.65 | +12.75 |
| 100 | 83.90 | 94.00 | +10.10 |

### Observations

1. **More data improves the model a lot.** Every setting gains at least 10 percentage points
   over the pretrained model.
2. **No EWC is best.** λ = 0 reaches 97.50 %, 0.7 percentage points more than the best EWC
   setting (λ = 0.1, 96.80 %).
3. **A larger λ slows down learning.** Between λ = 0.1 and λ = 10 the accuracy barely changes
   (96.80 % to 96.65 %, only 3 test images). At λ = 100 it drops clearly to 94.00 %.
4. **Why EWC cannot help here:** the new data contains the same 10 digits as the pretraining
   data, so there is nothing to forget. EWC keeps the weights close to the weak pretrained
   model and only restricts how much the model can learn from the new data (stability is not
   needed here, only plasticity). EWC is useful when new training would destroy old knowledge,
   as in use case 1.

### Limitations

- **One seed.** The differences between the small λ values (0.1 to 10) are within a few test
  images and may change with another seed.
- **2,000 test images.** One image corresponds to 0.05 percentage points.
- **Fixed epochs and learning rate.** Only λ was varied; more epochs could raise the accuracy
  further for all settings.
