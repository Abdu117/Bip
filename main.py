from pathlib import Path

from experiments import MNISTPretrained, MNISTPretrainedConfig
from models import CNN

def main():
    config = MNISTPretrainedConfig(
        name="Usecase 1 - MNIST pretrained",
        seed=42,
        model_weights_filepath=Path(".out/mnist_pretrained.pth"),

        batch_size=64,
        epochs=2,
        learning_rate=0.001,
        samples_per_class=3000,
        classes_kept=[0, 3]
    )

    pretrained_exp = MNISTPretrained(config)
    pretrained_exp.conduct_full()

    print("A few labels from the unused dataset:")
    for i in range(10):
        print(" ", pretrained_exp.unused_dataset[i*500][1])

    pretrained_model = CNN.from_cnn(pretrained_exp.model, 3)
    print(pretrained_model)


if __name__ == "__main__":
    main()
