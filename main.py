from pathlib import Path
from experiments import MNISTPretrained, MNISTPretrainedConfig

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

    exp = MNISTPretrained(config)
    exp.conduct_full()

    for i, loader in exp.unused_loaders.items():
        print(f"{i}: {len(loader.dataset)} images unused")


if __name__ == "__main__":
    main()
