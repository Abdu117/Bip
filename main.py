from pathlib import Path

from experiments import MnistPretrained, MnistPretrainedConfig, MnistDer, MnistDerConfig

def main():
    pretrained_config = MnistPretrainedConfig(
        name="Usecase 1 - MNIST pretrained",
        seed=42,
        model_weights_filepath=Path(".out/usecase1_pretrained.pth"),

        batch_size=64,
        epochs=3,
        learning_rate=0.001,
        samples_per_class=3000,
        classes_kept=[0, 3]
    )

    pretrained_exp = MnistPretrained(pretrained_config)
    pretrained_exp.conduct_full()


    der_config = MnistDerConfig(
        name="Usecase 1 - DER",
        seed=42,
        model_weights_filepath=Path(".out/usecase1_der.pth"),

        batch_size=64,
        learning_rate=0.001,
        evals=20,
        samples_per_class=200,
        alpha=0.05,
        pretrained=pretrained_exp,
        classes_added=[8],
        buffer_size=100
    )

    der_exp = MnistDer(der_config)
    der_exp.conduct_full()


if __name__ == "__main__":
    main()
