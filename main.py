from dataclasses import asdict

from experiments import (
    MnistPretrained, MnistPretrainedConfig,
    MnistConfig,
    MnistDer, MnistDerConfig
)

def main():
    pretrained_config = MnistPretrainedConfig(
        name="usecase1_pretrained",
        seed=42,

        batch_size=64,
        epochs=3,
        learning_rate=0.001,
        samples_per_class=3000,
        classes_kept=[0, 3]
    )

    pretrained_exp = MnistPretrained(pretrained_config)
    pretrained_exp.conduct_full()


    config = MnistConfig(
        name="usecase1_der",
        seed=42,

        batch_size=32,
        learning_rate=0.001,
        samples_per_class=500,
        alpha=0.05,
        pretrained=pretrained_exp,
        classes_added=[8],
    )

    der_config = MnistDerConfig(**asdict(config), buffer_size=200)
    der_exp = MnistDer(der_config)
    der_exp.conduct_full()


if __name__ == "__main__":
    main()
