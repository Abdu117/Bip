from dataclasses import asdict

from experiments import (
    MnistPretrained, MnistPretrainedConfig,
    MnistConfig,
    MnistDer, MnistDerConfig,
    MnistDerpp, MnistDerppConfig,
)

def main():
    """Run the pretrained experiment followed by the DER experiment."""
    pretrained_config = MnistPretrainedConfig(
        name="usecase1_pretrained",
        group="usecase1",
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
        name="usecase1_",
        group="usecase1",
        seed=42,

        batch_size=32,
        learning_rate=0.001,
        samples_per_class=200,
        pretrained=pretrained_exp,
        all_classes=[0, 3, 8],
        streamed_classes=[0, 3, 8],
    )
    config_args = asdict(config)
    config_args.pop("name")

    der_config = MnistDerConfig(
        **config_args,
        name=config.name + "der",
        buffer_size=200,
        alpha=0.5
    )
    MnistDer(der_config).conduct_full()

    derpp_config = MnistDerppConfig(
        **config_args,
        name=config.name + "derpp",
        buffer_size=200,
        alpha=0.5,
        beta=0.5
    )
    MnistDerpp(derpp_config).conduct_full()


if __name__ == "__main__":
    main()
