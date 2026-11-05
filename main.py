from dataclasses import asdict

from experiments import (
    MnistPretrained,
    MnistConfig,
    MnistDer,
    MnistDerpp,
)

def main():
    """Run the pretrained experiment followed by the DER experiment."""

    # Pretrained model
    pretrained_exp = MnistPretrained.from_kwargs(
        name="usecase1_pretrained",
        group="usecase1",
        seed=42,

        batch_size=64,
        epochs=3,
        learning_rate=0.001,
        samples_per_class=3000,
        classes_kept=[0, 3]
    )
    pretrained_exp.conduct_full()


    # Shared config between approaches
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

    # Conducting the different approaches using the pretrained model
    MnistDer.from_kwargs(
        **config_args,
        name=config.name + "der",
        buffer_size=200,
        alpha=0.5
    ).conduct_full()

    MnistDerpp.from_kwargs(
        **config_args,
        name=config.name + "derpp",
        buffer_size=200,
        alpha=0.5,
        beta=0.5
    ).conduct_full()


if __name__ == "__main__":
    main()
