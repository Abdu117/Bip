from pathlib import Path
from experiments import MNISTExStream, MNISTExStreamConfig


def usecase_1_config(use_replay: bool) -> MNISTExStreamConfig:
    """Improve all digits: pretrain on few samples, then stream unseen samples of the same digits."""
    return MNISTExStreamConfig(
        name=f"Usecase 1 - improve known classes ({'ExStream' if use_replay else 'no buffer'})",
        seed=42,
        model_weights_filepath=Path(f".out/usecase_1_{'exstream' if use_replay else 'no_buffer'}.pth"),

        batch_size=64,
        epochs=5,
        learning_rate=0.001,
        samples_per_class=100,

        pretrain_classes=list(range(10)),
        stream_classes=list(range(10)),
        stream_samples_per_class=1000,
        buffer_capacity=50,
        stream_learning_rate=0.0001,
        eval_every=2000,
        use_replay=use_replay,
    )


def usecase_2_config(use_replay: bool) -> MNISTExStreamConfig:
    """Learn a new digit: pretrain on 0 and 3, then stream only 6."""
    return MNISTExStreamConfig(
        name=f"Usecase 2 - learn a new class ({'ExStream' if use_replay else 'no buffer'})",
        seed=42,
        model_weights_filepath=Path(f".out/usecase_2_{'exstream' if use_replay else 'no_buffer'}.pth"),

        batch_size=64,
        epochs=5,
        learning_rate=0.001,
        samples_per_class=3000,

        pretrain_classes=[0, 3],
        stream_classes=[6],
        stream_samples_per_class=2000,
        buffer_capacity=20,
        stream_learning_rate=0.0001,
        eval_every=500,
        use_replay=use_replay,
    )


def main():
    for build_config in (usecase_1_config, usecase_2_config):
        for use_replay in (True, False):
            MNISTExStream(build_config(use_replay)).conduct_full()


if __name__ == "__main__":
    main()
