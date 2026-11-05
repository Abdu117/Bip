import torch
import torch.nn as nn


class Der(nn.Module):
    """Dark Experience Replay model.

    Wraps a classifier (the ``backbone``) and keeps a fixed-size reservoir
    buffer of previously seen samples together with the logits the model
    produced for them.  Those cached logits are the "dark knowledge" used as
    soft targets during replay.
    """

    def __init__(self, backbone: nn.Module, buffer_size: int):
        """Wrap a backbone and create an empty reservoir buffer."""
        super().__init__()

        if buffer_size < 0:
            raise ValueError("buffer_size must be positive")

        self.backbone = backbone
        self.buffer_size = buffer_size

        # Stored as individual tensors so that growing the buffer does not
        # require re-allocating one large tensor.
        self.buffer_inputs: list[torch.Tensor] = []
        self.buffer_labels: list[torch.Tensor] = []
        self.buffer_logits: list[torch.Tensor] = []
        self.nb_seen = 0

    @property
    def buffer_length(self) -> int:
        return len(self.buffer_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def add_to_buffer(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """Add a batch of samples using reservoir sampling.

        Reservoir sampling keeps the buffer a uniform random sample of every
        example seen so far, regardless of when it arrived.
        """
        inputs = inputs.detach()
        labels = labels.detach()
        logits = logits.detach()

        for index in range(inputs.size(0)):
            self.nb_seen += 1

            if self.buffer_length < self.buffer_size:
                self.buffer_inputs.append(inputs[index].clone())
                self.buffer_labels.append(labels[index].clone())
                self.buffer_logits.append(logits[index].clone())
                continue

            # Keep this sample only if it replaces an existing reservoir slot.
            buffer_index = torch.randint(0, self.nb_seen, (1,)).item()
            if buffer_index < self.buffer_size:
                self.buffer_inputs[buffer_index] = inputs[index].clone()
                self.buffer_labels[buffer_index] = labels[index].clone()
                self.buffer_logits[buffer_index] = logits[index].clone()

    @torch.no_grad()
    def sample_buffer(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return a random replay batch, or ``None`` when the buffer is empty."""
        if self.buffer_length == 0:
            return None

        sample_size = min(batch_size, self.buffer_length)
        indices = torch.randperm(self.buffer_length)[:sample_size].tolist()

        inputs = torch.stack([self.buffer_inputs[index] for index in indices])
        labels = torch.stack([self.buffer_labels[index] for index in indices])
        logits = torch.stack([self.buffer_logits[index] for index in indices])

        return inputs, labels, logits
