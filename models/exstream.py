import torch


class ExStreamBuffer:
    """Class-specific prototype buffers from ExStream (Hayes et al., 2019, Sec. III-A).

    Each class stores at most `capacity` prototypes and how many samples each one
    represents. When a new sample arrives for a full class, the two closest
    prototypes are merged into their count-weighted mean, and the new sample takes
    the freed slot.
    """

    def __init__(self, capacity: int):
        if capacity < 2:
            raise ValueError("ExStream needs at least 2 prototypes per class to merge a pair.")

        self.capacity = capacity
        self._prototypes: dict[int, torch.Tensor] = {}
        self._counts: dict[int, torch.Tensor] = {}


    def add(self, feature: torch.Tensor, label: int) -> None:
        if label not in self._prototypes:
            self._prototypes[label] = feature.unsqueeze(0).clone()
            self._counts[label] = torch.ones(1, device=feature.device)
            return

        prototypes = self._prototypes[label]
        counts = self._counts[label]

        if len(prototypes) < self.capacity:
            self._prototypes[label] = torch.cat([prototypes, feature.unsqueeze(0)])
            self._counts[label] = torch.cat([counts, torch.ones(1, device=feature.device)])
            return

        i, j = self._closest_pair(prototypes)
        prototypes[i] = (counts[i] * prototypes[i] + counts[j] * prototypes[j]) / (counts[i] + counts[j])
        counts[i] += counts[j]
        prototypes[j] = feature
        counts[j] = 1


    def all_prototypes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns every stored prototype and its class label."""
        labels = [
            torch.full((len(prototypes),), label, device=prototypes.device)
            for label, prototypes in self._prototypes.items()
        ]
        return torch.cat(list(self._prototypes.values())), torch.cat(labels)


    @staticmethod
    def _closest_pair(prototypes: torch.Tensor) -> tuple[int, int]:
        distances = torch.cdist(prototypes, prototypes)
        distances.fill_diagonal_(float("inf"))
        i, j = torch.unravel_index(distances.argmin(), distances.shape)
        return int(i), int(j)
