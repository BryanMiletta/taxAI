#Bryan Miletta - CS995 Capstone
#TaxAI
#level: Proto
#summary: Builds dataset for fine-tuning

### ### ### Import necessary Libraries
import torch

### ### ### CLASS: SQuAD dataset functions
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
### ### ### CLASS: END