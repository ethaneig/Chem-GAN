import argparse
import os
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from utils import graph_to_molecule, PROJECT_DIR, LATENT_DIM, BOND_DIM, ATOM_DIM
from modules import build_generator

def sample(generator, batch_size, latent_dim, bond_dim, atom_dim, graph_to_molecule):
    generator.eval()

    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim, device=next(generator.parameters()).device)
        adj_logits, feat_logits = generator(z)

        adj_argmax = torch.argmax(adj_logits, dim=1)
        adjacency = F.one_hot(adj_argmax, num_classes=bond_dim).permute(0, 3, 1, 2).float()

        diag_indices = torch.arange(adjacency.size(-1))
        adjacency[:, :, diag_indices, diag_indices] = 0

        feat_argmax = torch.argmax(feat_logits, dim=2)
        features = F.one_hot(feat_argmax, num_classes=atom_dim).float()

    molecules = [
        graph_to_molecule([adjacency[i].cpu().numpy(), features[i].cpu().numpy()])
        for i in range(batch_size)
    ]
    return molecules

NUM_MOLECULES = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample molecules from a trained generator")
    parser.add_argument("--checkpoint", default=os.path.join(PROJECT_DIR, "generator.pth"))
    parser.add_argument("--max-samples", type=int, default=10000, help="give up after sampling this many molecules")
    parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "valid_molecules.png"))
    args = parser.parse_args()

    generator = build_generator()
    generator.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    unique = {}
    total = 0
    while len(unique) < NUM_MOLECULES and total < args.max_samples:
        for mol in sample(generator, NUM_MOLECULES, LATENT_DIM, BOND_DIM, ATOM_DIM, graph_to_molecule):
            if mol is not None and len(Chem.GetMolFrags(mol)) == 1:
                unique.setdefault(Chem.MolToSmiles(mol), mol)
        total += NUM_MOLECULES

    smiles = list(unique)[:NUM_MOLECULES]
    print(f"Found {len(smiles)} valid, distinct molecules in {total} samples")
    for smi in smiles:
        print(smi)
    if smiles:
        MolsToGridImage([unique[smi] for smi in smiles], molsPerRow=5, subImgSize=(200, 200), legends=smiles).save(args.out)
        print(f"Saved grid to {args.out}")
