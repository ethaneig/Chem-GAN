import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import QED
from rdkit.Chem.Draw import MolsToGridImage
from modules import build_generator, build_discriminator
from gan import GraphWGAN
from generate import sample
from utils import graph_to_molecule, load_data, create_dataset, PROJECT_DIR, LATENT_DIM, BOND_DIM, ATOM_DIM


def calculate_chem_metrics(molecules):
    # Validity
    valid_mols = [mol for mol in molecules if mol is not None]
    validity = len(valid_mols) / len(molecules) if molecules else 0
    
    # drug-likeness (QED)
    qed = sum(QED.qed(mol) for mol in valid_mols) / len(valid_mols) if valid_mols else 0
    
    # diversity
    diversity = 0.0
    if len(valid_mols) > 1:
        fps = [Chem.RDKFingerprint(mol) for mol in valid_mols]
        similarities = []
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                similarities.append(DataStructs.FingerprintSimilarity(fps[i], fps[j]))
        diversity = 1.0 - (sum(similarities) / len(similarities))
    
    return {
        'validity': validity,
        'diversity': diversity,
        'qed': qed
    }


if __name__ == "__main__":
    out_dir = os.path.join(PROJECT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    adjacencymatrix, featuresmatrix = create_dataset(load_data())

    generator = build_generator()
    discriminator = build_discriminator()

    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # Create WGAN instance
    wgan = GraphWGAN(generator, discriminator)
    wgan.compile(opt_g, opt_d)

    dataloader = torch.utils.data.DataLoader(
        list(zip(adjacencymatrix, featuresmatrix)),
        batch_size=32,
        shuffle=True,
    )
    valid_mol_history = []
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_metrics = {
            'loss_gen': 0,
            'loss_dis': 0,
        }
        
        batch_count = 0
        i = 0
        for real_batch in dataloader:
            adjacency, features = real_batch
            if not isinstance(adjacency, torch.Tensor):
                adjacency = torch.tensor(adjacency)
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features)
                
            metrics = wgan.train_step((adjacency, features))
            
            for k, v in metrics.items():
                epoch_metrics[k] += v
            batch_count += 1
            
            if i % 100 == 0:
                mols = sample(wgan.generator, 10, LATENT_DIM, BOND_DIM, ATOM_DIM, graph_to_molecule)
                valid_count = sum(1 for mol in mols if mol is not None)
                valid_mol_history.append(valid_count / len(mols))
                wgan.generator.train()
                MolsToGridImage(
                    mols, 
                    molsPerRow=5, 
                    subImgSize=(200, 200),
                    legends=["valid" if mol is not None else "invalid" for mol in mols]
                ).save(os.path.join(out_dir, f"samples_{epoch}_{i}.png"))
                print(f"Epoch {epoch} step {i} - Valid: {valid_count}/{len(mols)}")
                
                torch.save(generator.state_dict(), os.path.join(out_dir, f"generator_checkpoint_{epoch}_{i}.pth"))
            
            i += 1
        
        for k in epoch_metrics:
            epoch_metrics[k] /= max(batch_count, 1)

        mols = sample(wgan.generator, 100, LATENT_DIM, BOND_DIM, ATOM_DIM, graph_to_molecule)
        wgan.generator.train()
        chem_metrics = calculate_chem_metrics(mols)
        
        print(f"Epoch {epoch+1}/{num_epochs} - " + 
            f"Gen Loss: {epoch_metrics['loss_gen']:.4f}, " +
            f"Dis Loss: {epoch_metrics['loss_dis']:.4f}, " +
            f"Validity: {chem_metrics['validity']:.2f}, " +
            f"Diversity: {chem_metrics['diversity']:.4f}, " +
            f"QED: {chem_metrics['qed']:.4f}")
        
        if epoch % 10 == 0 and valid_mol_history:
            plt.figure(figsize=(10, 5))
            plt.plot(valid_mol_history)
            plt.title("Molecule Validity over Training")
            plt.xlabel("Sampling Step (x100)")
            plt.ylabel("Validity Ratio")
            plt.ylim(0, 1)
            plt.savefig(os.path.join(out_dir, "validity.png"))
            plt.close()

    torch.save(generator.state_dict(), os.path.join(out_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(out_dir, "discriminator.pth"))