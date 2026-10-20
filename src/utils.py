import os
from rdkit import Chem, RDLogger
import pandas as pd
import numpy as np

RDLogger.DisableLog("rdApp.*")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(path=os.path.join(PROJECT_DIR, "qm9.csv")):
    data = pd.read_csv(path)
    smiles = data["smiles"].values
    return smiles

ATOM_MAPPING = {
     'C':0,
      0:'C',
     'N':1,
      1:'N',
     'O': 2,
      2:'O',
     'F': 3,
     3: 'F'
}

BOND_MAPPING = {
    "SINGLE": 0,
    0: Chem.BondType.SINGLE,
    "DOUBLE": 1,
    1: Chem.BondType.DOUBLE,
    "TRIPLE": 2,
    2: Chem.BondType.TRIPLE,
    "AROMATIC": 3,
    3: Chem.BondType.AROMATIC,
}

NUM_ATOMS = 9 
ATOM_DIM = 4 + 1
BOND_DIM = 4 + 1
LATENT_DIM = 64

def smiles_to_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
  
    adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), 'float32')
    features = np.zeros((NUM_ATOMS, ATOM_DIM), 'float32')

    for atom in molecule.GetAtoms():
        j = atom.GetIdx()
        atomtype = ATOM_MAPPING[atom.GetSymbol()]
        features[j] = np.eye(ATOM_DIM)[atomtype]
    
        for neighbor in atom.GetNeighbors():
            k = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(j,k)
            bondtypeindex = BOND_MAPPING[bond.GetBondType().name]
            adjacency[bondtypeindex, [j,k], [k,j]] = 1
            
    adjacency[-1,np.sum(adjacency, axis=0)==0] = 1
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1
            
    return adjacency, features
        
def graph_to_molecule(graph):
    adjacency, features = graph
    molecule = Chem.RWMol()
    
    keep_indexes = np.where((np.argmax(features, axis=1) != ATOM_DIM-1) &(np.sum(adjacency[:-1], axis=(0,1)) != 0))[0]
    features = features[keep_indexes]
    adjacency = adjacency[:, keep_indexes, :][:, :, keep_indexes]
    for atomtypeindex in np.argmax(features, axis=1):
        atom = Chem.Atom(ATOM_MAPPING[atomtypeindex])
        _ = molecule.AddAtom(atom)
        
    (bondsjk, atomj, atomk) = np.where(np.triu(adjacency)==1)
    for (bondsjk, atomj, atomk) in zip(bondsjk, atomj, atomk):
        if atomj == atomk or bondsjk == BOND_DIM-1:
            continue
        bondtype = BOND_MAPPING[bondsjk]
        molecule.AddBond(int(atomj), int(atomk), bondtype)

    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
      return None
    return molecule

def create_dataset(smiles):
    adjacencymatrix, featuresmatrix = [],[]
    for smile in smiles:
        adjacency, features = smiles_to_graph(smile)
        adjacencymatrix.append(adjacency)
        featuresmatrix.append(features)
        
    adjacencymatrix = np.array(adjacencymatrix)
    featuresmatrix = np.array(featuresmatrix)
    
    return adjacencymatrix, featuresmatrix