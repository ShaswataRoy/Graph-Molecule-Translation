import rdkit
import random
import itertools
from rdkit import Chem
from rdkit.Chem import rdFMCS
from collections import defaultdict, deque
from fuseprop.vocab import MAX_VALENCE

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)
    return mol

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def valence_check(atom, bt):
    cur_val = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
    return cur_val + bt <= MAX_VALENCE[atom.GetSymbol()]

#mol must be RWMol object
def get_sub_mol(mol, sub_atoms):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()

def enum_root(smiles, num_decode):
    mol = Chem.MolFromSmiles(smiles)
    roots = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
    outputs = []
    for perm_roots in itertools.permutations(roots):
        if len(outputs) >= num_decode: break
        mol = Chem.MolFromSmiles(smiles)
        for i,a in enumerate(perm_roots):
            mol.GetAtomWithIdx(a).SetAtomMapNum(i + 1)
        outputs.append(Chem.MolToSmiles(mol))

    while len(outputs) < num_decode:
        outputs = outputs + outputs
    return outputs[:num_decode]
