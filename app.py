import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from decode import *
from rdkit.Chem.Descriptors import qed
import re

title = st.header('Draw the Molecule')
input_smiles = st_ketcher()

# p = re.compile(r"([a-zA-z][0-9]*)\(\*\)")
# smiles = p.sub("[\\1:1]", input_smiles)
smiles = input_smiles.replace("*", "[C:1]")

n_mol = 3

with st.sidebar:
    n_mol = st.number_input("Number of Molecules", value=5, placeholder="Type a number...")
    genre = st.radio(
    "Property",
    ["QED", "Custom"],
    captions = ["Quantitative Estimate of Druglikeness", \
                "Custom Property"])
    sa_values = st.slider('SA Score',0., 10., (1.,5.),step=0.1)

def calc_qed(smiles_list):
    scores = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scores.append(0)
        else:
            scores.append(qed(mol))
    return np.float32(scores)

def sa_func(smiles_list):
    scores = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scores.append(100)
        else:
            scores.append(Chem.Crippen.MolLogP(mol))
    return np.float32(scores)

def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')

col1, col2 = st.columns(2)
with col1:
    st.header('Input Molecule')

with col2:
    st.header('Scores')

with col1:
    m = Chem.MolFromSmiles(input_smiles)
    im=Draw.MolToImage(m,size = (200,200),fitImage = True,useSVG=True)
    st.image(im)
with col2:
    v_spacer(height=3)
    if input_smiles:
        st.markdown(f" ### QED : {qed(m):1.3f}")
        st.markdown(f" ### SA : {sa_func([smiles])[0]:1.3f}")
    v_spacer(height=3)



if st.button('Generate'):
    # title = st.header('Output Molecules')

    
    
    with col1:
        st.header('Output Molecule')

    with col2:
        st.header('Scores')

    n_outputs = 0

    while n_outputs<n_mol:
        output_mols = decode(smiles,n_mol)
        qed_scores  = calc_qed(output_mols)
        sa_scores  = sa_func(output_mols)
        indices = np.argsort(qed_scores)[::-1]
        n_outputs += np.sum([sa_values[0] <= x <= sa_values for x in sa_scores])

        for indx in indices:
            if sa_scores[indx]<sa_values[0] or sa_scores[indx]>sa_values[1]:
                continue
            with col1:
                m = Chem.MolFromSmiles(output_mols[indx])
                im=Draw.MolToImage(m,size = (200,200),fitImage = True,useSVG=True)
                st.image(im)
            with col2:
                v_spacer(height=3)
                st.markdown(f" ### QED : {qed_scores[indx]:1.3f}")
                st.markdown(f" ### SA : {sa_scores[indx]:1.3f}")
                v_spacer(height=3)



