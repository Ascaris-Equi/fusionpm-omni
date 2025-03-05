# fusionpm-omni
✨ fusionpm-omni: A versatile MHC-peptide binding prediction framework supporting diverse species (including human, rhesus macaque, chimpanzee, mouse, dog, pig, horse, and more) and compatible with both MHC class I &amp; II via pseudo-sequence extraction.

## Installation

    pip install torch pandas numpy

## Usage

Prepare your input file input.csv with Peptide and Pseudo_Sequence columns, then run:

    python inference.py --input input.csv --output output.csv --model best_model.pth

## Features

    Multi-species support: Compatible with human, mouse, dog, pig, and more.
    Supports both MHC class I & II: Achieved via pseudo-sequence extraction.
    Transformer-based model: High accuracy and reliability.

Author：All rights reserved.
