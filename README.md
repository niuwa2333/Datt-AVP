# 🧬 AVPs Prediction using Deep Learning 🧬
## Project Overview
Welcome to our project where we harness the power of Deep Learning and Pre-trained Protein Language Models to predict potential Anti viral peptides(AVPs). By analyzing protein sequences, our model identifies proteins that may act as virulence factors, which is crucial for biomedical research and drug development.

### Features
Deep Learning Techniques: Utilizing state-of-the-art deep learning architectures to enhance prediction accuracy.

Pre-trained Protein Language Models: Leveraging pre-trained models to understand the language of proteins and make informed predictions.

User-Friendly: The User interface designed to be accessible with minimal setup required.

## Getting Started
To get started with our virulence factor prediction tool, you'll need a PC equipped with a GPU and Python installed. Follow these simple steps:

### Clone the Repository:
``` bash
git clone https://github.com/niuwa2333/DTVF.git
```

### Navigate to the Project Directory:
``` bash
cd DTVF
```

### Run the UI Script:
``` bash
python run.py 
```

## Data
We've included the training and testing datasets in the repository, pre-processed as .h5 format embeddings, to facilitate easy access and usage.

## Using this model to predict potential virulence factors

### Feature Extraction with ProtT5
We understand the importance of an intuitive workflow, especially when dealing with complex models like ProtT5. To this end, we've included a user-friendly Jupyter Notebook in our repository, get_embeddings.ipynb, which guides you through the process of extracting features from protein sequences using the ProtT5 model. These features are crucial for the operation of our virulence factor prediction tool.
We suggest using [Colab notebook](https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing) by ProtT5 authors
### Getting Started with get_embeddings.ipynb
1. Access the Notebook: Ensure you have Jupyter Notebook installed on your system. You can install it via pip if you haven't already:
``` bash
pip install notebook
```
2. Launch the Notebook: Navigate to the directory containing get_embeddings.ipynb and start Jupyter Notebook:
```
jupyter notebook
```
3. Open get_embeddings.ipynb: In the Jupyter interface, locate and open the get_embeddings.ipynb notebook.

4. Follow the Instructions: The notebook contains step-by-step instructions and code cells that will guide you through the feature extraction process. Simply execute each cell in sequence to generate embeddings for your protein sequences.

5. Save the Embeddings: Once the embeddings are generated, you can save them as a .h5 file using the provided code in the notebook.
### GPU Requirement
Please note that the feature extraction process with ProtT5 requires a GPU to ensure efficient and timely execution. If you do not have a GPU available, you may experience slow performance or be unable to complete the process.

## Security Note
The entire process of feature extraction and model prediction is designed to be performed locally on your device. This ensures that your data remains private and there is no risk of data leakage.

By following these steps, you can leverage the power of the ProtT5 model to generate embeddings for your protein sequences and utilize them with our virulence factor prediction tool.
## Acknowledgement
We would like to express our gratitude to the following projects and their contributors:
[ProtTrans](https://github.com/agemagician/ProtTrans)

[Gradio](https://github.com/gradio-app/gradio)



## Citation
``` bibtex
@article{sun2025datt,
  title={Datt-AVP: Antiviral Peptide Prediction by Sequence-Based Dual Channel Network with Attention Mechanism},
  author={Sun, Jiawei and Qian, Weiye and Ma, Nan and Liu, Wenjia and Yang, Zhiyuan},
  journal={IEEE Transactions on Computational Biology and Bioinformatics},
  year={2025},
  publisher={IEEE}
}
```
