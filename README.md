
**Dataset**
Because of the large size of the datasets, we cannot submit our code with the dataset. The 4 training dataset ProteomeTools.mgf, NIST_Synthetic.mgf, NIST.mgf, and MassIVE.mgf can be downloaded at https://scholarworks.iu.edu/dspace/handle/2022/24023. The test set can be downloaded at https://scholarworks.iu.edu/dspace/handle/2022/25283. 

**Data Processing**
Put in a folder named ‘Data’. Then, run python3 Fast_data_processing.py. It will generate many files, each in a folder named after its dataset. For example, bin_vs_intensity0.npz (which is the target of our model) and one_hot_encoding0.npy (which is our inputs) contains the first 100 clean data points. Furthermore, bin_vs_intensity1.npz  and one_hot_encoding1.npy contains the next 100 clean data points, so on and so forth. 


**Shallow Network**
Model4_ED is trained using euclidean distance loss (MSE). Model4_Cosine.pt is the weight of the network with the same architecture as model4_ED, but trained using cosine similarity measure. Because the weight model4_ED.pt and model4_Cosine.pt are large (several Gbs) and because I (Dang) have limited internet, I will refer you to my other teammates’ code for the weights of the model. 


**PeptideNet.ipynb**
This Jupyter Notebook file consists of code to create and run the architecture of PeptideNet cnn based on predfull network. Data is generated into Data Generator explicitly coded to handle the data of this kind which is generated using the Fast_Data_Preprocessing.py and consist of some visualization of network’s training phase along with prediction of spectrum with best sets of weights.

* To run the script that test the perform evaluation on the test set: run **Python3 Testset_eval.py** 

**Predictions Visualization**
Two spectra predictions from each Shallow CNN and one spectra prediction from Normal CNN are visualized compared with the experimental spectra using visualization.Rmd. Blue spectra are experimental spectra, and green spectra are predicted spectra. The data for visualization is saved by saving the corresponding predicted intensity and experimental intensity of three random-chosen sequences. 
