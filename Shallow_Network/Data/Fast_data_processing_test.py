"""
Author: Searidang Pa 
"""
from numpy import save, load
import numpy as np
import sys
import os 
from pathlib import Path
np.set_printoptions(threshold=sys.maxsize)
from scipy import sparse 

"""
The test set is already normalized
"""


#source: http://education.expasy.org/student_projects/isotopident/htdocs/aa-list.html
dict_mono_mass = {'A': 71.03711, 'R': 156.10111, 'N': 114.04293, \
			 'D':115.02694, 'C':103.00919, 'E': 129.04259, \
			 'Q': 128.05858, 'G': 57.02146, 'H': 137.05891,\
			 'I': 113.08406, 'L': 113.08406, 'K': 128.09496, \
			 'M': 131.04049, 'F': 147.06841, 'P': 97.05276,\
			'S': 87.03203, 'T': 101.04768, 'W': 186.07931, \
			'Y': 163.06333, 'V': 99.06841 }
dict_one_hot = {'A': 0, 'R': 1, 'N': 2, 'D':3, 'C':4, \
			'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9, \
			'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,\
			'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19\
			}



Files = ["hcd_testingset.mgf"]
for filename in Files:

	if Path(filename.strip(".mgf")).is_dir() == False:
		os.makedirs(filename.strip(".mgf"))		#create directory for saving the processed data

	if filename == "hcd_testingset.mgf":
		Peptide_name_header = 'Title'

	elif filename == Files[0] or filename == Files[3]: 
		Peptide_name_header = 'SEQ'
	else:
		Peptide_name_header = 'Title'

	charge = 2
	chunk_size = 100
	chunk_index = 0

	skip_bool = True

	f = open(filename, "r")

	One_hot_encoding = np.zeros((chunk_size, 21, 25))
	Y = np.zeros((chunk_size, 20000))


	one_hot_temp = np.zeros((21,25))
	y_temp = np.zeros((1, 20000))


	count_peptide = 0

	num_data_clean= 0

	for line in f:	
		if line.strip() == 'BEGIN IONS': 
			one_hot_temp = np.zeros((21,25))
			y_temp = np.zeros((1, 20000))
		

		h = line.split('=')
		
		# parsing header 
		if  h[0] == 'CHARGE':
			if str(charge) in h[1] : 
				skip_bool = False

		elif h[0] == 'PEPMASS':
			Pepmass = float(h[1].strip())

		elif h[0] == Peptide_name_header: #one hot encoding
			col = 0
			for j in h[1].strip():	
				if j not in dict_mono_mass:
					skip_bool = True
					break
				else:
					if col < 25:
						one_hot_temp[dict_one_hot[j], col] = 1;
						one_hot_temp[20, col] = int(dict_mono_mass[j])
						col = col + 1;
	
		else:
			pass 

		#parsing the spectra
		k = line.split(" ")
		if len(k) >= 2 and k[0] != 'BEGIN' and k[0] != 'END':
			y_temp[0, int(round(float(k[0]), 1) *10)] = float(k[1])

	
		if line.strip() == 'END IONS':
		
			if skip_bool == False:
				#append data point to be saved
				One_hot_encoding[count_peptide, :, :] = one_hot_temp
				Y[count_peptide, :] = y_temp

				count_peptide += 1
				num_data_clean +=1
				
			if count_peptide == chunk_size:	#check if we want to flush out
				
				save(filename.strip(".mgf") +'/one_hot_encoding' + \
						str(chunk_index) + '.npy', One_hot_encoding)
			
				Y = sparse.csr_matrix(Y)

				PATH = filename.strip(".mgf")+'/bin_vs_intensity' + \
					str(chunk_index) + '.npz'

				sparse.save_npz(PATH, Y)

				chunk_index +=1

				#reset count_peptide and the buffer
				count_peptide = 0
				One_hot_encoding = np.zeros((chunk_size, 21, 25))
				Y = np.zeros((chunk_size, 20000))
			

			skip_bool = True 


	#---------------the smallest last chunk------------------------
	One_hot_encoding = One_hot_encoding[0: count_peptide, :, :]

	Y = Y[0: count_peptide, :]

	save(filename.strip(".mgf") +'/one_hot_encoding' + \
						str(chunk_index) + '.npy', One_hot_encoding)



	Y = sparse.csr_matrix(Y)

	PATH = filename.strip(".mgf")+'/bin_vs_intensity' + \
			str(chunk_index) + '.npz'
	sparse.save_npz(PATH, Y)
		
	save("num_data_clean_test.npy", num_data_clean)
	





