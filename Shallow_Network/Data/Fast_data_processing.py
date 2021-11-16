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

#--------------declaring variables-----------
charge = 2
chunk_size = 100

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

Files = ["ProteomeTools.mgf", "NIST_Synthetic.mgf", \
			"NIST.mgf", "MassIVE.mgf"]

Num_data_clean = np.zeros((len(Files),1))
file_index = 0
chunk_index = 0

for filename in Files:
	#create directory for saving the processed data
	if Path(filename.strip(".mgf")).is_dir() == False:
		os.makedirs(filename.strip(".mgf"))		

	if filename == Files[0] or filename == Files[3]: 
		Peptide_name_header = 'SEQ'
	else:
		Peptide_name_header = 'Title'

	f = open(filename, "r")

	One_hot_encoding = np.zeros((chunk_size, 21, 25))
	Y = np.zeros((chunk_size, 20000))

	one_hot_temp = np.zeros((21,25))
	y_temp = np.zeros((1, 20000))
	Max_peak = np.zeros((chunk_size, 1));

	max_peak_temp = 0 
	count_peptide = 0
	num_data_clean= 0

	#skip all data except data with charge 2 that meet 
	#the two prepocessing conditions 
	skip_bool = True 

	for line in f:	
		if line.strip() == 'BEGIN IONS': 
			num_frag = 0
			Exact_mass = 0
			one_hot_temp = np.zeros((21,25))
			y_temp = np.zeros((1, 20000))
			max_peak_temp = 0 
			skip_bool = True

		# parsing header 
		h = line.split('=')

		if  h[0] == 'CHARGE':
			if str(charge) in h[1] : 
				skip_bool = False

		elif h[0] == 'PEPMASS':
			Pepmass = float(h[1].strip())

		elif h[0] == Peptide_name_header: #one hot encoding
			col = 0
			for j in h[1].strip():	

				#if the sequence contains some unknown variation, we skip it. 
				if j not in dict_mono_mass: 
					skip_bool = True
					break
				else:
					if col < 25:
						one_hot_temp[dict_one_hot[j], col] = 1;
						one_hot_temp[20, col] = int(dict_mono_mass[j])
						col = col + 1
						Exact_mass = Exact_mass + dict_mono_mass[j]

		else:	#don't care about the rest of the header
			pass 

		#parsing the spectra
		k = line.split('\t')
		if len(k) >= 2:
			num_frag = num_frag + 1

			if int(round(float(k[0]), 1) *10) < 20000:
				y_temp[0, int(round(float(k[0]), 1) *10)] = float(k[1])

				if max_peak_temp < float(k[1]):
					max_peak_temp = float(k[1])


		if line.strip() == 'END IONS':
			if skip_bool == False:
				precusor_mass_diff = abs(Pepmass - Exact_mass)/(Exact_mass * 106)
				if num_frag < 20 or num_frag > 500 or precusor_mass_diff > 200:		
					skip_bool = True

			if skip_bool == False:
				#append data point to be saved
				One_hot_encoding[count_peptide, :, :] = one_hot_temp
				Y[count_peptide, :] = y_temp
				Max_peak[count_peptide] = max_peak_temp

				count_peptide += 1
				num_data_clean +=1
			
			if count_peptide == chunk_size:	#check if we want to flush out
			
				save(filename.strip(".mgf") +'/one_hot_encoding' + \
						str(chunk_index) + '.npy', One_hot_encoding)

				Max_peak = np.repeat(Max_peak, 20000, axis = 1)
				Y = np.divide(Y, Max_peak) #normalization 
				Y = sparse.csr_matrix(Y)

				PATH = filename.strip(".mgf")+'/bin_vs_intensity' + \
					str(chunk_index) + '.npz'

				sparse.save_npz(PATH, Y)

				chunk_index +=1

				#reset count_peptide and the buffer
				count_peptide = 0
				One_hot_encoding = np.zeros((chunk_size, 21, 25))
				Y = np.zeros((chunk_size, 20000))
				Max_peak = np.zeros((chunk_size, 1));

			 


	#---------------the smallest last chunk------------------------
	One_hot_encoding = One_hot_encoding[0: count_peptide, :, :]
	Max_peak = Max_peak[0: count_peptide, :]
	Y = Y[0: count_peptide, :]

	save(filename.strip(".mgf") +'/one_hot_encoding' + \
						str(chunk_index) + '.npy', One_hot_encoding)

	Max_peak = np.repeat(Max_peak, 20000, axis = 1)
	Y = np.divide(Y, Max_peak)
	Y = sparse.csr_matrix(Y)

	PATH = filename.strip(".mgf")+'/bin_vs_intensity' + \
			str(chunk_index) + '.npz'
	sparse.save_npz(PATH, Y)
		
	Num_data_clean[file_index] = num_data_clean
	file_index += 1


save("Num_data_clean.npy", Num_data_clean)


