import uproot
import numpy as np
from scipy import sparse
import scipy.sparse
from pathlib import Path
import pdb
import os
def loadsig(name):
	sig_root = uproot.open(name)
	qtree_sig = sig_root["qtree"]
	TX_sig = qtree_sig["TrackPostX"].array(library = 'np') # in alternativa library = 'pd' 
	TY_sig = qtree_sig["TrackPostY"].array(library = 'np')
	TZ_sig = qtree_sig["TrackPostZ"].array(library = 'np')
	TrackEn_sig = qtree_sig["TrackEnergy"].array(library = 'np') # Energy in every simulation step
	NTrack_sig = qtree_sig["NTrack"].array(library = 'np') # Number of step of the simulation
	DepEn_sig = qtree_sig["DepositedEnergy"].array(library = 'np') # Total energy

# The 2 electrons of the bb are stored in 2 consecutive entries
	TXsig = TX_sig[range(0, len(TX_sig), 2)]
	TX2 = TX_sig[range(1, len(TX_sig), 2)]
	TYsig = TY_sig[range(0, len(TY_sig), 2)]
	TY2 = TY_sig[range(1, len(TY_sig), 2)]
	TZsig = TZ_sig[range(0, len(TZ_sig), 2)]
	TZ2 = TZ_sig[range(1, len(TZ_sig), 2)]

	TrackEnsig = TrackEn_sig[range(0, len(TX_sig), 2)]
	TrackEn2 = TrackEn_sig[range(1, len(TX_sig), 2)]
	
	DepEnsig = DepEn_sig[range(0, len(TX_sig), 2)]
	DepEn2 = DepEn_sig[range(1, len(TX_sig), 2)]
	
	NTracksig = NTrack_sig[range(0, len(TX_sig), 2)]
	NTrack2 = NTrack_sig[range(1, len(TX_sig), 2)]
	

	dim = round(len(TX_sig)/2)

	for i in range(0, dim):
	    TXsig[i] = np.append(TXsig[i], TX2[i])
	    TYsig[i] = np.append(TYsig[i], TY2[i])
	    TZsig[i] = np.append(TZsig[i], TZ2[i])
	    TrackEnsig[i] = np.append(TrackEnsig[i], TrackEn2[i])
	    DepEnsig[i] = DepEnsig[i] + DepEn2[i]
	    NTracksig[i] = NTracksig[i] + NTrack2[i]
	    
	
	return TXsig, TYsig, TZsig, TrackEnsig, DepEnsig, NTracksig

def loadbkg(name):
	bkg_root = uproot.open(name)
	qtree_bkg = bkg_root["qtree"]

	TXbkg = qtree_bkg["TrackPostX"].array(library = 'np') # in alternativa library = 'pd' 
	TYbkg = qtree_bkg["TrackPostY"].array(library = 'np')
	TZbkg = qtree_bkg["TrackPostZ"].array(library = 'np')
	TrackEnbkg = qtree_bkg["TrackEnergy"].array(library = 'np')
	DepEnbkg = qtree_bkg["DepositedEnergy"].array(library = 'np')
	NTrackbkg = qtree_bkg["NTrack"].array(library = 'np')
	
	return TXbkg, TYbkg, TZbkg, TrackEnbkg, DepEnbkg, NTrackbkg

def degrader(TX, TY, TZ, E):
	print("Converting to histogram...")
	resx = 5
	resy = 5
	resz = 1
	npart = len(TX)
	binspanX = np.arange(-50, 50, resx)
	binspanY = np.arange(-50, 50, resy)
	binspanZ = np.arange(-50, 50, resz)
	
	Data = np.zeros((npart, 20, 20, 100))
	for particle in range(0, npart):
		TX[particle] = TX[particle] - TX[particle][0]
		TY[particle] = TY[particle] - TY[particle][0]
		TZ[particle] = TZ[particle] - TZ[particle][0]
		
		TracBin = np.zeros((20, 20, 100)) # Single particle histogram

		tx = TX[particle]
		ty = TY[particle]
		tz = TZ[particle]
			
		X = np.digitize(tx, binspanX[:-1])
		Y = np.digitize(ty, binspanY[:-1])
		Z = np.digitize(tz, binspanZ[:-1])
		
		for i in range(len(tx)-1):
			TracBin[X[i], Y[i], Z[i]] = TracBin[X[i], Y[i], Z[i]] + E[particle][i]
		
		if np.abs(np.sum(TracBin) - np.sum(E[particle])) > 1e-3:
			print(np.sum(TracBin))
			print(np.sum(E[particle]))
		Data[particle] = TracBin

	Data_r = Data.reshape(npart*20*20, 100)
	Data_sparse = sparse.csr_matrix(Data_r)
	return Data_sparse
	
	
#------------------------------------------------------------------------------------------------

inputfolder = Path("./root files")
outputfolder = Path("./data3")
for file in inputfolder.iterdir():
	if file.suffix == ".root":
		name = os.path.basename(file)
		name = os.path.splitext(name)[0]
		print("opening "+name)
		is_signal = file.name[0] == "b"
		is_background = file.name[0] == "e"
		if is_signal:
			TX, TY, TZ, E, Etot, N = loadsig(file)
		elif is_background:
			TX, TY, TZ, E, Etot, N = loadbkg(file)
			
		Data_sparse = degrader(TX, TY, TZ, E)
		scipy.sparse.save_npz("data3/"+name, Data_sparse)
