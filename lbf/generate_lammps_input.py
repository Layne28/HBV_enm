#Generate LAMMPS input script for simulating HBV dimers
#using an elastic network model with Go-like interactions

import argparse
import numpy as np
import MDAnalysis
import parmed as pmd
import math

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', default='dimer.pdb',
                        help='PDB containing HBV dimer structure')
    parser.add_argument('--conn_file', default='connectivity.txt',
                        help='File containing residue connectivity (list of intra-dimer residue-residue bonds)')
    parser.add_argument('--lammps_data_file', default='dimer_data.lammps',
                        help='Name of LAMMPS data file (to be written to)')
    parser.add_argument('--template_input_file', default='dimer.in',
                        help='Name of template LAMMPS input file')
    parser.add_argument('--lammps_input_file', default='dimer.in',
                        help='Name of LAMMPS input file (to be written to)')
    parser.add_argument('--spring_constant', default=10,
                        help='spring constant in kcal/mol/Angstrom^2')
    parser.add_argument('--concentration', default=100,
                        help='concentration in uM')
    parser.add_argument('--enm_dist_cutoff', default=8.0,
                        help='distance cutoff for adding bond between residues (Angstroms)')
    #Note: in future, this should be read from PDB file 
    #Or, we should create multiple dimers in this file based on
    #AB and CD pdb templates
    parser.add_argument('--ndimer', default=1,
                        help='Number of dimers')
    
    args = parser.parse_args()
    
    #Add connectivity information to pdb file if not there
    #conn_data = np.loadtxt(args.conn_file)
    conn_data = get_connectivity(args.pdb, args.enm_dist_cutoff)
    write_connectivity_to_pdb(conn_data, args.pdb)

    #Get position data from pdb file
    pos_data = extract_positions(args.pdb)
    print(pos_data)
    centered_pos_data = center_positions(pos_data)

    #Write LAMMPS data files
    write_lammps_harmonic_coeffs(args.lammps_data_file, conn_data, args)
    write_lammps_data_file(args.lammps_data_file, conn_data, centered_pos_data, args)

    #Write LAMMPS input script based on template
    write_lammps_input_script()

def center_positions(pos_data):
    com = np.mean(pos_data, axis=0)
    return pos_data-com

def extract_positions(pdb_file):
    with open(pdb_file) as f:
        lines = f.readlines()
    atoms = [line for line in lines if line.startswith('ATOM')]
    xpos = []
    ypos = []
    zpos = []
    for l in atoms:
        pieces = l.split()
        xpos.append(float(pieces[6]))
        ypos.append(float(pieces[7]))
        zpos.append(float(pieces[8]))
    return np.c_[xpos, ypos, zpos]

def extract_atom_ids(pdb_file):
    with open(pdb_file) as f:
        lines = f.readlines()
    atoms = [line for line in lines if line.startswith('ATOM')]
    ids = []
    for l in atoms:
        pieces = l.split()
        ids.append(int(pieces[1]))
    return ids

def get_connectivity(pdb_file, cutoff):
    
    ind1_list = []
    ind2_list = []
    #compute distances between CA within distance in pdb
    pos = extract_positions(pdb_file)
    for i in range(pos.shape[0]-1):
        for j in range(i+1, pos.shape[0]):
            r0 = float(np.linalg.norm(pos[j] - pos[i]))
            if r0<=cutoff:
                ind1_list.append(i+1)
                ind2_list.append(j+1)
                
    return np.c_[ind1_list,ind2_list]
                

def write_lammps_harmonic_coeffs(myfile, connectivity, params, recompute_connectivity=True):
    
    #If no connectivity provided, then compute it via a 
    #distance-based criterion.
    if recompute_connectivity==True:
        connectivity = get_connectivity(params.pdb, params.enm_dist_cutoff)

    ids = extract_atom_ids(params.pdb)
    pos = extract_positions(params.pdb)
    r0_list = []
    for i in range(connectivity.shape[0]):
        i1 = int(connectivity[i][0]-1)
        i2 = int(connectivity[i][1]-1)
        print(i, i1+1, i2+1)
        r0 = np.linalg.norm(pos[i2][:]-pos[i1][:])
        print(i, i1+1, i2+1, r0)
        r0_list.append(r0)
        
    outfile = myfile.replace('.lammps', '_harmonic_bond_coeffs.lammps')
    with open(outfile,'w') as f:
        f.write('#ENM harmonic bond coefficients\n')
        f.write('#format: bond_coeff ${bond_type} ${K}(kcal/mol/Ang^2) ${r0}(Ang)\n')
        f.write('\n')
        for i in range(connectivity.shape[0]):
            f.write('bond_coeff %d %.04f %.04f\n' % (i+1, float(params.spring_constant), r0_list[i]))

def write_lammps_data_file(myfile, connectivity, positions, params):
    '''
    Write LAMMPS data file (includes atoms and their positions, bonds, etc.)
    '''
    #Compute some things
    vol = params.ndimer/(6.022*10**23*10**-6*params.concentration) # in liters
    L = math.pow(vol, 1.0/3.0)*10**9 #decimeters to angstroms
    xlo = -L/2
    xhi = L/2
    n_CA_per_dimer = 298 #Cp149x2

    with open(myfile,'w') as f:
        f.write('LAMMPS data file: CG HBV dimer\n')
        f.write('\n')
        f.write('%d atoms\n' % n_CA_per_dimer*params.ndimer)
        f.write('%d bonds\n' % connectivity.shape[0])
        f.write('\n')
        f.write('1 atom types\n')
        f.write('%d bond types\n' % connectivity.shape[0])
        f.write('\n')
        f.write('%.04f %.04f xlo xhi\n' % (xlo, xhi))
        f.write('%.04f %.04f ylo yhi\n' % (xlo, xhi))
        f.write('%.04f %.04f zlo zhi\n' % (xlo, xhi))
        f.write('\n')
        f.write('Masses\n')
        f.write('\n')
        f.write('1 110.0\n') #this is the average mass of a single amino acid
        f.write('\n')

        #Write out particle positions
        f.write('Atoms # bond \n')
        f.write('\n')
        for i in range(n_CA_per_dimer):
            f.write('%d 1 1 %f %f %f\n' % (i+1, positions[i,0], positions[i,1], positions[i,2]))
        f.write('\n')

        #Write out bond information
        f.write('Bonds # (bond_id bond_type atom_id_1 atom_id_2)\n')
        f.write('\n')
        for i in range(connectivity.shape[0]):
            f.write('%d %d %d %d\n' % (i+1, i+1, connectivity[i][0], connectivity[i][1]))
        f.write('\n')
        
        # f.write('Bond Coeffs\n')
        # f.write('\n')
        # for i in range(connectivity.shape[0]):
        #     f.write('%d %.01f %.01f\n' % (i+1,params.spring_constant, 5.0))
        # #f.write('\n')
    
def write_lammps_input_script():
    return 0

def write_connectivity_to_pdb(conn_data, pdb_file):
    '''
    Write out bond connectivity to a new pdb file along with CA positions.
    For now do this manually, eventually we should use MDAnalysis for more complicated situations
    '''
    
    with open(pdb_file,'r') as f:
        old_pdb_lines = f.readlines()
        
    if not(any(l.startswith('CONECT') for l in old_pdb_lines)):
    
        new_pdb_file = pdb_file.replace('.pdb','_with_connectivity.pdb')
        with open(new_pdb_file,'w') as f:
            for line in old_pdb_lines[:-1]:
                f.write(line)
            for i in range(conn_data.shape[0]):
                f.write(f'CONECT{int(conn_data[i][0]):5d}{int(conn_data[i][1]):5d}\n')# % (conn_data[i][0], conn_data[i][1]))
            f.write(old_pdb_lines[-1])
        
        #Also create psf file
        #structure = pmd.load_file(new_pdb_file)
        #structure.save(new_pdb_file.replace('.pdb','.psf'),overwrite=True)
    
    else:
        print('PDB file already has CONECT information. Not adding any.')

if __name__=='__main__':
    main()