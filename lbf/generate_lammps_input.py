#Generate LAMMPS input script for simulating HBV dimers
#using an elastic network model with Go-like interactions

import argparse
import numpy as np
import MDAnalysis
import parmed as pmd

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', default='dimer.pdb',
                        help='PDB containing HBV dimer structure')
    parser.add_argument('--conn_file', default='connectivity.txt',
                        help='File containing residue connectivity (list of intra-dimer residue-residue bonds)')
    parser.add_argument('--lammps_file', default='dimer.in',
                        help='Name of LAMMPS input file (to be written to)')
    
    args = parser.parse_args()
    
    print(args.pdb)
    
    #Add connectivity information to pdb file if not there
    conn_data = np.loadtxt(args.conn_file)
    write_connectivity_to_pdb(conn_data, args.pdb)

    write_lammps_input_script()
    
def write_lammps_input_script():
    return 0

def write_connectivity_to_pdb(conn_data, pdb_file):
    '''
    Write out bond connectivity to a new pdb file along with CA positions.
    For now do this manually, eventually we should use MDAnalysis for more complicated situations
    '''
    new_pdb_file = pdb_file.replace('.pdb','_with_connectivity.pdb')
    with open(pdb_file,'r') as f:
        old_pdb_lines = f.readlines()
    with open(new_pdb_file,'w') as f:
        for line in old_pdb_lines[:-1]:
            f.write(line)
        for i in range(conn_data.shape[0]):
            f.write(f'CONECT{int(conn_data[i][0]):5d}{int(conn_data[i][1]):5d}\n')# % (conn_data[i][0], conn_data[i][1]))
        f.write(old_pdb_lines[-1])
    
    #Also create psf file
    #structure = pmd.load_file(new_pdb_file)
    #structure.save(new_pdb_file.replace('.pdb','.psf'),overwrite=True)

if __name__=='__main__':
    main()