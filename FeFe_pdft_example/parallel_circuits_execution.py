import numpy as np
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit

def glue_circuits(QC_LIST):
  n_list = [qc.num_qubits for qc in QC_LIST]
  nqubit = sum(n_list)
  psi    = QuantumCircuit(nqubit) #glue circuits together with psi
  c_list = [ClassicalRegister(nx) for nx in n_list]
  for crx in c_list:
    psi.add_register(crx) 
  j = 0 #starting index of qubits in psi for each glued circuit
  for qcx,crx,nx in zip(QC_LIST,c_list,n_list):
    psi.append(qcx,range(j,j+nx))
    for ell in range(nx): psi.measure(j+ell, crx[ell]) #adds measurement
    j += nx
  return psi

def cut_results(R):
  k_list = [k.split() for k in R.keys()]
  nqubit = [len(s) for s in k_list[0]]
  r_list = []
  for x,nx in enumerate(nqubit):
    kx = list(set([k[x] for k in k_list]))
    rx = {kxa:0.0 for kxa in kx}
    r_list.append(rx)
  for k in k_list:
    for x,nx in enumerate(nqubit):
      r_list[x][k[x]] += R[' '.join(k)]
  return r_list[::-1]

# --------------------------------------------------------------

def random_bitstring_circuit(n,ghz=True):
  np.random.seed(n)
  qrg = QuantumRegister(n)
  psi = QuantumCircuit(qrg)
  if(ghz):
    psi.h(0)
    for i in range(1,n): psi.cx(i-1,i)
  psi.barrier()
  x = np.random.randint(2, size=(n,))
  for i,xi in enumerate(x):
    if(xi==1): psi.x(i)
  print("bitstring ",x)
  return psi
'''
qcs = [random_bitstring_circuit(n) for n in [2,4,6]]
#a list of circuits
print("This list of circuits...")
for qcx in qcs:
  print(qcx.draw())

qc = glue_circuits(qcs)

print("Has been glued into this circuit...")
print(qc.draw())

# --------------------------------------------------------------

simulator = AerSimulator(method='matrix_product_state')
qc = transpile(qc, simulator)
r = simulator.run(qc,shots=1024).result().get_counts(0)

# --------------------------------------------------------------

print("The results...")
print(r)

print("Have been cut into...")
rs = cut_results(r)
for rx in rs:
  print(rx)
'''
