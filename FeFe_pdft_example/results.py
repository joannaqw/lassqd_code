from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_cloud", token='Y7MltCDIjlpYfECD24IJFT-YE5P78bptUmYFelvHKYW-', instance='crn:v1:bluemix:public:quantum-computing:us-east:a/3c257b5d547d42acbf496bcc62136e93:83362ed3-683d-4465-b4ab-1760d2286fa0::')

# Save account to disk and save it as the default.
#QiskitRuntimeService.save_account(channel="ibm_cloud", token="<IBM Cloud API key>", instance="<IBM Cloud CRN>", name="account-name", set_as_default=True)
print(service.least_busy(operational=True, min_num_qubits=30,reservation_lookahead=600,simulator=False))
# Load the saved credentials
#service = QiskitRuntimeService(name="account-name")
exit()



service = QiskitRuntimeService(
    channel='ibm_cloud',
    instance='Qiskit Runtime-lassqd ',
    token='crn:v1:bluemix:public:quantum-computing:us-east:a/3c257b5d547d42acbf496bcc62136e93:83362ed3-683d-4465-b4ab-1760d2286fa0::'
    #'Y7MltCDIjlpYfECD24IJFT-YE5P78bptUmYFelvHKYW-'
    #'0ee91a25538c16cbdfbad65fa17c990908455d7091ddbb6b0915d7ddffb4cbb1e0203e4cee0fd9a290adbf3b987b40f1184039c5415d477ed64033bafa04c0b2'
)
#job = service.job('cxjzv043ej4g008gfnhg')
exit()
print(job.error_message())
job_result = job.result()
counts_list = job.result().get_counts()
for counts in counts_list:
    print(counts)

# To get counts for a particular pub result, use 
#
# pub_result = job_result[<idx>].data.<classical register>.get_counts()
#
# where <idx> is the index of the pub and <classical register> is the name of the classical register. 
# You can use circuit.cregs to find the name of the classical registers.
