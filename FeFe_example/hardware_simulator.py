import ffsim
from datetime import datetime
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Session  # , Options
from qiskit_ibm_runtime import SamplerV2 as Sampler


class hardware_simulator:
    def __init__(self):
        dt = datetime.now().isoformat().replace(" ", "_")
        self.args = {}
        self.args["optimization_level"] = 3
        self.args["dynamical_decoupling"] = True
        self.args["twirling"] = True
        self.args["shots"] = 100000
        self.args["initial_layout"] = None
        self.args["device_name"] = ""
        self.args["logfile"] = "%s_logfile.txt" % dt
        self.args["circuits_per_job"] = 300
        self.args["provider"] = IBMProvider()

    def get_service(self, instance="quantum-demonstrations/main/qsci"):
        return QiskitRuntimeService(
            channel="ibm_cloud",
            token="Y7MltCDIjlpYfECD24IJFT-YE5P78bptUmYFelvHKYW-",
            instance="crn:v1:bluemix:public:quantum-computing:us-east:a/3c257b5d547d42acbf496bcc62136e93:83362ed3-683d-4465-b4ab-1760d2286fa0::",
        )
        # return QiskitRuntimeService(channel='ibm_cloud', instance=instance)

    def print_arguments(self):
        logfile = open(self.args["logfile"].replace("logfile", "info"), "w")
        for k in self.args.keys():
            logfile.write("%s %s \n" % (k, str(self.args[k])))

    def transpile_circuits(self, circuits, service):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        backend = service.backend(self.args["device_name"])
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=self.args["optimization_level"],
            initial_layout=self.args["initial_layout"],
        )
        pm.pre_init = ffsim.qiskit.PRE_INIT
        print("starting transpilation ")
        qc_pm = []
        for jqc, qc in enumerate(circuits):
            print("transpiling circuit ", jqc)
            transpiled_qc = pm.run(qc)
            qc_pm.append(transpiled_qc)
            print("circuit transpiled  ")
        return qc_pm

    def set_options(self, sampler):
        sampler.options.default_shots = self.args["shots"]
        if self.args["dynamical_decoupling"]:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        if self.args["twirling"]:
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
        # sampler.options.transpilation.skip_transpilation = False
        sampler.options.max_execution_time = 10800
        sampler.options.update(default_shots=sampler.options.default_shots)
        return sampler

    # sim = True, don't sumbit jobs
    def submit_jobs(self, circuits, service):
        circuits = self.transpile_circuits(circuits, service)
        # options = self.set_options()
        with Session(service=service, backend=self.args["device_name"]) as session:
            logfile = open(self.args["logfile"], "w")
            job_indices = [
                k // self.args["circuits_per_job"] for k, ck in enumerate(circuits)
            ]
            for j in list(set(job_indices)):
                cj = [ck for k, ck in enumerate(circuits) if job_indices[k] == j]
                sampler = Sampler(mode=session)  # fakesampler her
                sampler = self.set_options(sampler)
                print(sampler.options)
                job = sampler.run(cj)
                print("submitted job %s " % job._job_id)
                logfile.write("%s \n" % job._job_id)
                with open("job_id.txt", "w") as f:
                    f.write(job._job_id)
        del service

    def read_from_file(self, fname):
        f = open(fname, "r").readlines()
        f = [(fi.split()[0], fi.split()[1:]) for fi in f]
        for k in self.args.keys():
            values_for_k = [vi for ki, vi in f if ki == k]
            if values_for_k:  # Only set if there's at least one match
                self.args[k] = values_for_k[0]
            else:
                print(f"Warning: No value found for key '{k}' in file '{fname}'")
            # self.args[k] = [vi for ki,vi in f if ki==k][0]
        self.args["optimization_level"] = int(self.args["optimization_level"][0])
        self.args["dynamical_decoupling"] = bool(self.args["dynamical_decoupling"][0])
        self.args["twirling"] = bool(self.args["twirling"][0])
        self.args["shots"] = int(self.args["shots"][0])
        layout = self.args["initial_layout"]
        layout = [
            fi.replace(",", "").replace("]", "").replace("[", "") for fi in layout
        ]
        layout = [int(fi) for fi in layout]
        self.args["initial_layout"] = layout
        self.args["device_name"] = self.args["device_name"][0]
        self.args["logfile"] = self.args["logfile"][0]
        self.args["circuits_per_job"] = int(self.args["circuits_per_job"][0])
        self.args["provider"] = IBMProvider()

    def retrieve(self, num_qubits):
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        results = []

        def get_bin(x, n):
            return format(x, "b").zfill(num_qubits)

        for job_id in [f.rstrip() for f in open(self.args["logfile"], "r").readlines()]:
            result = service.job(job_id=job_id).result()
            R = []
            for i in range(len(result)):
                result_i_dict = result[i].data.__dict__
                k = [key for key in result_i_dict.keys()][0]
                counts_i = result_i_dict[k].get_counts()
                R.append(counts_i)
            print("job_id ", job_id, " done ")
            results += R
        return results
