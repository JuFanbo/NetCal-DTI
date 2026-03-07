from utils import *
from NetCalDTI import NetCalDTI
from Benchmark.CNNGIN import CNNGIN
from Benchmark.attention import attention
from Benchmark.GraphDTA import GraphDTA
from Benchmark.DrugBAN import DrugBAN
from Benchmark.ML_DTI import ML_DTI
from Benchmark.HyperAttentionDTI import HyperAttentionDTI
from Benchmark.TransformerCPI import TransformerCPI
from Benchmark.le import le


def ablation():
    performance("0", le, 256, 5e-4, runs=10)
    performance("1", CNNGIN, 256, 5e-4, runs=10)
    performance("2", attention, 256, 5e-4, runs=10)
    performance("3", NetCalDTI, 256, 5e-4, runs=10)


def perf(target_cold,drug_cold):
    performance("NetCal-DTI", NetCalDTI,target_cold=target_cold,drug_cold=drug_cold,batch_size=256,lr=5e-4)
    performance("GraphDTA", GraphDTA,target_cold=target_cold,drug_cold=drug_cold,batch_size=512,lr=0.0005)
    performance("DrugBAN", DrugBAN, target_cold=target_cold,drug_cold=drug_cold,batch_size=64,lr=5e-5)
    performance("ML_DTI", ML_DTI, target_cold=target_cold,drug_cold=drug_cold,batch_size=256,lr=1e-3)
    performance("HyperAttentionDTI", HyperAttentionDTI, target_cold=target_cold,drug_cold=drug_cold,batch_size=32,lr=5e-5)
    performance("TransformerCPI", TransformerCPI, target_cold=target_cold,drug_cold=drug_cold,batch_size=64,lr=1e-3)

if __name__ == "__main__":
    performance("NetCal-DTI", NetCalDTI,drug_cold=True,batch_size=256,lr=5e-4)