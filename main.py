from utils import *
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


def perf():
    performance("NetCal-DTI", NetCalDTI, 256, 5e-4)
    performance("GraphDTA", GraphDTA, 512, 0.0005)
    performance("DrugBAN", DrugBAN, 64, 5e-5)
    performance("ML_DTI", ML_DTI, 256, 1e-3)
    performance("HyperAttentionDTI", HyperAttentionDTI, 32, 5e-5)
    performance("TransformerCPI", TransformerCPI, 64, 1e-3)
    # use AdamW for TransformerCPI,weight_decay = 1e-4


if __name__ == "__main__":
    perf()
    ablation()