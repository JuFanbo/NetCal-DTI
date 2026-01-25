from utils import *
from benchmark import *
if __name__ == "__main__":
    performance("NetCal-DTI",NetCalDTI(),128,3e-4)
    performance("Attention", NetCalDTI(calibration=False), 128, 3e-4)
    performance("CNN_GNN", CNNGIN(),128,3e-4)
    performance("GraphDTA",GraphDTA(),512,0.0005)
    performance("DrugBAN",DrugBAN(),64,5e-5)
    performance("ML_DTI",ML_DTI(),256,1e-3)
    performance("HyperAttentionDTI",HyperAttentionDTI(),32,5e-5)
    performance("TransformerCPI",TransformerCPI(),64,1e-4)