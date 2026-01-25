from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import tempfile
import os
import re
import pandas as pd
from func_timeout import func_set_timeout
def extract_best_affinity(vina_output):
    # 使用正则表达式匹配模式表格行
    pattern = r"^\s*(\d+)\s+([-]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)"

    # 按行分割输出
    lines = vina_output.splitlines()

    # 寻找表格开始位置
    table_started = False
    for i, line in enumerate(lines):
        if "mode |   affinity | dist from best mode" in line:
            # 表格头行后两行是数据行（跳过分隔线）
            data_lines = lines[i + 2:]
            break
    else:
        raise ValueError("未找到结果表格")

    # 解析第一行数据（最佳构象）
    for line in data_lines:
        match = re.match(pattern, line)
        if match:
            # 提取结合能值（第二列）
            affinity = float(match.group(2))
            return affinity

    raise ValueError("未在输出中找到结合能数据")



def smiles_to_pdbqt(smiles):
    """从SMILES生成PDBQT文件，只保留第一个构象"""

    # 创建临时文件
    temp_file = "temp_obabel.pdbqt"

    # 使用OpenBabel转换
    subprocess.run(
        ["obabel", "-:" + smiles, "-opdbqt", "-O" + temp_file, '--partialcharge','gasteiger',"--steps","1000","-h","--ff","MMFF94","--gen3d"],
        check=True,
        capture_output=True
    )

    # 读取生成的文件，只保留第一个构象
    with open(temp_file, 'r') as f:
        lines = f.readlines()

    # 找到第一个构象的结束位置
    first_conformer_end = 0
    seen_remark_name = False

    for i, line in enumerate(lines):
        if "REMARK  Name =" in line:
            if seen_remark_name:
                first_conformer_end = i
                break
            else:
                seen_remark_name = True
        elif i == len(lines) - 1:
            first_conformer_end = len(lines)

    # 只保留第一个构象
    with open("temp.pdbqt", 'w') as f:
        f.writelines(lines[:first_conformer_end])

    # 删除临时文件
    os.remove(temp_file)

    return "temp.pdbqt"

@func_set_timeout(10)
def get_affinity(smiles):
    smiles_to_pdbqt(smiles)
    result = subprocess.run(f'./qvina02 --config config.txt', shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Vina执行失败: {result.stderr}")
    return extract_best_affinity(result.stdout)
lst = pd.read_csv('result.csv').values.tolist()
output = []
lst = [i for i in lst if i[3] != 'yes']
r = lst[:100]
for i in r:
    try:
        toadd = i + [get_affinity(i[4])]
        print(toadd)
        output += [toadd]
        df = pd.DataFrame(output)
        df.to_csv('top100_novel.csv', index=False)
    except:
        continue
