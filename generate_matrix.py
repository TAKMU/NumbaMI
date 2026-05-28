import argparse
from mi_entropy import create_mi_matrix
import pandas as pd

parser = argparse.ArgumentParser(
                    prog='Create MI Matrix',
                    description='Create a Mutual Information matrix',
                    )
parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input file')
parser.add_argument('--output', '-o', type=str, required=True, help='Path to the output file')
parser.add_argument('--num_threads', '-n', type=int, default=-1, help='Number of threads to use')
args = parser.parse_args()
fname = args.input
output_file = args.output
num_threads = args.num_threads

print("Starting MI matrix generation...")
df = pd.read_csv(fname, sep=",", index_col=0)
mi_matrix = create_mi_matrix(df, num_threads)
mi_matrix.write_csv(output_file)

