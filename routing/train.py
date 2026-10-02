import argparse
import json
from routing.model import train

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',required=True)
    parser.add_argument('--output',default='artifacts')
    parser.add_argument('--reports',default='reports')
    parser.add_argument('--min-class',type=int,default=20)
    args=parser.parse_args()
    result=train(args.csv,args.output,args.reports,args.min_class)
    print(json.dumps({k:result[k] for k in ['input_rows','usable_rows','split_rows','baseline','model','routing']},indent=2))
