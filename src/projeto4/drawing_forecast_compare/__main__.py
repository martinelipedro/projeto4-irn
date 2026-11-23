import argparse
from ..io import ROOT,read_json
from .data import prepare


def main():
    parser=argparse.ArgumentParser(description='Benchmark de previsão: GRU pura × DeepSets sem recorrência.')
    parser.add_argument('stage',choices=['prepare','train','evaluate','export','all'],nargs='?',default='all')
    parser.add_argument('--config',default=str(ROOT/'configs/drawing_forecast_compare.json'))
    parser.add_argument('--download-raw',action='store_true')
    args=parser.parse_args();config=read_json(args.config)
    if args.stage=='prepare':prepare(config,args.download_raw)
    if args.stage in ['train','all']:
        from .train import search
        search(config)
    if args.stage in ['evaluate','all']:
        from .train import evaluate
        evaluate(config)
    if args.stage in ['export','all']:
        from .export import export
        export(config)


if __name__=='__main__':main()
