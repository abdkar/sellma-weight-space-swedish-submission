"""Small command-line interface; the guided notebook calls the same functions."""
import argparse
from pathlib import Path
from .common import load_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    data=commands.add_parser("data"); data.add_argument("--out",default="data")
    prepare=commands.add_parser("prepare"); prepare.add_argument("--models",required=True)
    prepare.add_argument("--root",default=".")
    run=commands.add_parser("run"); run.add_argument("--config",required=True)
    run.add_argument("--group",choices=["original","patched","sign","orthogonal"])
    analyze=commands.add_parser("analyze"); analyze.add_argument("--config",required=True)
    args=parser.parse_args()
    if args.command=="data":
        from .data import write_dataset
        write_dataset(args.out)
    elif args.command=="prepare":
        from .weights import prepare
        prepare(args.root,args.models)
    elif args.command=="run":
        from .experiment import run
        print(run(load_json(args.config),[args.group] if args.group else None))
    else:
        from .analysis import analyze
        summary,effects,coverage=analyze(load_json(args.config))
        print(summary.to_string(index=False));print(effects.to_string(index=False));print(coverage)

if __name__=="__main__": main()
