"""Prepare missing Markdown catalogue calls; run only after cost approval."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import markdown_catalogue as catalogue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('prepare')
    plan.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('plan', type=Path)
    run.add_argument('--budget-usd', type=float, required=True)
    review = sub.add_parser('accept')
    review.add_argument('stem')
    review.add_argument('--reviewer', required=True)
    review.add_argument('--notes', required=True)
    args = p.parse_args()
    if args.action == 'prepare':
        data = catalogue.prepare()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in data.items() if k != 'requests'}, indent=2))
    elif args.action == 'run':
        from dotenv import load_dotenv
        load_dotenv(catalogue.ROOT.parent / '.env')
        result = catalogue.execute(json.loads(args.plan.read_text()), args.budget_usd)
        print(json.dumps({'requests': len(result), 'skipped_existing': sum(bool(r.get('skipped')) for r in result),
                          'cost_usd': sum(r.get('cost_usd', 0) for r in result),
                          'errors': sum(bool(r.get('error')) for r in result),
                          'unfinished_attempts': sum('raw_text' not in r and not r.get('skipped') for r in result)}, indent=2))
    else:
        print(catalogue.accept(args.stem, args.reviewer, args.notes))


if __name__ == '__main__':
    main()
