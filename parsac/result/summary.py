#!/usr/bin/env python
from __future__ import print_function

def configure_argument_parser(parser):
    parser.add_argument('xmlfile',       type=str, help='XML formatted configuration file')
    parser.add_argument('-g', '--groupby', type=str, choices=('source', 'run'), help='What identifier to group the results by, i.e., "source" or "run".')
    parser.add_argument('-o', '--orderby', type=str, choices=('count', 'lnl'), help='What property to order the result groups by, i.e., "count" or "lnl".')
    parser.set_defaults(range=None, orderby='count',  groupby='run')

def main(args):
    # Import custom modules
    from .. import result

    current_result = result.Result(args.xmlfile)

    res, source2history = current_result.get(groupby=args.groupby, constraints={}, run_id=None, limit=-1)
    run2source = current_result.get_sources()

    group2maxlnl = dict([(s,curres[:, -1].max()) for s, curres in source2history.items()])

    # Order sources (runs or clients) according to counts or ln likelihood.
    print('Points per %s:' % args.groupby)
    sources = source2history.keys()
    if args.orderby == 'count':
        sources = sorted(sources, key=lambda x: len(source2history[x]), reverse=True)
    else:
        sources = sorted(sources, key=lambda x: group2maxlnl[x], reverse=True)
    for source in sources:
        dat = source2history[source]
        label = source
        if args.groupby == 'run':
            label = '%4s %10s' % (source, run2source[source])
        print('  %s: %5i points, best lnl = %.10g' % (label, len(dat), group2maxlnl[source]))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)
