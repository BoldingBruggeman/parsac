import subprocess
import argparse

from . import license
from . import run
from . import result

#    try:
#        ver = subprocess.check_output(['git', 'describe', '--tags'])
#        break
#    except CalledProcessError:
#        ver = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='.')
def get_sha():
#    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='.')
    return '0.1'

def main():
    parser = argparse.ArgumentParser(description='AutoCalibration Python - acpy')
    parser.add_argument('--version', '-v', action='version', version='git sha: %s' % get_sha())
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run', help='Run the auto-calibration')
    run.configure_argument_parser(parser_run)
    parser_run.set_defaults(func=run.main)

    parser_plot = subparsers.add_parser('plot', help='Plot maximum likelihood for the various parameters being estimated')
    result.plot.configure_argument_parser(parser_plot)
    parser_plot.set_defaults(func=result.plot.main)

    parser_plotbest = subparsers.add_parser('plotbest', help='Plot statistical results - default for the best parameter set')
    result.plotbest.configure_argument_parser(parser_plotbest)
    parser_plotbest.set_defaults(func=result.plotbest.main)

    parser_animate_2d = subparsers.add_parser('animate_2d', help='Generate sequence of X-Y plots showing parameter variations - as .pngs')
    result.animate_2d.configure_argument_parser(parser_animate_2d)
    parser_animate_2d.set_defaults(func=result.animate_2d.main)

    parser_license = subparsers.add_parser('license', help='Print license information')
#    license.configure_argument_parser(parser_license)
    parser_license.set_defaults(func=license.main)


    args = parser.parse_args()
    args.func(args)


#if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #configure_argument_parser(parser)
    #args = parser.parse_args()
    #main(args)

