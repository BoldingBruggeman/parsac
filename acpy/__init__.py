import subprocess
import argparse
import sys

from . import service
service.read()
from . import autocalibration
from . import sensitivity
from . import ensemble

def main():
    parser = argparse.ArgumentParser(description='AutoCalibration Python - acpy')
    parser.add_argument('--version', '-v', action='version')
    subparsers = parser.add_subparsers()

    parser_sa = subparsers.add_parser('sensitivity', help='Sensitivity analysis')
    sensitivity.configure_argument_parser(parser_sa)
    parser_sa.set_defaults(func=sensitivity.main)

    parser_ac = subparsers.add_parser('calibration', help='Auto calibration')
    autocalibration.configure_argument_parser(parser_ac)
    parser_ac.set_defaults(func=autocalibration.main)

    parser_ensemble = subparsers.add_parser('ensemble', help='Ensemble simulation')
    ensemble.configure_argument_parser(parser_ensemble)
    parser_ensemble.set_defaults(func=ensemble.main)

    parser_service = subparsers.add_parser('service', help='Service information')
    parser_service.set_defaults(func=service.main)

    args = parser.parse_args()
    if getattr(args, 'func', None) is None:
        print('acpy must be called with a subcommand. Use -h to see options')
        sys.exit(2)
    args.func(args)
