"""Shared bounded public command-line options."""
import argparse
import math


def bounded_number(name, low, high, *, integer=False):
    def parse(value):
        try:
            number = int(value) if integer else float(value)
        except (ValueError, TypeError):
            raise argparse.ArgumentTypeError(f'{name} must be a number in [{low}, {high}]') from None
        if not math.isfinite(number) or not low <= number <= high:
            raise argparse.ArgumentTypeError(f'{name} must be in [{low}, {high}]')
        return number
    return parse


def add_desktop_flags(parser):
    parser.add_argument('--pool-silence-timeout', type=bounded_number('pool-silence-timeout', 30, 1800), default=180)
    parser.add_argument('--on-battery', choices=('pause', 'run'), default='pause')
    parser.add_argument('--intensity', type=bounded_number('intensity', 10, 100, integer=True), default=100)
    parser.add_argument('--benchmark', nargs='?', const=60, type=bounded_number('benchmark', 10, 600, integer=True))
    parser.add_argument('--difficulty', type=bounded_number('difficulty', 1, 2**64), default=2**21)
