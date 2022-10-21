#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='show greetings')
parser.add_argument('firstname', type=str,
                    help='first name')
parser.add_argument('lastname', type=str,
                    help='last name')

args = parser.parse_args()

print(f"Hello, {args.firstname} {args.lastname}!")


