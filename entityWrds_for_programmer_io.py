'''
Vineet Kumar, sioom.ai
'''
# python3 entityWrds_for_programmer_io.py experiments/{number}

from sys import argv
from pathlib import Path
import textwrap
import pickle

dataframes_dirPath = Path(argv[len(argv) - 1]).resolve(strict=True)
entityWrds_for_programmer_io_file = dataframes_dirPath.joinpath(
    'entityWrds_for_programmer_io')
with entityWrds_for_programmer_io_file.open('rb') as file:
    nonNumEntityWrds_per_entityLbl = pickle.load(file)

entityWrds_from_entityLbl = []
try:
    while True:
        strng = input("Enter entityWrd>")
        for entityLbl in nonNumEntityWrds_per_entityLbl:
            entityWrds_from_entityLbl.clear()
            for entityWrd_pair in nonNumEntityWrds_per_entityLbl[entityLbl][
                    'nonNumEntityWrds']:
                if isinstance(entityWrd_pair,
                              tuple) and len(entityWrd_pair) == 2:
                    if (strng == entityWrd_pair[0]
                            or strng == entityWrd_pair[1]):
                        entityWrds_from_entityLbl.append(str(entityWrd_pair))
                elif isinstance(entityWrd_pair, str):
                    if strng == entityWrd_pair:
                        entityWrds_from_entityLbl.append(str(entityWrd_pair))
                else:
                    assert False, f"invalid {entityWrd_pair} in {entityWrds_for_programmer_io_file}"
            if entityWrds_from_entityLbl:
                print(f"entityWrds belonging to {entityLbl}")
                print(
                    textwrap.fill(",".join(entityWrds_from_entityLbl),
                                  width=80,
                                  initial_indent=5 * " ",
                                  subsequent_indent=5 * " "))
except KeyboardInterrupt:
    pass
