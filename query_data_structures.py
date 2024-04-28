'''
Vineet Kumar, sioom.ai

python3 query_data_structures.py experiments/{number}

model_nums
('9000', '1500', '86', '200', '3500', '80', '911', '850', '2', '968', '240', '1', '928', '960', '57', '900', '164', '90', '500', '88', '62', '3', '100', '929', '300', '740', '5', '98', '2500', '940', '626', '323')

nonNumEntityWrds_per_entityLbl.keys()
dict_keys(['units_price_$', 'units_mileage_mi', 'restore', 'more', 'less', 'range1', 'range2', 'remove', 'brand', 'model', 'color', 'assoc_brand_modelNum', 'multilabel', 'other', 'entWrdLblOther'])

nonNumEntityWrds_per_entityLbl['assoc_brand_modelNum']['items']
[('chevrolet', '2500'), ('Chrysler', '200'), ('Porsche', '968'), ('Porsche', '928'), ('Oldsmobile', '98'), ('Volvo', '960'), ('Alfa Romeo', '164'), ('Chrysler', '300'), ('Volvo', '940'), ('chevrolet', '3500'), ('Polestar', '2'), ('Toyota', '86'), ('MAZDA', '626'), ('Polestar', '3'), ('Volvo', '850'), ('Audi', '100'), ('MAZDA', '323'), ('Oldsmobile', '88'), ('MAZDA', '929'), ('Audi', '90'), ('audi', '5'), ('chevrolet', '1500'), ('Maybach', '57'), ('FIAT', '500'), ('Saab', '900'), ('Saab', '9000'), ('Audi', '80'), ('Volvo', '240'), ('gmc', '1500'), ('Volvo', '740'), ('Maybach', '62'), ('chrysler', '300'), ('Polestar', '5'), ('Porsche', '911'), ('Polestar', '1')]

nonNumEntityWrds_per_entityLbl['multilabel']['items']
[('brand', 'genesis'), ('model', 'genesis'), ('brand', ('genwsis', 'genesis')), ('model', ('genwsis', 'genesis')), ('brand', ('yenesis', 'genesis')), ('model', ('yenesis', 'genesis')), .......('brand', ('genesos', 'genesis')), ('model', ('genesos', 'genesis'))]

nonNumEntityWrds_per_entityLbl['entWrdLblOther']['items']
['dollars', 'between', 'mileage', 'above', 'make', 'under', 'larger', 'more', 'mi', 'dollar', 'little', 'over', 'lower', 'mileages', '-', 'price', 'miles', 'below', 'brand', 'color', 'smaller', '$', 'greater', 'year', 'mile', 'model', 'less', 'through', 'to', 'range', 'higher']

'toyota' in nonNumEntityWrds_per_entityLbl['brand']['items']
True
------------------------------------------------------------------------------
Usage:
* user-input can be any python expression
    * currently a word (e.g. 'toyota') or a tuple (e.g. ('chevrolet', '2500') )
        is implemented
***NOTE: It is best for a user to input a one-word string (as opposed to a
    multiple word string or a tuple) because this program will then find all
    the strings that have this given one-word***
------

'''
# python3 query_data_structures.py experiments/{number}

from sys import argv
from pathlib import Path
import textwrap
import pickle
from ast import literal_eval

dataframes_dirPath = Path(argv[len(argv) - 1]).resolve(strict=True)
data_structures_file = dataframes_dirPath.joinpath('data_structures.pickle')
with data_structures_file.open('rb') as file:
    # the file has multiple data-structures, so each must be loaded
    # separately; these data-structures are in the same order as that
    # in the file generate_dataset/Fill_entityWrds.py when they were pickled
    nonNumEntityWrds_per_entityLbl = pickle.load(file)
    model_nums = pickle.load(file)  # not used

entityWrds_from_entityLbl = []
try:
    while True:
        strng = input("Enter entityWrd>")
        try:
            strng = literal_eval(strng)
        except (ValueError, SyntaxError):
            print(f"invalid user-input: {strng}")
            continue

        for entityLbl in nonNumEntityWrds_per_entityLbl:
            entityWrds_from_entityLbl.clear()
            for entityWrd_pair in nonNumEntityWrds_per_entityLbl[entityLbl][
                    'items']:
                if isinstance(strng, str):
                    if isinstance(entityWrd_pair,
                                  tuple) and len(entityWrd_pair) == 2:
                        if (strng == entityWrd_pair[0]
                                or strng == entityWrd_pair[1]):
                            entityWrds_from_entityLbl.append(
                                str(entityWrd_pair))
                    elif isinstance(entityWrd_pair, str):
                        entityWrd_pair_lst = entityWrd_pair.split()
                        for strg in strng.split():
                            if strg in entityWrd_pair_lst:
                                entityWrds_from_entityLbl.append(
                                    str(entityWrd_pair))
                                break
                    else:
                        msg = (f"invalid {entityWrd_pair} in "
                               f"nonNumEntityWrds_per_entityLbl[{entityLbl}]")
                        assert False, msg
                elif isinstance(strng, tuple):
                    if isinstance(entityWrd_pair, tuple):
                        if (strng == entityWrd_pair):
                            entityWrds_from_entityLbl.append(
                                str(entityWrd_pair))
                    elif isinstance(entityWrd_pair, str):
                        pass
                    else:
                        msg = (f"invalid {entityWrd_pair} in "
                               f"nonNumEntityWrds_per_entityLbl[{entityLbl}]")
                        assert False, msg
                else:
                    assert False, f"invalid user-input: {strng}"
            if entityWrds_from_entityLbl:
                print(f"entityWrds belonging to '{entityLbl}'")
                print(
                    textwrap.fill(", ".join(entityWrds_from_entityLbl),
                                  width=80,
                                  initial_indent=5 * " ",
                                  subsequent_indent=5 * " "))
except KeyboardInterrupt:
    pass
