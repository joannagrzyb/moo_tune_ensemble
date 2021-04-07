# from tabulate import tabulate
from texttable import Texttable

import latextable

rows = [['Rocket', 'Organisation', 'LEO Payload (Tonnes)', 'Maiden Flight'],
        ['Saturn V', 'NASA', '140', '1967'],
        ['Space Shuttle', 'NASA', '24.4', '1981'],
        ['Falcon 9 FT-Expended', 'SpaceX', '22.8', '2017'],
        ['Ariane 5 ECA', 'ESA', '21', '2002']]

table = Texttable()
table.set_cols_align(["c"] * 4)
# table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.set_deco(Texttable.HEADER)
table.add_rows(rows)

# print('Tabulate Table:')
# print(tabulate(rows, headers='firstrow'))

print('\nTexttable Table:')
print(table.draw())

# print('\nTabulate Latex:')
# print(tabulate(rows, headers='firstrow', tablefmt='latex'))

print('\nTexttable Latex:')
print(latextable.draw_latex(table, caption="A comparison of rocket features."))
