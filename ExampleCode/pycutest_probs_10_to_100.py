"""
Isha Slavin.
"""

from __future__ import print_function
import pycutest

# all pycutest problems.
probs = pycutest.find_problems(constraints='U', userN=True)
# print('probs: ', probs)
probs = sorted(probs)

# find problems that are >= dimension 10 and <= dimension 100.
probs_10_to_100 = []
for p in probs:
    if p == 'ARGLINB':
        pass
    else:
        prob = pycutest.import_problem(p)
        print('prob: ', prob)
        x0 = prob.x0
        # only want <= 100.
        if 100 >= len(x0) >= 10:
            print(len(x0))
            probs_10_to_100.append(p)

print('probs under 100: ')
print(probs_10_to_100)

# write list to .txt file (inside this project!).
textfile = open("pycutest_probs_to_use.txt", "w")
# textfile.write(str(probs_10_to_100))
for element in probs_10_to_100:
    textfile.write(element + '\n')
textfile.close()
# the list of problems is now written into a .txt file.
# now in benchmarking_with_pycutest_2.py, we will read in the list from the .txt file!

"""
CHECK.
"""
new_probs_10_to_100 = []
f = open("pycutest_probs_to_use.txt", "r")
lines = f.readlines()
for line in lines:
    print(line)
    new_probs_10_to_100.append(line.rstrip())
print('\n')
print('new list: ')
print(new_probs_10_to_100)
