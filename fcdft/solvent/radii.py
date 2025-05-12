import numpy
from pyscf.lib.parameters import BOHR

unknown = 1.999999 # This should be deprecated once 'del unknown' in pyscf.data.radii is deleted.

# Retrived from Jaguar 9.0 Manual
VDWJAG = 1/BOHR * numpy.array((unknown,  # Ghost atom
    1.150,      #  1 H
    1.181,      #  2 He [1]
    1.226,      #  3 Li [1]
    1.373,      #  4 Be [5]
    2.042,      #  5 B  [5]
    1.900,      #  6 C  [1]
    1.600,      #  7 N  [1]
    1.600,      #  8 O  [1]
    1.682,      #  9 F  [1]
    1.621,      # 10 Ne [1]
    1.491,      # 11 Na [1]
    1.510,      # 12 Mg [1]
    2.249,      # 13 Al [5]
    2.147,      # 14 Si [1]
    2.074,      # 15 P  [1]
    1.900,      # 16 S  [1]
    1.974,      # 17 Cl [1]
    1.934,      # 18 Ar [1]
    1.906,      # 19 K  [1]
    1.700,      # 20 Ca [5]
    1.647,      # 21 Sc
    1.587,      # 22 Ti
    1.572,      # 23 V
    1.511,      # 24 Cr
    1.480,      # 25 Mn
    1.456,      # 26 Fe
    1.436,      # 27 Co
    1.417,      # 28 Ni [1]
    1.748,      # 29 Cu [1]
    1.381,      # 30 Zn [1]
    2.192,      # 31 Ga [1]
    2.140,      # 32 Ge [5]
    2.115,      # 33 As [1]
    2.103,      # 34 Se [1]
    2.095,      # 35 Br [1]
    2.071,      # 36 Kr [1]
    2.057,      # 37 Rb [5]
    1.821,      # 38 Sr [5]
    1.673,      # 39 Y
    1.562,      # 40 Zr
    1.583,      # 41 Nb
    1.526,      # 42 Mo
    1.499,      # 43 Tc
    1.481,      # 44 Ru
    1.464,      # 45 Rh
    1.450,      # 46 Pd [1]
    1.574,      # 47 Ag [1]
    1.424,      # 48 Cd [1]
    2.232,      # 49 In [1]
    2.196,      # 50 Sn [1]
    2.210,      # 51 Sb [5]
    2.235,      # 52 Te [1]
    2.250,      # 53 I  [1]
    2.202,      # 54 Xe [1]
    2.259,      # 55 Cs [5]
    1.851,      # 56 Ba [5]
    1.761,      # 57 La
    unknown,    # 58 Ce
    unknown,    # 59 Pr
    unknown,    # 60 Nd
    unknown,    # 61 Pm
    unknown,    # 62 Sm
    unknown,    # 63 Eu
    unknown,    # 64 Gd
    unknown,    # 65 Tb
    unknown,    # 66 Dy
    unknown,    # 67 Ho
    unknown,    # 68 Er
    unknown,    # 69 Tm
    unknown,    # 70 Yb
    unknown,    # 71 Lu
    1.570,      # 72 Hf
    1.585,      # 73 Ta
    1.534,      # 74 W
    1.477,      # 75 Re
    1.560,      # 76 Os
    1.420,      # 77 Ir
    1.377,      # 78 Pt [1]
    1.647,      # 79 Au [1]
    1.353,      # 80 Hg [1]
    2.174,      # 81 Tl [1]
    2.148,      # 82 Pb [1]
    2.185,      # 83 Bi [5]
    unknown,    # 84 Po [5]
    unknown,    # 85 At [5]
    unknown,    # 86 Rn [5]
    unknown,    # 87 Fr [5]
    unknown,    # 88 Ra [5]
    unknown,    # 89 Ac
    unknown,    # 90 Th
    unknown,    # 91 Pa
    unknown,    # 92 U [1]
    unknown,    # 93 Np
    unknown,    # 94 Pu
    unknown,    # 95 Am
    unknown,    # 96 Cm
    unknown,    # 97 Bk
    unknown,    # 98 Cf
    unknown,    # 99 Es
    unknown,    #100 Fm
    unknown,    #101 Md
    unknown,    #102 No
    unknown,    #103 Lr
))