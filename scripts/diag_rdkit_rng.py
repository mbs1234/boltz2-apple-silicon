"""Can RDKit's unseeded (randomSeed=-1) ETKDG embedding be reset to its fresh-process state?"""
import subprocess
import sys

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

SMI = "OCCN(CC1)CCN1C2=NC(C)=NC(NC3=NC=C(C(NC4=C(C)C=CC(OCC5=CC=CC=C5)=C4)=O)S3)=C2"


def embed():
    m = Chem.AddHs(Chem.MolFromSmiles(SMI))
    o = AllChem.ETKDGv3()
    o.clearConfs = False
    AllChem.EmbedMolecule(m, o)
    return m.GetConformer().GetPositions().round(6).tobytes()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        sys.stdout.write(embed().hex())
        sys.exit()
    fresh1 = bytes.fromhex(subprocess.run([sys.executable, __file__, "child"], capture_output=True,
                                          text=True).stdout)
    fresh2 = bytes.fromhex(subprocess.run([sys.executable, __file__, "child"], capture_output=True,
                                          text=True).stdout)
    print("fresh process A == fresh process B:", fresh1 == fresh2)
    a, b = embed(), embed()
    print("in-process 1st == fresh:", a == fresh1, "| 2nd == fresh:", b == fresh1)
    print("has rdBase.SeedRandomNumberGenerator:", hasattr(rdBase, "SeedRandomNumberGenerator"))
    if hasattr(rdBase, "SeedRandomNumberGenerator"):
        for s in (42, 0, 1, 5489):
            rdBase.SeedRandomNumberGenerator(s)
            print(f"  after SeedRandomNumberGenerator({s}): embed == fresh:", embed() == fresh1)
