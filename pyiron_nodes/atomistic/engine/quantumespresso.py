from core import as_function_node


@as_function_node
def QuantumEspresso(kpts=(3, 3, 3), ecutwfc=20, smearing=0.02):
    pseudopotentials = {
        'Ca': 'Ca_pbe_v1.uspp.F.UPF',
        'Mg': 'Mg.pbe-n-kjpaw_psl.0.3.0.UPF',
    }

    from ase.calculators.espresso import Espresso, EspressoProfile
    profile = EspressoProfile(
        command='pw.x', pseudo_dir='/home/jovyan/dpg_tutorial_2026/espresso/pseudo'
    )
    calc = Espresso(
        pseudopotentials=pseudopotentials,
        kpts=kpts,
        input_data={
            "ecutwfc": ecutwfc,
            "calculation": "scf",
            "occupations": "smearing",
            "degauss": smearing,
        },
        profile=profile,
    )
    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=calc)
    return out
