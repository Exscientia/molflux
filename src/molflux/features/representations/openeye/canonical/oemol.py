from typing import Any, Dict, List, Optional, Union

try:
    from openeye.oechem import OEMolBase, OEWriteMolToBytes
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import to_oemol
from molflux.features.representations.openeye.canonical._utils import (
    ensure_oemol_title,
    standardise_oemol,
)
from molflux.features.typing import ArrayLike
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
A canonical OpenEye molecule representation.
"""


class CanonicalOemol(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: ArrayLike,
        strip_salts: bool = False,
        set_neutral_ph: bool = False,
        reasonable_protomer: bool = False,
        reasonable_tautomer: bool = False,
        explicit_h: bool = False,
        remove_formal_charges: bool = False,
        perceive_chiral: bool = True,
        assign_hyb: bool = True,
        rekekulise: bool = False,
        remove_stereo: bool = False,
        clear_non_chiral_stereo: bool = False,
        remove_non_standard_stereo: bool = False,
        sd_title_tag: str = "",
        clear_sd_data: bool = False,
        gen_2d_coords: bool = False,
        as_bytes: bool = True,
        tautomer_options: Optional[Dict[str, Any]] = None,
        tautomer_timeouts: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Dict[str, List[Union[bytes, OEMolBase]]]:
        r"""Featurises the input molecules as canonical OEMols.

        Args:
            samples: The sample molecules to featurise.
            strip_salts: Whether to remove everything but the largest component
                from the molecules, in case they contain disconnected
                components such as salts. Defaults to `False`.
            set_neutral_ph: Whether to set a neutral pH model to the molecules.
                Defaults to `False`.
            reasonable_protomer: Whether to attempt to produce a single protomer
                that will be a suitable representation of the molecule in a
                biological system. This state is defined as an aqueous
                environment with pH~7.4 and a tautomer from among the
                predominate tautomers that is favored by medicinal chemists.
                Defaults to `False`.
            reasonable_tautomer: Whether to generate reasonable tautomers and
                select the top-ranked. Defaults to `False`.
            explicit_h: Whether to convert the implicit hydrogens on the atoms
                of a molecule to explicit hydrogen atoms. Defaults to `False`.
            remove_formal_charges: Whether to remove formal charges from the
                input molecules. Defaults to `False`.
            perceive_chiral: Whether to perceive whether an atom or a baond in
                a molecule is chiral and update the `chiral` property
                accordingly. Defaults to `True`.
            assign_hyb: Whether to assign hybridization to all atoms in the
                given molecules. Defaults to `True`.
            rekekulise: Whether to reassign the single-double bond distribution
                in aromatic rings. Defaults to `False`.
            remove_stereo: Whether to remove stereochemistry information from
                the input molecule. Defaults to `False`.
            clear_non_chiral_stereo: Whether to remove false atom stereo
                information from non-chiral atoms. Defaults to `False`.
            remove_non_standard_stereo: OpenEye's definition of what is chiral
                includes tertiary nitrogens. This isn't really from a synthetic
                chemists standpoint - so this option removes chiral centres from
                these if set to `True`. Defaults to `False`.
            sd_title_tag: The SD tag to use as the molecules' title. Defaults to
                the empty string.
            clear_sd_data: Whether to clear all SD data from the molecules.
                Defaults to `False`.
            gen_2d_coords: Whether to generate 2D coordinates for the input
                molecules. Defaults to `False`.
            as_bytes: Whether to encode the output molecules as bytes. Defaults
                to `True`.
            tautomer_options: Dictionary of settings for the tautomerisation stage. Keys and value types allowed are:
                - max_generated: int
                - rank: bool
                - warts: bool
                - level: int
                - carbon_hybrid: bool
                - save_stereo: bool
                - max_zone_size: int
                - max_tautomeric_atoms: int
                - max_time: int
                - kekule: bool
                - clear_coords: bool
                - hybrid_level: int
                - max_to_return: int
                - racemic_type: int
                NOTE: This parameter has an effect only if `reasonable_tautomer` is set to True.
            tautomer_timeouts: List of timeouts to be passed to the tautomerisation stage.
                For each molecule, each timeout is tried at the tautomerisation step until the first reasonable
                tautomer is generated. Once one is found, the other timeouts are skipped.
                NOTE: This parameter has an effect only if `reasonable_tautomer` is set to True.

        Returns:
            Canonicalised copies of the input molecules. These are returned
            as bytes if `as_bytes` was set to `True`, and as OEMol objects
            otherwise.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('canonical_oemol')
            >>> samples = ['C']
            >>> representation.featurise(samples, as_bytes=True)
            {'canonical_oemol': [b'\x0b\xa5\n\x8f\x19\x85mol_0\x13\x83\x81C\x80 \x81\x00\x10\x8f\x81\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00!\x81\x00']}
        """
        canonical_mols = []
        for idx, sample in enumerate(samples):
            with featurisation_error_harness(sample):
                mol = to_oemol(sample)
                canonical_mol = standardise_oemol(
                    mol,
                    strip_salts=strip_salts,
                    set_neutral_ph=set_neutral_ph,
                    reasonable_protomer=reasonable_protomer,
                    reasonable_tautomer=reasonable_tautomer,
                    explicit_h=explicit_h,
                    remove_formal_charges=remove_formal_charges,
                    perceive_chiral=perceive_chiral,
                    assign_hyb=assign_hyb,
                    rekekulize=rekekulise,
                    remove_stereo=remove_stereo,
                    clear_non_chiral_stereo=clear_non_chiral_stereo,
                    remove_non_standard_stereo=remove_non_standard_stereo,
                    sd_title_tag=sd_title_tag,
                    clear_sd_data=clear_sd_data,
                    gen_2d_coords=gen_2d_coords,
                    tautomer_options=tautomer_options,
                    tautomer_timeouts=tautomer_timeouts,
                )
                canonical_mol = ensure_oemol_title(canonical_mol, title=f"mol_{idx}")
                if as_bytes:
                    canonical_mol = OEWriteMolToBytes(".oeb", canonical_mol)
                canonical_mols.append(canonical_mol)

        return {self.tag: canonical_mols}
