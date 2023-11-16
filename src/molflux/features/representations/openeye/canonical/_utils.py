import logging
from functools import reduce
from typing import List, Optional

try:
    from openeye import oechem, oequacpac
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

logger = logging.getLogger(__name__)

_DEFAULT_TAUTOMER_TIMEOUTS = [0.01, 0.1, 1, 10, 100]

# setting log levels for oequacpac, as it can be very verbose
oequacpac.oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)


def ensure_oemol_title(mol: oechem.OEMolBase, title: str) -> oechem.OEMolBase:
    if mol.GetTitle() == "":
        mol.SetTitle(title)
    return mol


def standardise_oemol(
    mol: oechem.OEMolBase,
    *,
    strip_salts: bool = False,
    set_neutral_ph: bool = False,
    reasonable_protomer: bool = False,
    reasonable_tautomer: bool = False,
    explicit_h: bool = False,
    remove_formal_charges: bool = False,
    perceive_chiral: bool = True,
    assign_hyb: bool = True,
    rekekulize: bool = False,
    remove_stereo: bool = False,
    clear_non_chiral_stereo: bool = False,
    remove_non_standard_stereo: bool = False,
    sd_title_tag: str = "",
    clear_sd_data: bool = False,
    gen_2d_coords: bool = False,
    tautomer_options: Optional[oequacpac.OETautomerOptions] = None,
    tautomer_timeouts: Optional[List[float]] = None,
) -> oechem.OEMolBase:
    """
    Returns a standardised OEMol object.
    """

    if remove_formal_charges and (reasonable_protomer or set_neutral_ph):
        raise ValueError("cannot set neutral pH and remove formal charges.")

    # create a copy of the input molecule for standardisation
    std_mol = mol.CreateCopy()

    # use a given SD tag as molecule title
    if sd_title_tag:
        if isinstance(std_mol, oechem.OEMCMolBase):
            title = oechem.OEGetSDData(std_mol.GetActive(), sd_title_tag)
        else:
            title = oechem.OEGetSDData(std_mol, sd_title_tag)

        std_mol.SetTitle(title)

    if clear_sd_data:
        if isinstance(std_mol, oechem.OEGraphMol):
            oechem.OEClearSDData(std_mol)
        else:
            for conf in std_mol.GetConfs():
                oechem.OEClearSDData(conf)

    if strip_salts:
        oechem.OETheFunctionFormerlyKnownAsStripSalts(std_mol)

    if explicit_h and not oechem.OEHasExplicitHydrogens(std_mol):
        oechem.OEAddExplicitHydrogens(std_mol)

    if not reasonable_tautomer and (tautomer_options or tautomer_timeouts):
        logger.warning(
            "Tautomer options were given without setting reasonable_tautomer=True. No tautomerisation will"
            " be done, which might not be the intended user behaviour.",
        )

    if reasonable_tautomer:
        pka_norm = False

        if tautomer_options is None:
            tautomer_options = {}
        tautomer_options = generate_tautomer_options(**tautomer_options)

        if tautomer_timeouts is None:
            tautomer_timeouts = _DEFAULT_TAUTOMER_TIMEOUTS.copy()

        # flag for whether we found a reasonable tautomer
        successful_tautomerisation = False

        # go through each timeout and try to generate reasonable tautomers
        for max_timeout in tautomer_timeouts:
            tautomer_options.SetMaxSearchTime(max_timeout)

            reas_tauts = list(
                oequacpac.OEGetReasonableTautomers(std_mol, tautomer_options, pka_norm),
            )

            # if no reasonable tautomers were generated, skip to the next max_timeout
            if len(reas_tauts) == 0:
                continue

            # OEGetReasonableTautomers returns a list of 1 or more (depending on
            # tautomerOptions set); the first one is returned
            if len(reas_tauts) > 1:
                logger.debug(
                    "Multiple reasonable tautomers returned for molecule -> taking top ranked",
                )

            std_mol = reas_tauts[0]
            successful_tautomerisation = True
            break

        # if the loop above finished with no reasonable tautomer found at any stage, raise an error
        if not successful_tautomerisation:
            raise RuntimeError(
                f"Could not generate a reasonable tautomer in the given time for {oechem.OEMolToSmiles(std_mol)}",
            )

    if set_neutral_ph:
        oequacpac.OESetNeutralpHModel(std_mol)

    if reasonable_protomer:
        oequacpac.OEGetReasonableProtomer(std_mol)

    # remove formal charges / OERemoveFormalCharge returns False for molecules
    # with unknown atom types
    if remove_formal_charges:
        if not oequacpac.OERemoveFormalCharge(std_mol):
            logger.error("failed to remove formal charges from %s.", mol.GetTitle())

    if assign_hyb and not std_mol.HasPerceived(oechem.OEPerceived_Hybridization):
        oechem.OEAssignHybridization(std_mol)

    if remove_stereo:
        std_mol = remove_mol_stereo(std_mol)

    elif clear_non_chiral_stereo:
        clear_non_chiral_mol_stereo(std_mol)

    if remove_non_standard_stereo:
        remove_non_standard_mol_stereo(std_mol)

    if perceive_chiral and not std_mol.HasPerceived(oechem.OEPerceived_Chiral):
        oechem.OEPerceiveChiral(std_mol)

    if rekekulize:
        rekekulize_mol(std_mol)

    if gen_2d_coords:
        oechem.OEGenerate2DCoordinates(std_mol)

    return std_mol


def remove_mol_stereo(mol: oechem.OEMolBase) -> oechem.OEMolBase:
    """
    Returns a copy of the input molecule `mol` with all stereochemistry
    information removed.
    """
    cmol = mol.CreateCopy()
    cmol.Clear()

    oechem.OEUncolorMol(
        cmol,
        mol,
        oechem.OEUncolorStrategy_RemoveAtomStereo
        | oechem.OEUncolorStrategy_RemoveBondStereo
        | oechem.OEUncolorStrategy_RemoveGroupStereo,
    )

    return cmol


def clear_non_chiral_mol_stereo(mol: oechem.OEMolBase) -> None:
    """
    Clears all false stereo information from non-chiral atoms and bonds.
    """
    clear_non_chiral_atom_stereo(mol)
    clear_non_chiral_bond_stereo(mol)


def clear_non_chiral_atom_stereo(mol: oechem.OEMolBase) -> None:
    """
    Removes false atom stereo information from non-chiral atoms.
    """
    pred = oechem.OEAndAtom(
        oechem.OEHasAtomStereoSpecified(),
        oechem.OENotAtom(oechem.OEIsChiralAtom()),
    )

    for atom in mol.GetAtoms(pred):
        atom.SetStereo(
            list(atom.GetAtoms()),
            oechem.OEAtomStereo_Tetrahedral,
            oechem.OEAtomStereo_Undefined,
        )


def clear_non_chiral_bond_stereo(mol: oechem.OEMolBase) -> None:
    """
    Removes false bond stereo information from non-chiral bonds.
    """
    pred = reduce(
        oechem.OEAndBond,
        (
            oechem.OEHasBondStereoSpecified(),
            oechem.OENotBond(oechem.OEIsChiralBond()),
            oechem.OEHasOrder(2),
        ),
    )

    for bond in mol.GetBonds(pred):
        bond.SetStereo(
            [bond.GetBgn(), bond.GetEnd()],
            oechem.OEBondStereo_CisTrans,
            oechem.OEBondStereo_Undefined,
        )


def remove_non_standard_mol_stereo(mol: oechem.OEMolBase) -> None:
    """
    OpenEye's definition of what is chiral includes tertiary nitrogens.
    This isn't really from a synthetic chemists standpoint - so this
    function removes chiral centres from these.
    """
    pred = oechem.OEAndAtom(
        oechem.OEHasAtomStereoSpecified(),
        oechem.OEAndAtom(oechem.OEHasAtomicNum(7), oechem.OEHasHvyDegree(3)),
    )

    for atom in mol.GetAtoms(pred):
        atom.SetStereo(
            list(atom.GetAtoms()),
            oechem.OEAtomStereo_Tetrahedral,
            oechem.OEAtomStereo_Undefined,
        )


def rekekulize_mol(
    mol: oechem.OEMolBase,
    aro_model: int = oechem.OEAroModel_OpenEye,
) -> None:
    """
    Reassigns the single-double bond distribution in aromatic rings. In rare instances,
    identical molecules can have a different distribution, which can cause problems with
    algorithms that hash fragments.
    """
    oechem.OEFindRingAtomsAndBonds(mol)
    oechem.OEAssignAromaticFlags(mol, aro_model)

    for bond in mol.GetBonds():
        if bond.IsAromatic():
            bond.SetIntType(5)
        else:
            bond.SetIntType(bond.GetOrder())

    oechem.OECanonicalOrderAtoms(mol)
    oechem.OECanonicalOrderBonds(mol)
    oechem.OEClearAromaticFlags(mol)
    oechem.OEKekulize(mol)

    oechem.OEAssignAromaticFlags(mol, aro_model)


def generate_tautomer_options(
    max_generated: int = 4096,
    rank: bool = True,
    warts: bool = False,
    level: int = 0,
    carbon_hybrid: bool = True,
    save_stereo: bool = False,
    max_zone_size: int = 35,
    max_tautomeric_atoms: int = 70,
    max_time: int = 120,
    kekule: bool = False,
    clear_coords: bool = False,
    hybrid_level: int = oequacpac.OEHybridizationLevel_EnolEnamine,
    max_to_return: int = 256,
    racemic_type: int = oequacpac.OERacemicType_LocalSampled,
) -> oequacpac.OETautomerOptions:
    """
    Generates an oequacpac.OETautomerOptions object from the given settings.

    Args:
        max_generated: int = 4096
            Maximum number of tautomers that may be generated for each input molecule.
        rank: bool = True
            Whether or not tautomers are ranked and ordered before being returned.
        warts: bool = False
            Whether to set warts application of {title}_1, {title}_2, etc., on tautomers.
        level: int = 0
            Level for tautomerization. Higher levels allow less likely atomic transitions to occur.
        carbon_hybrid: bool = True
            Setting for whether or not carbon hybridization changes are allowed. Allowing sp2-sp3 changes
            of carbon atoms can significantly increase computation time.
        save_stereo: bool = False
            Setting for whether atoms and bonds with labeled stereochemistry are fixed or allowed to
            participate in tautomerization.
        max_zone_size: int = 35
            Maximum number of atoms allowed in a continuous tautomerization zone.
        max_tautomeric_atoms: int = 70
            Maximum number of tautomeric atoms allowed in a molecule.
        max_time: 120
            Maximum search time in seconds allowed for a molecule.
        kekule: bool = False
            Setting for whether or not generated tautomers are kekule or have aromaticity perceived.
        clear_coords: bool = False
            Whether input coordinates are cleared.
        hybrid_level: int = oequacpac.OEHybridizationLevel_EnolEnamine
            Setting for the level of hybridization changes allowed when carbon hybridization is allowed.
            Allowing carbon hybridization significantly increase computation time.
        max_to_return: int = 256
            Maximum number of tautomers that may be returned for each input molecule.
        racemic_type: int = oequacpac.OERacemicType_LocalSampled
            Setting for the way to handle loss of stereocenters, if the OETautomerOptions.GetSaveStereo method
            returns false. If the OETautomerOptions.GetSaveStereo method returns true, this option is irrelevant.

    Returns:
        oequacpac.OETautomerOptions
    """
    options = oequacpac.OETautomerOptions(
        max_generated,
        rank,
        warts,
        level,
        carbon_hybrid,
        save_stereo,
        max_zone_size,
        max_tautomeric_atoms,
        max_time,
        kekule,
        clear_coords,
        hybrid_level,
        max_to_return,
        racemic_type,
    )

    return options
