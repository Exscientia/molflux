import json
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from more_itertools import always_iterable

from molflux.features.bases import RepresentationBase, RepresentationInfo
from molflux.features.representations.openeye._utils import (
    get_sd_data,
    iterate_mols_or_confs,
    to_oemol,
    to_smiles,
)
from molflux.features.typing import ArrayLike
from molflux.features.utils import featurisation_error_harness


class DType(Enum):
    """DType callable Enum. Used to stringify data types that featurisations
    can be cast to when parsing SD tags."""

    FLOAT: Type = float
    INT: Type = int
    STR: Type = str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call method to call the underlying callable's method."""
        return self.value(*args, **kwargs)


class SDFeature(RepresentationBase):
    """Representation used to retrieve precalculated featurisations from SD tags
    on molecules. Featurisations are retrieved as if they are JSON strings unless
    otherwise specified."""

    def _info(self) -> RepresentationInfo:
        """Representation Info for SDFeature type Representation."""
        return RepresentationInfo(
            description=(
                "Prism Representation that retrieves a featurisation from "
                "an SD tag(s) by name."
            ),
        )

    def _featurise(
        self,
        samples: ArrayLike,
        *,
        sd_tags: Optional[Union[str, List[str]]] = None,
        as_json: Union[bool, List[str], Dict[str, bool]] = True,
        dtype: Union[DType, str, Dict[str, Union[str, DType]]] = "str",
        separator: Union[str, Dict[str, str]] = "\n",
        **kwargs: Any,
    ) -> Dict[str, List[Any]]:
        """Featurise method for SDFeature Representation. Defines the SD tags to
        check for precalculated representations and defines how to parse each SD
        tag.

        Args:
            samples: Samples that must be OEMolBase parsable.
            sd_tags: The SD tags to retrieve the featurisations of.
            as_json: Whether to parse all SD tags as JSON tags (`True`), or names of
                SD tags in `sd_tags` to parse as JSON tags, or a mapping of
                which tags to parse as JSON values and which to not. Tags in `sd_tags`
                that are not parsed as JSON are expected to be `separator`
                separated strings and are parsed as such.
            dtype: The data-type to parse each split molflux of an SD tag representation.
                Either a single type to apply to all, or a mapping between SD tags
                and the dtype to parse them as. Only applicable to SD tags that are
                *not* parsed `as_json`. Default is `"str"`.
            separator: The separator to split each feature molflux of an SD tag
                representation. Either a single split to apply to all `sd_tags`,
                or a mapping between SD tags and the `separator` to split
                molfluxs with. Only applicable to SD tags that are *not* parsed
                `as_json`. Default is `"\\n"`.

        Returns:
            A dictionary with the `sd_tags` as keys and a list of length
            `len(samples)` with parsed featurisations as values.

        Raises:
            ValueError: If the required `sd_tags` are not present on a molecule.

        Notes:
            If passing multi conformer molecules, its assumed that the SD tags
            will be present on the conformer, however if not present it this
            representation will fall back onto the parent molecules SD tags.

        Examples:
            >>> from molflux.features import load_representation
            >>> from openeye import oechem
            >>> import json

            >>> representation = load_representation("sd_feature")

            >>> mol = oechem.OEGraphMol()
            >>> assert oechem.OESmilesToMol(mol, "CCC")
            >>> assert oechem.OESetSDData(mol, "featurisation", json.dumps([0, 1, 1, 1, 0]))
            >>> assert oechem.OESetSDData(mol, "separated_features", "feature1\\nfeature2\\nfeature3")
            >>> # Ask for these specific tags.
            >>> representation.update_state(sd_tags=["featurisation", "separated_features"], as_json=["featurisation"])
            >>> representation.featurise([mol])
            {'sd_feature::featurisation': [[0, 1, 1, 1, 0]], 'sd_feature::separated_features': [['feature1', 'feature2', 'feature3']]}
        """
        if not sd_tags:
            raise ValueError(
                "Must supply SD tags to retrieve featurisations for. Supplied:"
                f" {sd_tags}",
            )
        sd_tags_list: Sequence[str] = list(always_iterable(sd_tags))

        # If a sequence is supplied, default should be False, otherwise True.
        # This is because if someone supplies explicit tags to parse as JSON,
        # it would be expected that the others are *not* parsed as JSON
        default_as_json = not isinstance(as_json, Sequence)

        as_json_mapping = {}
        sep_mapping = {}
        dtype_mapping = {}

        # Populate the above mappings - this happens here to avoid having to
        # determine each for each sd tag *within the inner loop over molecules*
        # (i.e. determining the parsing for each SD tag for each mol).
        for tag in sd_tags_list:
            # Explicitly supplied mapping of which tags to parse as JSON (and not)
            if isinstance(as_json, dict):
                # Don't populate with tags not supplied.
                if tag in as_json:
                    as_json_mapping[tag] = as_json[tag]
            elif isinstance(as_json, bool):
                # Parse all tags or not
                as_json_mapping[tag] = as_json
            elif isinstance(as_json, Sequence):
                # List of tags to parse as JSON
                as_json_mapping[tag] = tag in as_json

            # Only populate these if they are *not* to be parsed as JSON
            if not as_json_mapping.get(tag, default_as_json):
                if isinstance(separator, dict):
                    # Don't populate with tags not supplied.
                    if tag in separator:
                        sep_mapping[tag] = separator[tag]
                elif isinstance(separator, str):
                    # Parse all tags or not
                    sep_mapping[tag] = separator

                if isinstance(dtype, dict):
                    # Don't populate with tags not supplied.
                    if tag in dtype:
                        dt = dtype[tag]
                        dtype_mapping[tag] = (
                            DType[dt.upper()] if not isinstance(dt, DType) else dt
                        )
                elif isinstance(dtype, (str, DType)):
                    # Parse all tags or not
                    dtype_mapping[tag] = (
                        DType[dtype.upper()] if not isinstance(dtype, DType) else dtype
                    )

        # To hold the feature representations.
        features = defaultdict(list)

        # Turn to oemolecules
        mols = map(to_oemol, samples)
        for mol in iterate_mols_or_confs(mols):
            with featurisation_error_harness(to_smiles(mol)):
                for tag in sd_tags_list:
                    parse_json = as_json_mapping.get(tag, default_as_json)
                    # This will raise if the tag is not present.
                    value = get_sd_data(mol, tag=tag, dtype=str)

                    if parse_json:
                        value = json.loads(value)
                    else:
                        # Split the value by the separator and cast as datatype.
                        dtype_type = dtype_mapping.get(tag, DType["STR"])
                        tag_sep = sep_mapping.get(tag, "\n")
                        value = [*map(dtype_type, value.split(tag_sep))]
                    features[f"{self.tag}::{tag}"].append(value)

        return dict(features)
