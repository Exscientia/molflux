import pytest

from molflux.splits.catalogue import list_splitting_strategies
from molflux.splits.load import load_splitting_strategy
from molflux.splits.strategy import SplittingStrategy

strategy_name = "scaffold_rdkit"


@pytest.fixture(scope="module")
def fixture_test_strategy():
    return load_splitting_strategy(strategy_name)


@pytest.fixture(scope="module")
def fixture_sample_dataset():
    return [
        "c1ccc(cc1)C(C#N)OC2C(C(C(C(O2)COC3C(C(C(C(O3)CO)O)O)O)O)O)O",
        "Cc1c(cco1)C(=O)Nc2ccccc2",
        "CC(=CCCC(=CC=O)C)C",
        "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4cccc5",
        "c1ccsc1",
        "c1ccc2c(c1)ncs2",
        "c1cc(c(c(c1)Cl)c2c(cc(cc2Cl)Cl)Cl)Cl",
        "CC12CCC3c4ccc(cc4CCC3C1CCC2O)O",
        "C1C2C3C(C1C4C2O4)C5(C(=C(C3(C5(Cl)Cl)Cl)Cl)Cl)Cl",
        "CC(=C)C1Cc2c(ccc3c2OC4COc5cc(c(cc5C4C3=O)OC)OC)O1",
        "C1CC(=O)NC1",
        "c1ccc2cc(ccc2c1)Cl",
        "CCCC=C",
        "CCC1(C(=O)NCNC1=O)c2ccccc2",
        "CCCCCCCCCCCCCC",
        "CC(C)Cl",
        "CCC(C)CO",
        "c1ccc(cc1)C#N",
        "CCOP(=S)(OCC)Oc1cc(nc(n1)C(C)C)C",
        "CCCCCCCCCC(C)O",
        "c1cc(c(cc1Cl)Cl)c2c(ccc(c2Cl)Cl)Cl",
        "C1CCC(CC1)n2c(=O)c3c([nH]c2=O)CCC3",
        "CCOP(=S)(OCC)SCSCC",
        "CCOc1ccc(cc1)NC(=O)C",
        "CCN(CC)c1c(cc(c(c1N(=O)=O)N)C(F)(F)F)N(=O)=O",
        "CCCCCCCO",
        "Cn1c2c(c(=O)n(c1=O)C)[nH]cn2",
        "CCCCC1(C(=O)NC(=O)NC1=O)CC",
        "c1cc(ccc1C(=C(Cl)Cl)c2ccc(cc2)Cl)Cl",
        "CCCCCCCC(=O)OC",
        "CCc1ccc(cc1)CC",
        "CCOP(=S)(OCC)SCSC(C)(C)C",
        "Cc1cccc(c1)NC(=O)Oc2cccc(c2)NC(=O)OC",
        "C=C(Cl)Cl",
        "Cc1cccc-2c1Cc3c2cccc3",
        "CCCCC=O",
        "c1ccc(cc1)Nc2ccccc2",
        "CN(C)C(=O)SCCCCOc1ccccc1",
        "CCCOP(=S)(OCCC)SCC(=O)N1CCCCC1C",
        "CCCCCCCI",
        "c1ccc(cc1)c2cccc(c2)Cl",
        "C=CCCCO",
        "C1CC2(C1)C(=O)NC(=O)NC2=O",
        "CC1CCC(C(C1)O)C(C)C",
        "CC(C)OC=O",
        "CCCCCC(C)O",
        "CC(=O)Nc1ccc(cc1)Br",
        "c1ccc(cc1)n2c(=O)c(c(cn2)N)Br",
        "CC1=C(C(C(=C(N1)C)C(=O)OC)c2ccccc2N(=O)=O)C(=O)OC",
        "Cc1ccc2ccc(nc2c1)C",
        "CCCCCCC#C",
        "CCC1(C(=O)NC(=O)NC1=O)C2=CCCCC2",
        "c1ccc2c(c1)ccc3c2ccc4c3cccc4",
        "CCC(C)n1c(=O)c(c([nH]c1=O)C)Br",
        "c1cc(c(c(c1)Cl)Cl)c2c(c(cc(c2Cl)Cl)Cl)Cl",
        "Cc1ccccc1O",
        "CC(C)CCC(C)(C)C",
        "Cc1ccc(c2c1cccc2)C",
        "Cc1cc2c3ccccc3ccc2c4c1cccc4",
        "CCCC(=O)C",
        "c1c(c(c(c(c1Cl)Cl)c2c(c(cc(c2Cl)Cl)Cl)Cl)Cl)Cl",
        "CCCOC(=O)CC",
        "CC12CC(C3(C(C1CC(C2(C(=O)CO)O)O)CCC4=CC(=O)C=CC43C)F)O",
        "c1cc(ccc1N)O",
        "c1ccc(cc1)CNC(=O)Cn2ccnc2N(=O)=O",
        "c1ccc2c(c1)C(=O)C(=C(C2=O)O)C3CCC(CC3)c4ccc(cc4)Cl",
        "CCNc1nc(nc(n1)Cl)N(CC)CC",
        "c1cnc(cn1)C(=O)N",
        "CCC(CC)(C(=O)NC(=O)N)Br",
        "c1ccc(c(c1)c2ccccc2Cl)Cl",
        "c1cc(oc1C=NN2CC(=O)NC2=O)N(=O)=O",
        "c1cc(ccc1N(=O)=O)Oc2ccc(cc2Cl)Cl",
        "CC1(C2CCC1(C(=O)C2)C)C",
        "C=CCC1(C(=O)NC(=O)NC1=O)c2ccccc2",
        "CCCCC(=O)OCC",
        "CC(C)CCOC(=O)C",
        "CCCCCC(=O)OCN1C(=O)C(NC1=O)(c2ccccc2)c3ccccc3",
        "c1cc(cc(c1)Cl)c2cc(ccc2Cl)Cl",
        "CCCBr",
        "CCCC1COC(O1)(Cn2cncn2)c3ccc(cc3Cl)Cl",
        "CN(C=O)C(=O)CSP(=S)(OC)OC",
        "Cc1c2c(nccn2)ncn1",
        "C(=S)(N)N",
        "Cc1ccc(cc1)C",
        "CCc1ccccc1CC",
        "C(C(Cl)(Cl)Cl)(Cl)(Cl)Cl",
        "CC(C)C(c1ccc(cc1)OC(F)F)C(=O)OC(C#N)c2cccc(c2)Oc3ccccc3",
        "CCCN(=O)=O",
        "CC1CCC(C(=O)C1)C(C)C",
        "CCN1c2cc(ccc2NC(=O)c3c1nccc3)Cl",
        "c1cc(c(c(c1)Cl)Cl)N(=O)=O",
        "CCCC(C)C1(C(=O)NC(=S)NC1=O)CC=C",
        "c1ccc-2c(c1)-c3cccc4c3c2ccc4",
        "CCCOC(C)C",
        "Cc1cc(c2ccccc2c1)C",
        "CCC(=C(CC)c1ccc(cc1)O)c2ccc(cc2)O",
        "C(#N)c1c(c(c(c(c1Cl)Cl)Cl)C#N)Cl",
        "c1cc(c(cc1c2cc(ccc2Cl)Cl)Cl)Cl",
        "c1ccc(cc1)C2CO2",
        "CC(C)c1ccccc1",
        "CC12CCC3C(C1CCC2C(=O)CO)CCC4=CC(=O)CCC34C",
        "c1cnc2c(n1)c(c(c(c2Cl)Cl)Cl)Cl",
        "C1C(C(C(C(O1)O)O)O)O",
        "C(Cl)Cl",
        "CCc1cccc2c1cccc2",
        "COC=O",
        "c1ccc(c(c1)N(=O)=O)O",
        "Cc1c[nH]c(=O)[nH]c1=O",
        "CC(C)C",
        "c1nc2c(c(n1)O)ncn2C3C(C(C(O3)CO)O)O",
        "c1c(cc(c(c1I)O)I)C#N",
        "c1cc(c(cc1N(=O)=O)Cl)NC(=O)c2cc(ccc2O)Cl",
        "CCCCC",
        "c1ccc(cc1)O",
        "c1ccc2cc3cc(ccc3cc2c1)N",
        "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
        "c1ccc2cnccc2c1",
        "CC(C)N(c1ccc(cc1)Cl)C(=O)CSP(=S)(OC)OC",
        "CCCCCCc1ccccc1",
        "c1ccc(cc1)c2ccccc2Cl",
    ]


def test_is_in_catalogue():
    """That the strategy is registered in the catalogue."""
    catalogue = list_splitting_strategies()
    all_strategy_names = [name for names in catalogue.values() for name in names]
    assert strategy_name in all_strategy_names


def test_implements_protocol(fixture_test_strategy):
    """That the strategy implements the protocol."""
    strategy = fixture_test_strategy
    assert isinstance(strategy, SplittingStrategy)


def test_yields_one_fold_by_default(fixture_sample_dataset, fixture_test_strategy):
    """That the splitting strategy only yields one set of splits by default."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=range(len(dataset)), y=dataset)
    assert len(list(indices)) == 1


def test_default_split_fractions(fixture_sample_dataset, fixture_test_strategy):
    """That the dataset is split by default into 80:10:10 splits."""
    dataset = fixture_sample_dataset
    n_samples = len(dataset)
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=range(len(dataset)), y=dataset)
    train_indices, validation_indices, test_indices = next(indices)
    assert len(list(train_indices)) == 0.8 * n_samples
    assert len(list(validation_indices)) == 0.1 * n_samples
    assert len(list(test_indices)) == 0.1 * n_samples


def test_splits_are_disjoint(fixture_sample_dataset, fixture_test_strategy):
    """That data is spread across splits without overlap."""
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=range(len(dataset)), y=dataset)
    train_indices, validation_indices, test_indices = next(indices)
    assert set(train_indices).isdisjoint(set(validation_indices))
    assert set(train_indices).isdisjoint(set(test_indices))
    assert set(validation_indices).isdisjoint(set(test_indices))


def test_inconsistent_split_fractions_raise(
    fixture_sample_dataset,
    fixture_test_strategy,
):
    dataset = fixture_sample_dataset
    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=range(len(dataset)),
        y=dataset,
        train_fraction=0.6,
        validation_fraction=0.1,
        test_fraction=0.1,
    )
    with pytest.raises(AssertionError):
        next(indices)
