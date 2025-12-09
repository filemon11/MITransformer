from typing import TypedDict, Literal


class ParseResult(TypedDict):
    to_predict: str
    covariates: set[str]
    groups: set[str]
    random_effects: dict[str, set[str | Literal[0] | Literal[1]]]
    formula: str


def parse(formula: str) -> ParseResult:
    to_predict, formula = formula.split("~")

    # dependent
    to_predict = to_predict.strip()

    # covariates
    components = [c.strip() for c in split_nested(formula, "+")]
    covariates: set[str] = set()
    randoms: list[str] = []
    for c in components:
        if c[0] == "(" and c[-1] == ")":
            randoms.append(c[1:-1].strip())
        else:
            covariates.add(c)

    # random effects
    groups: set[str] = set()
    random_effects: dict[
        str, set[str | Literal[0] | Literal[1]]] = {}
    for re in randoms:
        covs, group = re.split("|")
        group = group.strip()
        groups.add(group)

        covs_split: list[str | Literal[0] | Literal[1]] = [
            cov.strip() for cov in covs.split("+")]
        covs_split = [
            (int(cov) if cov in ("0", "1") else cov)  # type: ignore
            for cov in covs_split]
        assert group not in random_effects.keys(), (
            "Independent random effects are not supported.")
        random_effects[group] = set(covs_split)

    for group, s in random_effects.items():
        assert not (0 in s and 1 in s), (
            f"Provided bot 1 and 0 for group {group}. Specify one option.")
        if not (0 in s or 1 in s):
            s.add(1)
            # Add explicit intercept for clarity
        for i in s:
            if i not in (1, 0):
                assert i in covariates, (
                    f"Cannot add random slope for {i} because it does not"
                    " appear as a covariate.")

    # Include formula in standard form
    formula_components = [
        f"({' + '.join([str(i) for i in s])}|{group})"
        for group, s in random_effects.items()]
    formula = (
        f"{to_predict} ~ {' + '.join(covariates)} "
        f"{' + '.join(formula_components)}")

    return {
        "groups": groups,
        "covariates": covariates,
        "random_effects": dict(random_effects),
        "to_predict": to_predict,
        "formula": formula
    }


def split_nested(
        string: str, split_at: str = ",",
        l_brackets: str = "({[",
        r_brackets: str = ")}]"
        ) -> tuple[str, ...]:
    """ATTENTION: do not cross brackets"""
    assert len(split_at) == 1

    components: list[str] = []
    level = 0
    c_incomplete = ""
    for c in string:
        if c == split_at and level == 0:
            components.append(c_incomplete)
            c_incomplete = ""
            continue
        elif c in l_brackets:
            level += 1
        elif c in r_brackets:
            level -= 1
        c_incomplete += c
    if len(c_incomplete) > 0:
        components.append(c_incomplete)

    return tuple(components)
