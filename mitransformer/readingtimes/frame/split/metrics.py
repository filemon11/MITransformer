import pandas as pd
import nltk  # type: ignore
import numpy as np
import numpy.typing as npt
import torch

from ...lingutils import (
    UntokSplitFunc, UntokSplitAdd,
    UntokSplitHead, UntokSplitFirst,
    pos_merge, TAGSET, CONTENT_POS)
from ....data import CoNLLUDataset, parse_list_of_words_with_spacy, TokenMapper
from ....data.dataset import (
    load_conllu_from_str, get_tokens, get_head_list, get_space_after,
    get_deprels, TokenList, Sequence, TransformMaskHeadChild,
    head_list_to_adjacency_matrix, shift_masks)
from ....train import LMTrainer
from ....train.trainer import inverse_sigmoid

from abc import ABC, abstractmethod

from typing import Type, Any, Iterable, Callable, Literal, Collection

LANG = "en"


# # split tokenised


class SplitTokMetricMaker(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(
            self, df: pd.DataFrame,
            *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        ...


class SplitTokWordMetricMaker(SplitTokMetricMaker):
    def __init__(self, word_col: str, *args, **kwargs):
        self.word_col = word_col


class SplitTokMetricMakerPOS(SplitTokWordMetricMaker):
    def __call__(
            self, df: pd.DataFrame,
            *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        return df.apply(
            lambda r: get_POS_tags(
                r[self.word_col]), axis=1), dict()  # type: ignore


class SplitTokMetricMakerPosition(SplitTokWordMetricMaker):
    def __call__(
            self, df: pd.DataFrame,
            *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        return df.apply(
            lambda r: list(range(len(r[self.word_col]))), axis=1), dict()  # type: ignore


class SplitTokConlluDatasetMetricMaker(SplitTokMetricMaker):
    def __init__(self, conllu_col: str | None = None, *args, **kwargs):
        self.conllu_col = conllu_col

    @abstractmethod
    def __call__(
            self, df: pd.DataFrame,
            dataset: CoNLLUDataset | None, *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        ...

    def _call_method(
            self, df: pd.DataFrame,
            dataset: CoNLLUDataset | None,
            name: str, method: Callable[
                [TokenList, bool, bool], Sequence | np.ndarray],
            trim: bool
            ) -> tuple[pd.Series, dict[str, Any]]:
        if dataset is not None:
            attr = getattr(dataset, name)
            if trim:
                attr = [sen[2:-1] for sen in attr]
            return pd.Series(attr), {"dataset": dataset}
        else:
            assert self.conllu_col is not None
            series = pd.Series(
                [method(
                    li, not trim,
                    not trim) for li in df[self.conllu_col]])
            return series, {}


class SplitTokMetricMakerWord(SplitTokConlluDatasetMetricMaker):
    def __call__(
            self, df: pd.DataFrame,
            dataset: CoNLLUDataset | None = None, *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        return self._call_method(
            df, dataset, "tokens", get_tokens, trim=True)


class SplitTokMetricMakerSpaceAfter(SplitTokConlluDatasetMetricMaker):
    def __call__(
            self, df: pd.DataFrame,
            dataset: CoNLLUDataset | None = None, *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        return self._call_method(
            df, dataset, "space_after",
            lambda li, x, y: get_space_after(li), trim=False)


class SplitTokMetricMakerHeadlist(SplitTokConlluDatasetMetricMaker):
    def __call__(
            self, df: pd.DataFrame,
            dataset: CoNLLUDataset | None = None, *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        return self._call_method(
            df, dataset, "headlist", get_head_list, trim=False)


class SplitTokMetricMakerDeprels(SplitTokConlluDatasetMetricMaker):
    def __call__(
            self, df: pd.DataFrame,
            dataset: CoNLLUDataset | None = None, *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        return self._call_method(
            df, dataset, "deprels", get_deprels, trim=True)


class SplitTokMetricMakerTokenlist(SplitTokMetricMaker):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(
            self, df: pd.DataFrame, words: Iterable[str],
            max_len: int | None = None,
            min_len: int | None = None, *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        conllu = parse_list_of_words_with_spacy(words, min_len=None)
        tokenlists = load_conllu_from_str(conllu, max_len)
        return pd.Series(tokenlists), {}


class SplitTokMetricMakerSurprisal(SplitTokMetricMaker):
    def __init__(self, conllu_col: str | None = None, *args, **kwargs):
        self.conllu_col = conllu_col

    def __call__(
            self, df: pd.DataFrame,
            token_mapper_dir: str,
            transform: TransformMaskHeadChild,
            trainer: LMTrainer,
            dataset: CoNLLUDataset | None = None,
            masks_setting: Literal[
            "current", "next"] = "current", *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        if dataset is None:
            assert self.conllu_col is not None

            tokenlists: Sequence[TokenList] = df[self.conllu_col].tolist()
            dataset = CoNLLUDataset.from_conllu(
                tokenlists, transform, masks_setting=masks_setting)

            token_mapper: TokenMapper = TokenMapper.load(token_mapper_dir)
            dataset.map_to_ids(token_mapper)

        assert dataset is not None
        pred_probs, attention_logits = trainer.predict(
            dataset,
            make_prob=True,
            only_true=True)
        probs = [(-np.log(p[1:-1])).tolist() for p in pred_probs]
        assert all(len(p) == len(t) for p, t in zip(probs, df["word"])), (
            ([(len(p), len(t)) for p, t in zip(probs, df["word"])]))
        return pd.Series(probs), {
            "attention_logits": attention_logits,
            "dataset": dataset,
            "transform": transform,
            "token_mapper_dir": token_mapper_dir,
            "trainer": trainer,
            "masks_setting": masks_setting}


class SplitTokMetricMakerMask(SplitTokMetricMaker):
    def __init__(self, head_col: str, *args, **kwargs):
        self.head_col = head_col

    def __call__(
            self, df: pd.DataFrame, transform: TransformMaskHeadChild,
            masks_setting: Literal[
                "current", "next"] = "current",
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        masks: list[dict[str, np.ndarray]] = []
        for heads in df[self.head_col]:
            masks.append(
                transform(
                    head_list_to_adjacency_matrix(heads)))
        masks = [{
            key: value for key, value
            in shift_masks(masks_setting, m).items()
            if value is not None} for m in masks]
        return pd.Series(masks), {
            "transform": transform,
            "masks_setting": masks_setting
        }


class SplitTokMetricMakerHeadDistance(SplitTokMetricMaker):
    def __init__(self, mask_col: str, pos_col: str, *args, **kwargs):
        self.mask_col = mask_col
        self.pos_col = pos_col

    def __call__(
            self, df: pd.DataFrame,
            gov_name: str = "head_current",
            dep_name: str = "child_current",
            only_content_words_cost: bool = False,
            content_pos: Collection[str] = CONTENT_POS[TAGSET],
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        return pd.Series([
            generate_head_distance(
                pos, mask, only_content_words_cost=only_content_words_cost,
                gov_name=gov_name, dep_name=dep_name)
            for pos, mask in zip(df[self.pos_col], df[self.mask_col])]), {
            "content_pos": content_pos,
            "only_content_words_cost": only_content_words_cost
        }


class SplitTokMetricMakerFirstDependentDistance(SplitTokMetricMaker):
    def __init__(self, mask_col: str, pos_col: str, *args, **kwargs):
        self.mask_col = mask_col
        self.pos_col = pos_col

    def __call__(
            self, df: pd.DataFrame,
            gov_name: str = "head_current",
            dep_name: str = "child_current",
            only_content_words_left: bool = False,
            only_content_words_cost: bool = False,
            content_pos: Collection[str] = CONTENT_POS[TAGSET],
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        return pd.Series([
            generate_first_dep_distance(
                pos, mask,
                only_content_words_left=only_content_words_left,
                only_content_words_cost=only_content_words_cost,
                gov_name=gov_name,
                dep_name=dep_name)
            for pos, mask in zip(df[self.pos_col], df[self.mask_col])]), {
            "content_pos": content_pos,
            "only_content_words_left": only_content_words_left,
            "only_content_words_cost": only_content_words_cost
        }


class SplitTokMetricMakerFirstDependentCorrect(SplitTokMetricMaker):
    def __init__(
            self, mask_col: str, pos_col: str,
            first_dependent_distance_col: str, *args, **kwargs):
        self.mask_col = mask_col
        self.pos_col = pos_col
        self.first_dependent_distance_col = first_dependent_distance_col

    def __call__(
            self, df: pd.DataFrame,
            attention_logits: dict[str, np.ndarray],
            gov_name: str = "head_current",
            dep_name: str = "child_current",
            only_content_words_left: bool = False,
            content_pos: Collection[str] = CONTENT_POS[TAGSET],
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        attention_matrices = [
            {name: masks[i] for name, masks in attention_logits.items()}
            for i in range(len(list(attention_logits.values())[0]))]
        return pd.Series([
            generate_first_dependent_correct(
                att_mat, fdd, pos,
                content_pos=content_pos,
                gov_name=gov_name,
                dep_name=dep_name,
                only_content_words_left=only_content_words_left)
            for att_mat, fdd, pos in zip(
                attention_matrices, df[self.first_dependent_distance_col],
                df[self.pos_col])]), {
            "content_pos": content_pos,
            "only_content_words_left": only_content_words_left
        }


class SplitTokMetricMakerFirstDependentDistanceWeight(SplitTokMetricMaker):
    def __init__(
            self, mask_col: str, *args, **kwargs):
        self.mask_col = mask_col

    def __call__(
            self, df: pd.DataFrame,
            attention_logits: dict[str, np.ndarray],
            gov_name: str = "head_current",
            dep_name: str = "child_current",
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        attention_matrices = [
            {name: masks[i] for name, masks in attention_logits.items()}
            for i in range(len(list(attention_logits.values())[0]))]
        return pd.Series([
            generate_first_dependent_distance_weight(
                mask, att_mat,
                gov_name=gov_name,
                dep_name=dep_name)
            for mask, att_mat in zip(
                df[self.mask_col],
                attention_matrices)]), {}


class SplitTokMetricMakerFirstDependentDeprel(SplitTokMetricMaker):
    def __init__(
            self, mask_col: str, pos_col: str, deprel_col: str,
            *args, **kwargs):
        self.mask_col = mask_col
        self.pos_col = pos_col
        self.deprel_col = deprel_col

    def __call__(
            self, df: pd.DataFrame,
            only_content_words_left: bool = False,
            gov_name: str = "head_current",
            dep_name: str = "child_current",
            content_pos: Collection[str] = CONTENT_POS[TAGSET],
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        return pd.Series([
            generate_first_dependent_deprel(
                mask, pos, deprel, content_pos=content_pos,
                gov_name=gov_name,
                dep_name=dep_name,
                only_content_words_left=only_content_words_left)
            for mask, pos, deprel in zip(
                df[self.mask_col],
                df[self.pos_col],
                df[self.deprel_col])]), {
            "content_pos": content_pos,
            "only_content_words_left": only_content_words_left}


class SplitTokMetricMakerLeftDependentsDistanceSum(SplitTokMetricMaker):
    def __init__(
            self, mask_col: str, pos_col: str, *args, **kwargs):
        self.mask_col = mask_col
        self.pos_col = pos_col

    def __call__(
            self, df: pd.DataFrame,
            gov_name: str = "head_current",
            dep_name: str = "child_current",
            only_content_words_left: bool = False,
            only_content_words_cost: bool = False,
            content_pos: Collection[str] = CONTENT_POS[TAGSET],
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        return pd.Series([
            generate_left_dependents_distance_sum(
                mask, pos, content_pos=content_pos,
                gov_name=gov_name,
                dep_name=dep_name,
                only_content_words_left=only_content_words_left,
                only_content_words_cost=only_content_words_cost)
            for mask, pos, in zip(
                df[self.mask_col],
                df[self.pos_col])]), {
            "content_pos": content_pos,
            "only_content_words_left": only_content_words_left,
            "only_content_words_cost": only_content_words_cost}


class SplitTokMetricMakerLeftDependentsCount(SplitTokMetricMaker):
    def __init__(
            self, mask_col: str, pos_col: str, *args, **kwargs):
        self.mask_col = mask_col
        self.pos_col = pos_col

    def __call__(
            self, df: pd.DataFrame,
            gov_name: str = "head_current",
            dep_name: str = "child_current",
            only_content_words_left: bool = False,
            only_content_words_cost: bool = False,
            content_pos: Collection[str] = CONTENT_POS[TAGSET],
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        return pd.Series([
            generate_left_dependents_count(
                mask, pos, content_pos=content_pos,
                gov_name=gov_name,
                dep_name=dep_name,
                only_content_words_left=only_content_words_left,
                only_content_words_cost=only_content_words_cost)
            for mask, pos, in zip(
                df[self.mask_col],
                df[self.pos_col])]), {
            "content_pos": content_pos,
            "only_content_words_left": only_content_words_left,
            "only_content_words_cost": only_content_words_cost}


class SplitTokMetricMakerExpectedDistance(SplitTokMetricMaker):
    def __init__(
            self, mask_col: str, pos_col: str,
            first_dependent_distance_col: str,
            *args, **kwargs):
        self.mask_col = mask_col
        self.pos_col = pos_col
        self.first_dependent_distance_col = first_dependent_distance_col

    def __call__(
            self, df: pd.DataFrame,
            attention_logits: dict[str, np.ndarray],
            masks_setting: Literal["current", "next"] = "current",
            gov_name: str = "head_current",
            dep_name: str = "child_current",
            only_content_words_left: bool = False,
            content_pos: Collection[str] = CONTENT_POS[TAGSET],
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        attention_matrices = [
            {name: masks[i] for name, masks in attention_logits.items()}
            for i in range(len(list(attention_logits.values())[0]))]
        return pd.Series([
            generate_expected_distance(
                att_mat, pos,
                masks_setting=masks_setting,
                content_pos=content_pos,
                gov_name=gov_name,
                dep_name=dep_name,
                only_content_words_left=only_content_words_left)
            for att_mat, pos in zip(
                attention_matrices,
                df[self.pos_col])]), {
            "content_pos": content_pos,
            "masks_setting": masks_setting,
            "only_content_words_left": only_content_words_left
        }


class SplitTokMetricMakerKLDivergence(SplitTokMetricMaker):
    def __init__(
            self, mask_col: str, *args, **kwargs):
        self.mask_col = mask_col

    def __call__(
            self, df: pd.DataFrame,
            attention_logits: dict[str, np.ndarray],
            gov_name: str = "head_current",
            dep_name: str = "child_current",
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        attention_matrices = [
            {name: masks[i] for name, masks in attention_logits.items()}
            for i in range(len(list(attention_logits.values())[0]))]
        return pd.Series([
            generate_kl_divergence(
                mask, att_mat,
                gov_name=gov_name,
                dep_name=dep_name)
            for mask, att_mat in zip(
                df[self.mask_col],
                attention_matrices)]), {
        }


class SplitTokMetricMakerDemberg(SplitTokMetricMaker):
    def __init__(
            self, mask_col: str, pos_col: str,
            *args, **kwargs):
        self.mask_col = mask_col
        self.pos_col = pos_col

    def __call__(
            self, df: pd.DataFrame,
            gov_name: str = "head_current",
            dep_name: str = "child_current",
            only_content_words_left: bool = False,
            only_content_words_cost: bool = False,
            content_pos: Collection[str] = CONTENT_POS[TAGSET],
            *args, **kwargs) -> tuple[pd.Series, dict[str, Any]]:
        return pd.Series([
            generate_demberg(
                mask, pos, content_pos=content_pos,
                gov_name=gov_name,
                dep_name=dep_name,
                only_content_words_left=only_content_words_left,
                only_content_words_cost=only_content_words_cost,
                with_non_referent_establishment_cost=(
                    not only_content_words_cost))
            for mask, pos in zip(
                df[self.mask_col],
                df[self.pos_col])]), {
            "content_pos": content_pos,
            "only_content_words_left": only_content_words_left,
            "only_content_words_cost": only_content_words_cost
        }


def get_POS_tags(sentence: list[str]) -> list[str]:
    tagged = nltk.tag.pos_tag(sentence, tagset=TAGSET)
    return [pos_merge(tag) for _, tag in tagged]


def generate_head_distance(
        pos_tags: Sequence[str],
        masks: dict[str, np.ndarray],
        only_content_words_cost: bool = False,
        gov_name: str = "head_current",
        dep_name: str = "child_current",
        content_pos: Collection[str] = CONTENT_POS[TAGSET]
        ) -> npt.NDArray[np.int_]:
    """WARNING: untested"""
    # assumes that governor and child mask are called head and child
    # these are already triangulated
    mask_gov = masks[gov_name].copy()
    mask_dep = masks[dep_name].copy()

    assert mask_gov is not None and mask_dep is not None

    content_word_mask = get_content_word_mask_by_POS_tags(
        pos_tags,
        content_pos,
        dummies_present=False)

    mask_gov[:, 0] = 0
    mask_dep[:, 0] = 0      # no cost if no children
    mask = mask_gov + mask_dep.T
    mask[
        np.arange(mask.shape[0]),
        np.arange(mask.shape[0])] = mask_gov[:, 1]
    mask[:, 1] = 0

    # restrict this to new content words OR IDEA: do not add if
    # right head (all intermediate elements have the same head)
    # has a head left of this element

    length = mask_gov.shape[0]
    indices: npt.NDArray[np.int_] = np.tile(
        np.flip(np.arange(1, length+1)), length).reshape(length, -1)
    indices = indices - np.flip(
        np.arange(1, length+1))[..., np.newaxis]
    indices = indices.T

    costs = (mask * indices).sum(1)

    costs = costs[2:]
    if only_content_words_cost:
        costs[~content_word_mask] = 0
    assert len(pos_tags) == len(costs)
    cost_array: npt.NDArray[np.int_] = np.array(costs)
    return cost_array


def generate_first_dep_distance(
        pos_tags: Sequence[str],
        masks: dict[str, np.ndarray],
        gov_name: str = "head_current",
        dep_name: str = "child_current",
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        ) -> npt.NDArray[np.int_]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    """WARNING: untested"""

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    assert masks[gov_name] is not None
    assert masks[dep_name] is not None
    mask_gov = masks[gov_name].copy()
    mask_dep = masks[dep_name].copy()

    content_word_mask = get_content_word_mask_by_POS_tags(
        pos_tags,
        CONTENT_POS[TAGSET],
        dummies_present=False)
    if only_content_words_left:

        mask_gov[:, 2:][:, ~content_word_mask] = False
        mask_dep[:, 2:][:, ~content_word_mask] = False

    assert mask_gov is not None and mask_dep is not None

    mask_dep[:, 1] = 0
    mask_dep[:, 0] = 0      # no cost if no children
    mask_gov[:, 1] = 0
    mask_gov[:, 0] = 0

    mask = np.tril(np.logical_or(mask_gov, mask_dep))
    mask = np.logical_or(mask, mask.T)

    first_connection: npt.NDArray[np.int_] = np.argmax(
        mask, axis=1)

    length = mask_gov.shape[0]
    indices: npt.NDArray[np.int_] = np.tile(
        np.flip(np.arange(1, length+1)), length).reshape(length, -1)
    indices = indices - np.flip(
        np.arange(1, length+1))[..., np.newaxis]

    distances = -indices[
        np.arange(0, first_connection.shape[0]),
        first_connection]

    distances[first_connection == 0] = 0    # if root, then distance 0

    costs = distances[2:]
    if only_content_words_cost:
        costs[~content_word_mask] = 0

    return costs


def generate_first_dependent_correct(
        attention_matrices: dict[str, torch.Tensor],
        first_dependent_distance: Sequence[int],
        pos_tags: Sequence[str],
        gov_name: str = "head_current",
        dep_name: str = "child_current",
        only_content_words_left: bool = False,
        content_pos: Collection[str] = CONTENT_POS[TAGSET]
        ) -> npt.NDArray[np.bool]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    # TODO: Discard punctuation
    """WARNING: untested"""

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    mask_gov = torch.sigmoid(attention_matrices[gov_name][0].clone())
    mask_dep = torch.sigmoid(attention_matrices[dep_name][0].clone())

    if only_content_words_left:
        # disregard connections to non-content words
        content_word_mask = get_content_word_mask_by_POS_tags(
            pos_tags,
            content_pos,
            dummies_present=False)
        mask_gov[:, 2:][:, ~content_word_mask] = 0
        mask_dep[:, 2:][:, ~content_word_mask] = 0

    # mask_gov_ = np.tril(mask_gov) >= 0.5
    # mask_dep_ = np.tril(mask_dep) >= 0.5
    # mask = np.tril(np.logical_or(mask_gov_, mask_dep_))

    # first_connection: npt.NDArray[np.int_] = np.argmax(
    #    mask, axis=1)
    mask_gov_ = np.tril(mask_gov)
    mask_dep_ = np.tril(mask_dep)
    first_connection_gov: npt.NDArray[np.int_] = np.argmax(
        mask_gov_, axis=1)
    first_connection_dep: npt.NDArray[np.int_] = np.argmax(
        mask_dep_, axis=1)
    first_connection = np.minimum(
        first_connection_gov, first_connection_dep)
    no_left_head = first_connection_gov < 2
    no_left_child = first_connection_dep < 2

    cond = np.logical_and(no_left_head, np.logical_not(no_left_child))
    first_connection[cond] = first_connection_dep[cond]

    cond = np.logical_and(no_left_child, np.logical_not(no_left_head))
    first_connection[cond] = first_connection_gov[cond]

    length = mask_gov.shape[0]
    indices: npt.NDArray[np.int_] = np.tile(
        np.flip(np.arange(1, length+1)), length).reshape(length, -1)
    indices = indices - np.flip(
        np.arange(1, length+1))[..., np.newaxis]

    distances = -indices[
        np.arange(0, first_connection.shape[0]),
        first_connection]
    distances[np.logical_and(no_left_head, no_left_child)] = 0
    # distance where there is no left element is set to 0

    distances = distances[2:]

    fdd_sen = np.copy(first_dependent_distance)
    fdd_sen[fdd_sen > 0] = 0
    fddc = fdd_sen == distances

    cost_array = np.array(fddc)

    return cost_array


def generate_first_dependent_distance_weight(
        masks: dict[str, np.ndarray],
        attention_matrices: dict[str, torch.Tensor],
        gov_name: str = "head_current",
        dep_name: str = "child_current",
        ) -> npt.NDArray[np.bool]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    """WARNING: untested"""

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    assert masks[gov_name] is not None
    assert masks[dep_name] is not None
    mask_gov = masks[gov_name].copy()
    mask_dep = masks[dep_name].copy()
    assert mask_gov is not None and mask_dep is not None

    # throw root connection and future connection together
    mask_dep[:, 0] = mask_dep[:, 0] + mask_dep[:, 1]
    mask_dep[:, 1] = 0
    mask_gov[:, 0] = mask_gov[:, 0] + mask_gov[:, 1]
    mask_gov[:, 1] = 0

    # first dependent only future dependent
    # if no left gov and dep dependent
    mask = np.tril(np.logical_or(mask_gov, mask_dep))
    mask[:, 0] = np.logical_and(mask_gov[:, 0], mask_dep[:, 0])

    first_connection: npt.NDArray[np.int_] = np.argmax(
        mask, axis=1)

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    mask_gov_pred = torch.sigmoid(
        attention_matrices[gov_name][0].clone())
    mask_dep_pred = torch.sigmoid(
        attention_matrices[dep_name][0].clone())

    # TODO to softmax
    mask_gov_pred = inverse_sigmoid(mask_gov_pred)
    mask_gov_pred = mask_gov_pred.softmax(-1)
    mask_dep_pred = inverse_sigmoid(mask_dep_pred).softmax(-1)
    mask_dep_pred = mask_dep_pred * mask_dep.sum(1)

    mask_dep_pred[:, 0] = mask_dep_pred[:, 0] + mask_dep_pred[:, 1]
    mask_dep_pred[:, 1] = 0
    mask_gov_pred[:, 0] = mask_gov_pred[:, 0] + mask_gov_pred[:, 1]
    mask_gov_pred[:, 1] = 0

    weights = torch.maximum(
        mask_gov_pred[
            np.arange(0, first_connection.shape[0]),
            first_connection],
        torch.minimum(
            mask_dep_pred[
                np.arange(0, first_connection.shape[0]),
                first_connection],
            torch.ones(
                (1,), device=first_connection.device)))
    # right weight

    return weights[2:].detach().numpy()


def generate_first_dependent_deprel(
        masks: dict[str, np.ndarray],
        pos_tags: Sequence[str],
        deprels: Sequence[str],
        gov_name: str = "head_current",
        dep_name: str = "child_current",
        only_content_words_left: bool = False,
        content_pos: Collection[str] = CONTENT_POS[TAGSET]
        ) -> list[str]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    """WARNING: untested"""

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    assert masks[gov_name] is not None
    assert masks[dep_name] is not None
    mask_gov = masks[gov_name].copy()
    mask_dep = masks[dep_name].copy()

    if only_content_words_left:
        content_word_mask = get_content_word_mask_by_POS_tags(
            pos_tags,
            content_pos,
            dummies_present=False)

        mask_gov[:, 2:][:, ~content_word_mask] = False
        mask_dep[:, 2:][:, ~content_word_mask] = False

    assert mask_gov is not None and mask_dep is not None

    root_idx = 1
    mask_dep[:, root_idx] = 0
    mask_dep[:, 0] = 0      # no cost if no children
    mask_gov[:, root_idx] = 0
    mask_gov[:, 0] = 0

    # mask = np.tril(np.logical_or(mask_gov, mask_dep))
    mask_gov = np.logical_or(mask_gov, mask_dep.T)
    mask_dep = np.logical_or(mask_dep, mask_gov.T)

    first_connection_gov: npt.NDArray[np.int_] = np.argmax(
        mask_gov, axis=1)
    first_connection_dep: npt.NDArray[np.int_] = np.argmax(
        mask_dep, axis=1)

    deprels_sen = ["dummy", "main_root"] + list(deprels)
    fdd_deprels: list[str] = []
    for i, (fc_g, fc_d) in enumerate(
            zip(first_connection_gov, first_connection_dep)):
        if fc_g == 0:
            fdd_deprels.append(deprels_sen[fc_d])
        elif fc_d == 0:
            fdd_deprels.append(deprels_sen[i])
        else:
            if fc_d < fc_g:
                fdd_deprels.append(deprels_sen[fc_d])
            else:
                fdd_deprels.append(deprels_sen[i])

    return fdd_deprels[2:]


def generate_left_dependents_distance_sum(
        masks: dict[str, np.ndarray],
        pos_tags: Sequence[str],
        gov_name: str = "head_current",
        dep_name: str = "child_current",
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        content_pos: Collection[str] = CONTENT_POS[TAGSET]
        ) -> npt.NDArray[np.int_]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    """WARNING: untested"""

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    mask_gov = masks[gov_name].copy()
    mask_dep = masks[dep_name].copy()

    assert mask_gov is not None and mask_dep is not None

    root_idx = 1
    mask_dep[:, root_idx] = 0
    mask_dep[:, 0] = 0          # no cost if no children
    mask_gov[:, root_idx] = 0
    mask_gov[:, 0] = 0

    content_word_mask = get_content_word_mask_by_POS_tags(
        pos_tags,
        content_pos,
        dummies_present=False)
    if only_content_words_left:
        # disregard connections to non-content words
        mask_gov[:, 2:][:, ~content_word_mask] = 0
        mask_dep[:, 2:][:, ~content_word_mask] = 0

    mask = np.tril(np.logical_or(mask_gov, mask_dep))

    length = mask_gov.shape[0]
    indices: npt.NDArray[np.int_] = np.tile(
        np.flip(np.arange(1, length+1)), length).reshape(length, -1)
    indices = indices - np.flip(
        np.arange(1, length+1))[..., np.newaxis]

    costs = (mask * indices).sum(1)

    costs = costs[2:]
    if only_content_words_cost:
        costs[~content_word_mask] = 0

    return costs


def generate_left_dependents_count(
        masks: dict[str, np.ndarray],
        pos_tags: Sequence[str],
        gov_name: str = "head_current",
        dep_name: str = "child_current",
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        content_pos: Collection[str] = CONTENT_POS[TAGSET]
        ) -> npt.NDArray[np.int_]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    """WARNING: untested"""

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    mask_gov = masks[gov_name].copy()
    mask_dep = masks[dep_name].copy()

    assert mask_gov is not None and mask_dep is not None

    root_idx = 1
    mask_dep[:, root_idx] = 0
    mask_dep[:, 0] = 0      # no cost if no children
    mask_gov[:, root_idx] = 0
    mask_gov[:, 0] = 0

    content_word_mask = get_content_word_mask_by_POS_tags(
        pos_tags,
        content_pos,
        dummies_present=False)
    if only_content_words_left:
        # disregard connections to non-content words
        mask_gov[:, 2:][:, ~content_word_mask] = 0
        mask_dep[:, 2:][:, ~content_word_mask] = 0

    mask = np.tril(np.logical_or(mask_gov, mask_dep))
    costs = mask.sum(1)

    costs = costs[2:]
    if only_content_words_cost:
        costs[~content_word_mask] = 0

    return costs


def generate_expected_distance(
        attention_matrices: dict[str, torch.Tensor],
        pos_tags: Sequence[str],
        gov_name: str = "head_current",
        dep_name: str = "child_current",
        only_content_words_left: bool = True,
        masks_setting: Literal["current", "next"] = "current",
        content_pos: Collection[str] = CONTENT_POS[TAGSET]) -> npt.NDArray:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    # TODO: Discard punctuation
    """WARNING: untested"""

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    mask_gov = attention_matrices[gov_name][0].clone().softmax(-1)
    mask_dep = attention_matrices[dep_name][0].clone().softmax(-1)

    mask_gov[:, 0] = 0
    mask_gov[:, 1] = 0
    mask_dep[:, 0] = 0
    mask_dep[:, 1] = 0

    if only_content_words_left:
        # disregard connections to non-content words
        content_word_mask = get_content_word_mask_by_POS_tags(
            pos_tags,
            content_pos,
            dummies_present=False)
        mask_gov[:, 2:][:, ~content_word_mask] = 0
        mask_dep[:, 2:][:, ~content_word_mask] = 0

    s = mask_gov.shape[0]
    r = torch.arange(1, s+1)
    dist_mat = torch.tril(-1 * (r.repeat(s, 1) - r.reshape(-1, 1)))

    if masks_setting == "next":
        dist_mat += 1

    cost = (dist_mat*(mask_gov + mask_dep)).sum(-1)

    return cost[2:].numpy()


def generate_kl_divergence(
        masks: dict[str, np.ndarray],
        attention_matrices: dict[str, torch.Tensor],
        gov_name: str = "head_current",
        dep_name: str = "child_current",
        ) -> npt.NDArray:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    # TODO: Discard punctuation
    """WARNING: untested"""

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    mask_gov = (attention_matrices[
        gov_name].mean(0).clone()).softmax(-1).numpy()
    mask_dep = (attention_matrices[
        dep_name].mean(0).clone()).softmax(-1).numpy()

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    assert masks[gov_name] is not None
    assert masks[dep_name] is not None
    mask_gov_gold = masks[gov_name].copy().astype(float)
    mask_dep_gold = masks[dep_name].copy().astype(float)

    mask_gov_gold /= mask_gov_gold.sum(-1)
    mask_dep_gold /= mask_dep_gold.sum(-1)

    # if masks_setting == "next":
    #     zeros = np.full((
    #         1, mask_gov_gold.shape[1]), False)
    #     mask_gov = np.concatenate(
    #         (zeros, mask_gov[:-1]), axis=0)
    #     mask_dep = np.concatenate(
    #         (zeros, mask_dep[:-1]), axis=0)
    #     mask_gov_gold = np.concatenate(
    #         (zeros, mask_gov_gold[:-1]), axis=0)
    #     mask_dep_gold = np.concatenate(
    #         (zeros, mask_dep_gold[:-1]), axis=0)

    # print(mask_gov_gold / mask_gov)

    kl_gov = (mask_gov_gold*np.log(mask_gov_gold / mask_gov))
    kl_dep = (mask_dep_gold*np.log(mask_dep_gold / mask_dep))

    kl_gov[np.isnan(kl_gov)] = 0
    kl_dep[np.isnan(kl_dep)] = 0

    kl_gov[np.isinf(kl_gov)] = 0
    kl_dep[np.isinf(kl_dep)] = 0
    
    cost = (kl_dep + kl_gov).sum(-1)[2:]

    return cost


def generate_demberg(
        masks: dict[str, np.ndarray],
        pos_tags: Sequence[str],
        gov_name: str = "head_current",
        dep_name: str = "child_current",
        only_content_words_left: bool = True,
        only_content_words_cost: bool = True,
        with_search: bool = False,
        with_non_referent_establishment_cost: bool = False,
        content_pos: Collection[str] = CONTENT_POS[TAGSET]
        ) -> npt.NDArray[np.int_]:
    """WARNING: untested"""

    # assumes that governor and child mask are called head and child
    # these are already triangulated
    mask_gov = masks[gov_name].copy()
    mask_dep = masks[dep_name].copy()

    assert mask_gov is not None and mask_dep is not None

    root_idx = 1
    mask_dep[:, root_idx] = 0
    mask_dep[:, 0] = 0      # no cost if no children
    mask_gov[:, root_idx] = 0
    mask_gov[:, 0] = 0

    cw = get_content_word_mask_by_POS_tags(
        pos_tags,
        content_pos,
        dummies_present=False)

    costs = get_content_word_cost_Demberg(
        cw,
        mask_gov, mask_dep,
        only_content_words_left=only_content_words_left,
        only_content_words_cost=only_content_words_cost,
        with_search=with_search,
        with_non_referent_establishment_cost=(
            with_non_referent_establishment_cost))[2:]

    return costs


def get_content_word_mask_by_POS_tags(
        pos_tags: Sequence[str],
        not_to_mask: Collection[str] = CONTENT_POS[TAGSET],
        dummies_present: bool = True) -> npt.NDArray[np.bool]:
    mask = np.array([tag in not_to_mask for tag in pos_tags], dtype=bool)
    if dummies_present:
        mask[0:2] = False
    return mask


def get_content_word_cost_Demberg(
        content_word_mask: npt.NDArray[np.bool],
        mask_gov: npt.NDArray[np.bool],
        mask_dep: npt.NDArray[np.bool],
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        with_search: bool = False,
        with_non_referent_establishment_cost: bool = True
        ) -> npt.NDArray[np.int_]:
    mask = np.logical_or(mask_gov, mask_dep)

    content_word_mask = np.concat((
        np.array([False, False]), content_word_mask))
    content_word_mask_left = content_word_mask.copy()

    if not only_content_words_left:
        content_word_mask_left[...] = 1
    # do not take into account connections to function words
    else:
        mask[:, ~content_word_mask_left] = 0
    mask[0, :] = 0
    mask[1, :] = 0
    mask[:, 0] = 0
    mask[:, 1] = 0

    first_connection: npt.NDArray[np.int_] = np.argmax(np.tril(mask), axis=1)

    upper_triang = np.triu(
        np.ones(
            (content_word_mask_left.shape[0], content_word_mask_left.shape[0])
            ))
    cw_cumul: np.ndarray = content_word_mask_left @ upper_triang
    # print(first_connection)
    # print(cw_cumul)
    cw_before_first = cw_cumul[first_connection]  # @ mask_gov.T
    # print(cw_before_first)
    cw_intermediate = cw_cumul - cw_before_first
    # print(cw_intermediate)
    # assume no crossing arcs
    # left_cw_child_nums =voc content_word_mask @ mask_dep.T
    # cw_intermediate -= left_cw_child_nums

    cw_intermediate[content_word_mask_left] -= 1
    if not with_search:
        cw_intermediate[first_connection == 0] = 0  # if dummy or root
    # print(np.sum(cw_intermediate == -1))

    # cw_intermediate[mask_gov[:, 0]] = 1  # for dummy set 1
    # cw_intermediate[mask_gov[:, 1]] = 1  # for root set 1

    # new discourse referents
    if only_content_words_cost:
        # if with_non_referent_establishment_cost:
        #     cost[~content_word_mask] = 1  # integration and instantiation
        # else:
        cw_intermediate[~content_word_mask] = 0
    # elif not with_non_referent_establishment_cost:
    #     cost[~content_word_mask] -= 1  # integration and instantiation

    if with_non_referent_establishment_cost:
        cw_intermediate[...] += 1
    else:
        cw_intermediate[content_word_mask] += 1
    # cost for function words is 0
    # raise Exception
    return cw_intermediate

# TODO: Vera Demberg integration cost
# referents = nouns & verbs
# cost: 0 for all non-referents
# all others: number of referents between first left dependency and token
# def generate_integration_Gibson(
# )


# maker, shift (yes, no, ignore), untokeniser/ignore, unsplit
gen_and_untok: dict[str, tuple[
    Type[SplitTokMetricMaker],
    bool | None,
    Type[UntokSplitFunc] | None,
    bool]] = {
        "pos": (SplitTokMetricMakerPOS, False, UntokSplitHead, True),
        "word": (SplitTokMetricMakerWord, False, UntokSplitAdd, True),
        "space_after": (SplitTokMetricMakerSpaceAfter, False, None, False),
        "position": (SplitTokMetricMakerPosition, False, UntokSplitFirst, True),
        "conllu": (SplitTokMetricMakerTokenlist, None, None, False),
        "deprel": (SplitTokMetricMakerDeprels, False, UntokSplitHead, True),
        "head": (SplitTokMetricMakerHeadlist, False, None, False),
        # TODO: use different detokenisation since head is shifted when
        # untokenising. Side note: This also holds for costs and distances
        # maybe one should produce them after detokenising
        "surprisal": (SplitTokMetricMakerSurprisal, False, UntokSplitAdd, True),
        "mask": (SplitTokMetricMakerMask, None, None, False),
        "head_distance": (
            SplitTokMetricMakerHeadDistance, True, UntokSplitHead, True),
        "first_dependent_distance": (
            SplitTokMetricMakerFirstDependentDistance, True,
            UntokSplitHead, True),
        "first_dependent_correct": (
            SplitTokMetricMakerFirstDependentCorrect, False,
            UntokSplitHead, True),
        "first_dependent_distance_weight": (
            SplitTokMetricMakerFirstDependentDistanceWeight,
            False, UntokSplitHead, True),
        "first_dependent_deprel": (
            SplitTokMetricMakerFirstDependentDeprel,
            True, UntokSplitHead, True),
        "left_dependents_distance_sum": (
            SplitTokMetricMakerLeftDependentsDistanceSum,
            True, UntokSplitHead, True),
        "left_dependents_count": (
            SplitTokMetricMakerLeftDependentsCount,
            True, UntokSplitHead, True),
        "expected_distance": (
            SplitTokMetricMakerExpectedDistance,
            False, UntokSplitHead, True),
        "kl_divergence": (
            SplitTokMetricMakerKLDivergence,
            False, UntokSplitHead, True),
        "demberg": (
            SplitTokMetricMakerDemberg,
            True, UntokSplitHead, True)
    }


# TODO: allow component head untok (first connection outside?)

# if self.transform_mask is not None:
#             masks.update(
#                 self.transform_mask(
#                     head_list_to_adjacency_matrix(heads)).items())
#
#         masks = shift_masks(self.masks_setting, masks)

# NECESSARY: way to add multiple columns (one for each mask)
