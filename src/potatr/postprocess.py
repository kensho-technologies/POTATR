# Copyright 2025-present Kensho Technologies, LLC.
# Portions copyright (C) 2021 Microsoft Corporation, from Table Transformer
# (https://github.com/microsoft/table-transformer), MIT License.
# Third-party portions have been modified. See THIRD_PARTY_NOTICES.md.

from collections import defaultdict
from collections.abc import Sequence
import logging
from typing import Any

import torch

from potatr.geometry import rescale_bboxes


logger = logging.getLogger(__name__)

# A bounding box, [x0, y0, x1, y1] in pixels.
BBox = list[float]
# A detected object, word or table structure, as a dict with at least a "bbox".
Object = dict[str, Any]


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_intersect(a: Sequence[float], b: Sequence[float]) -> BBox:
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def _bbox_include(current: BBox | None, bbox: Sequence[float]) -> BBox:
    """Union accumulator with None as identity (replaces Rect().include_rect)."""
    if current is None:
        return list(bbox)
    return [
        min(current[0], bbox[0]),
        min(current[1], bbox[1]),
        max(current[2], bbox[2]),
        max(current[3], bbox[3]),
    ]


class_idx2name = [
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
    "table caption",
    "table footer",
    "table rotated",
    "table column rotated",
    "table row rotated",
    "table column header rotated",
    "table projected row header rotated",
    "table spanning cell rotated",
    "table caption rotated",
    "table footer rotated",
    "no object",
]
class_thresholds = {k: 0.8 for k in class_idx2name}
class_thresholds["no object"] = 10


def apply_threshold(objects: list[Object], threshold: float) -> list[Object]:
    """Filter out objects below a certain score."""
    return [obj for obj in objects if obj["score"] >= threshold]


def iob(bbox1: Sequence[float], bbox2: Sequence[float]) -> float:
    """Compute the intersection area over box area, for bbox1."""
    bbox1_area = _bbox_area(bbox1)
    if bbox1_area > 0:
        return _bbox_area(_bbox_intersect(bbox1, bbox2)) / bbox1_area

    return 0


def refine_rows(rows: list[Object], tokens: list[Object]) -> list[Object]:
    """Suppress overlapping rows, remove rows with no text and sort the rest top to bottom.

    With words, a row is suppressed if it shares a word with a higher-scoring row; without words,
    if at least half of its area overlaps a higher-scoring row.
    """

    if len(tokens) > 0:
        rows = nms_by_containment(rows, tokens, overlap_threshold=0.5)
        remove_objects_without_content(tokens, rows)
    else:
        rows = nms(rows, match_criteria="object2_overlap", match_threshold=0.5, keep_higher=True)
    if len(rows) > 1:
        rows = sort_objects_top_to_bottom(rows)

    return rows


def refine_columns(columns: list[Object], tokens: list[Object]) -> list[Object]:
    """Suppress overlapping columns, remove columns with no text and sort the rest left to right.

    With words, a column is suppressed if it shares a word with a higher-scoring column; without
    words, if at least a quarter of its area overlaps a higher-scoring column.
    """

    if len(tokens) > 0:
        columns = nms_by_containment(columns, tokens, overlap_threshold=0.5)
        remove_objects_without_content(tokens, columns)
    else:
        columns = nms(
            columns, match_criteria="object2_overlap", match_threshold=0.25, keep_higher=True
        )
    if len(columns) > 1:
        columns = sort_objects_left_to_right(columns)

    return columns


def nms_by_containment(
    container_objects: list[Object], package_objects: list[Object], overlap_threshold: float = 0.5
) -> list[Object]:
    """Non-maxima suppression (NMS) of objects based on shared containment of other objects."""
    container_objects = sort_objects_by_score(container_objects)
    num_objects = len(container_objects)
    suppression = [False for obj in container_objects]

    packages_by_container, _, _ = slot_into_containers(
        container_objects,
        package_objects,
        overlap_threshold=overlap_threshold,
        unique_assignment=True,
        forced_assignment=False,
    )

    for object2_num in range(1, num_objects):
        object2_packages = set(packages_by_container[object2_num])
        if len(object2_packages) == 0:
            suppression[object2_num] = True
        for object1_num in range(object2_num):
            if not suppression[object1_num]:
                object1_packages = set(packages_by_container[object1_num])
                if len(object2_packages.intersection(object1_packages)) > 0:
                    suppression[object2_num] = True

    final_objects = [obj for idx, obj in enumerate(container_objects) if not suppression[idx]]
    return final_objects


def slot_into_containers(
    container_objects: list[Object],
    package_objects: list[Object],
    overlap_threshold: float = 0.5,
    unique_assignment: bool = True,
    forced_assignment: bool = False,
) -> tuple[list[list[int]], list[list[int]], list[float]]:
    """Slot each package object into the container that holds the largest fraction of its area.

    A package is slotted in if that fraction is at least overlap_threshold, or always with
    forced_assignment. With unique_assignment False, it is also slotted into every other container
    holding at least overlap_threshold of it. Returns (package indices by container, container
    indices by package, each package's largest fraction).
    """
    best_match_scores: list[float] = []

    container_assignments: list[list[int]] = [[] for container in container_objects]
    package_assignments: list[list[int]] = [[] for package in package_objects]

    if len(container_objects) == 0 or len(package_objects) == 0:
        return container_assignments, package_assignments, best_match_scores

    for package_num, package in enumerate(package_objects):
        match_scores = []
        package_area = _bbox_area(package["bbox"])
        for container_num, container in enumerate(container_objects):
            intersect_area = _bbox_area(_bbox_intersect(container["bbox"], package["bbox"]))
            overlap_fraction = intersect_area / package_area
            match_scores.append(
                {"container": container, "container_num": container_num, "score": overlap_fraction}
            )

        sorted_match_scores = sort_objects_by_score(match_scores)

        best_match_score = sorted_match_scores[0]
        best_match_scores.append(best_match_score["score"])
        if forced_assignment or best_match_score["score"] >= overlap_threshold:
            container_assignments[best_match_score["container_num"]].append(package_num)
            package_assignments[package_num].append(best_match_score["container_num"])

        if not unique_assignment:  # slot package into all eligible slots
            for match_score in sorted_match_scores[1:]:
                if match_score["score"] >= overlap_threshold:
                    container_assignments[match_score["container_num"]].append(package_num)
                    package_assignments[package_num].append(match_score["container_num"])
                else:
                    break

    return container_assignments, package_assignments, best_match_scores


def sort_objects_by_score(objects: list[Object], reverse: bool = True) -> list[Object]:
    """Put any set of objects in order from high score to low score."""
    if reverse:
        sign = -1
    else:
        sign = 1
    return sorted(objects, key=lambda k: sign * k["score"])


def remove_objects_without_content(page_spans: list[Object], objects: list[Object]) -> None:
    """Remove, in place, any objects (rows, columns, supercells, etc.) that contain no text."""
    for obj in objects[:]:
        object_text, _ = extract_text_inside_bbox(page_spans, obj["bbox"])
        if len(object_text.strip()) == 0:
            objects.remove(obj)


def extract_text_inside_bbox(
    spans: list[Object], bbox: Sequence[float]
) -> tuple[str, list[Object]]:
    """Extract the text inside a bounding box."""
    bbox_spans = get_bbox_span_subset(spans, bbox)
    bbox_text = extract_text_from_spans(bbox_spans)

    return bbox_text, bbox_spans


def get_bbox_span_subset(
    spans: list[Object], bbox: Sequence[float], threshold: float = 0.5
) -> list[Object]:
    """Reduce the set of spans to those that fall within a bounding box.

    threshold: the fraction of the span that must overlap with the bbox.
    """
    span_subset = []
    for span in spans:
        if overlaps(span["bbox"], bbox, threshold):
            span_subset.append(span)
    return span_subset


def overlaps(bbox1: Sequence[float], bbox2: Sequence[float], threshold: float = 0.5) -> bool:
    """Test if more than "threshold" fraction of bbox1 overlaps with bbox2."""
    area1 = _bbox_area(bbox1)
    if area1 == 0:
        return False
    return _bbox_area(_bbox_intersect(bbox1, bbox2)) / area1 >= threshold


def extract_text_from_spans(spans: list[Object], join_with_space: bool = True) -> str:
    """Convert a collection of page tokens/words/spans into a single text string."""

    if join_with_space:
        join_char = " "
    else:
        join_char = ""
    spans_copy = spans[:]

    if len(spans_copy) == 0:
        return ""

    spans_copy.sort(key=lambda span: span["span_num"])
    spans_copy.sort(key=lambda span: span["line_num"])
    spans_copy.sort(key=lambda span: span["block_num"])

    # Force the span at the end of every line within a block to have exactly one space
    # unless the line ends with a space or ends with a non-space followed by a hyphen
    line_texts = []
    line_span_texts = [spans_copy[0]["text"].strip()]
    for span1, span2 in zip(spans_copy[:-1], spans_copy[1:]):
        if (
            not span1["block_num"] == span2["block_num"]
            or not span1["line_num"] == span2["line_num"]
        ):
            line_text = join_char.join(line_span_texts).strip()
            if (
                len(line_text) > 0
                and not line_text[-1] == " "
                and not (len(line_text) > 1 and line_text[-1] == "-" and not line_text[-2] == " ")
            ):
                if not join_with_space:
                    line_text += " "
            line_texts.append(line_text)
            line_span_texts = [span2["text"].strip()]
        else:
            line_span_texts.append(span2["text"].strip())
    line_text = join_char.join(line_span_texts)
    line_texts.append(line_text)

    return join_char.join(line_texts).strip()


def sort_objects_left_to_right(objs: list[Object]) -> list[Object]:
    """Put the objects in order from left to right."""
    return sorted(objs, key=lambda k: k["bbox"][0] + k["bbox"][2])


def sort_objects_top_to_bottom(objs: list[Object]) -> list[Object]:
    """Put the objects in order from top to bottom."""
    return sorted(objs, key=lambda k: k["bbox"][1] + k["bbox"][3])


def align_columns(columns: list[Object], bbox: Sequence[float]) -> list[Object]:
    """Align the top and bottom of every column to the table bounding box, in place."""
    try:
        for column in columns:
            column["bbox"][1] = bbox[1]
            column["bbox"][3] = bbox[3]
    except (TypeError, IndexError) as err:
        logger.warning("Could not align columns: %s", err)

    return columns


def align_rows(rows: list[Object], bbox: Sequence[float]) -> list[Object]:
    """Align the left and right of every row to the table bounding box, in place."""
    try:
        for row in rows:
            row["bbox"][0] = bbox[0]
            row["bbox"][2] = bbox[2]
    except (TypeError, IndexError) as err:
        logger.warning("Could not align rows: %s", err)

    return rows


def refine_table_structures(
    table_structures: dict[str, list[Object]], class_thresholds: dict[str, float]
) -> dict[str, list[Object]]:
    """Threshold, align and suppress a table's headers and supercells.

    Headers and supercells below their class thresholds are dropped. The header is aligned to the
    rows (which marks the header rows), and supercells are aligned to the rows and columns, their
    overlaps resolved and any that break the header's tree structure removed. Rows and columns
    are otherwise unchanged.
    """
    rows = table_structures["rows"]
    columns = table_structures["columns"]

    # Process the headers
    headers = table_structures["headers"]
    headers = apply_threshold(headers, class_thresholds["table column header"])
    headers = nms(headers)
    headers = align_headers(headers, rows)

    # Process supercells
    supercells = [elem for elem in table_structures["supercells"] if not elem["subheader"]]
    subheaders = [elem for elem in table_structures["supercells"] if elem["subheader"]]
    supercells = apply_threshold(supercells, class_thresholds["table spanning cell"])
    subheaders = apply_threshold(subheaders, class_thresholds["table projected row header"])
    supercells += subheaders
    # Align before NMS for supercells because alignment brings them into agreement
    # with rows and columns first; if supercells still overlap after this operation,
    # the threshold for NMS can basically be lowered to just above 0
    supercells = align_supercells(supercells, rows, columns)
    supercells = nms_supercells(supercells)

    header_supercell_tree(supercells)

    table_structures["columns"] = columns
    table_structures["rows"] = rows
    table_structures["supercells"] = supercells
    table_structures["headers"] = headers

    return table_structures


def nms(
    objects: list[Object],
    match_criteria: str = "object2_overlap",
    match_threshold: float = 0.05,
    keep_higher: bool = True,
) -> list[Object]:
    """A customizable version of non-maxima suppression (NMS).

    Default behavior: If a lower-confidence object overlaps more than 5% of its area
    with a higher-confidence object, remove the lower-confidence object.

    objects: set of dicts; each object dict must have a 'bbox' and a 'score' field
    match_criteria: how to measure how much two objects "overlap"
    match_threshold: the cutoff for determining that overlap requires suppression of one object
    keep_higher: if True, keep the object with the higher metric; otherwise, keep the lower
    """
    if len(objects) == 0:
        return []

    objects = sort_objects_by_score(objects, reverse=keep_higher)

    num_objects = len(objects)
    suppression = [False for obj in objects]

    for object2_num in range(1, num_objects):
        object2_area = _bbox_area(objects[object2_num]["bbox"])
        for object1_num in range(object2_num):
            if not suppression[object1_num]:
                object1_area = _bbox_area(objects[object1_num]["bbox"])
                intersect_area = _bbox_area(
                    _bbox_intersect(objects[object1_num]["bbox"], objects[object2_num]["bbox"])
                )
                try:
                    if match_criteria == "object1_overlap":
                        metric = intersect_area / object1_area
                    elif match_criteria == "object2_overlap":
                        metric = intersect_area / object2_area
                    elif match_criteria == "iou":
                        metric = intersect_area / (object1_area + object2_area - intersect_area)
                    if metric >= match_threshold:
                        suppression[object2_num] = True
                        break
                except ZeroDivisionError:
                    pass

    return [obj for idx, obj in enumerate(objects) if not suppression[idx]]


def align_headers(headers: list[Object], rows: list[Object]) -> list[Object]:
    """Replace the header boxes with one header box covering the header rows, from the top.

    A row is a header row if a header box covers at least 50% of its height. The header always
    starts at row 0. If row 0 is a header row, the header extends through the header rows that
    follow it without a gap; otherwise it ends at the first header row, and later header rows are
    dropped. Only one header per table is supported. Marks each row's "header" flag, and returns
    a list with the header, or an empty list if there are no header rows.
    """

    aligned_headers: list[Object] = []

    for row in rows:
        row["header"] = False

    header_row_nums = []
    for header in headers:
        for row_num, row in enumerate(rows):
            row_height = row["bbox"][3] - row["bbox"][1]
            min_row_overlap = max(row["bbox"][1], header["bbox"][1])
            max_row_overlap = min(row["bbox"][3], header["bbox"][3])
            overlap_height = max_row_overlap - min_row_overlap
            if overlap_height / row_height >= 0.5:
                header_row_nums.append(row_num)

    if len(header_row_nums) == 0:
        return aligned_headers

    header_rect = None
    if header_row_nums[0] > 0:
        header_row_nums = list(range(header_row_nums[0] + 1)) + header_row_nums

    last_row_num = -1
    for row_num in header_row_nums:
        if row_num == last_row_num + 1:
            row = rows[row_num]
            row["header"] = True
            header_rect = _bbox_include(header_rect, row["bbox"])
            last_row_num = row_num
        else:
            # Break as soon as a non-header row is encountered.
            # This ignores any subsequent rows in the table labeled as a header.
            # Having more than 1 header is not supported currently.
            break

    header = {"bbox": header_rect or [0, 0, 0, 0]}
    aligned_headers.append(header)

    return aligned_headers


def align_supercells(
    supercells: list[Object], rows: list[Object], columns: list[Object]
) -> list[Object]:
    """Align each supercell to the rows and columns it covers, removing those that cover too few.

    A supercell covers a row if it overlaps at least 50% of the row's height, and a column if it
    overlaps at least 50% of the column's width. A supercell can't cross the header boundary, so
    the smaller group of rows (header or data) is dropped. Supercells that cover fewer than two
    grid cells are removed.
    """
    aligned_supercells = []

    for supercell in supercells:
        supercell["header"] = False
        row_bbox_rect = None
        col_bbox_rect = None
        intersecting_header_rows = set()
        intersecting_data_rows = set()
        for row_num, row in enumerate(rows):
            row_height = row["bbox"][3] - row["bbox"][1]
            supercell_height = supercell["bbox"][3] - supercell["bbox"][1]
            min_row_overlap = max(row["bbox"][1], supercell["bbox"][1])
            max_row_overlap = min(row["bbox"][3], supercell["bbox"][3])
            overlap_height = max_row_overlap - min_row_overlap
            if "span" in supercell:
                overlap_fraction = max(
                    overlap_height / row_height, overlap_height / supercell_height
                )
            else:
                overlap_fraction = overlap_height / row_height
            if overlap_fraction >= 0.5:
                if "header" in row and row["header"]:
                    intersecting_header_rows.add(row_num)
                else:
                    intersecting_data_rows.add(row_num)

        # Supercell cannot span across the header boundary; eliminate whichever
        # group of rows is the smallest
        supercell["header"] = False
        if len(intersecting_data_rows) > 0 and len(intersecting_header_rows) > 0:
            if len(intersecting_data_rows) > len(intersecting_header_rows):
                intersecting_header_rows = set()
            else:
                intersecting_data_rows = set()
        if len(intersecting_header_rows) > 0:
            supercell["header"] = True
        elif "span" in supercell:
            continue  # Require span supercell to be in the header
        intersecting_rows = intersecting_data_rows.union(intersecting_header_rows)
        # Determine vertical span of aligned supercell
        for row_num in intersecting_rows:
            row_bbox_rect = _bbox_include(row_bbox_rect, rows[row_num]["bbox"])
        if row_bbox_rect is None:
            continue

        intersecting_cols = []
        for col_num, col in enumerate(columns):
            col_width = col["bbox"][2] - col["bbox"][0]
            supercell_width = supercell["bbox"][2] - supercell["bbox"][0]
            min_col_overlap = max(col["bbox"][0], supercell["bbox"][0])
            max_col_overlap = min(col["bbox"][2], supercell["bbox"][2])
            overlap_width = max_col_overlap - min_col_overlap
            if "span" in supercell:
                overlap_fraction = max(overlap_width / col_width, overlap_width / supercell_width)
                # Multiply by 2 effectively lowers the threshold to 0.25
                if supercell["header"]:
                    overlap_fraction = overlap_fraction * 2
            else:
                overlap_fraction = overlap_width / col_width
            if overlap_fraction >= 0.5:
                intersecting_cols.append(col_num)
                col_bbox_rect = _bbox_include(col_bbox_rect, col["bbox"])
        if col_bbox_rect is None:
            continue

        supercell_bbox = _bbox_intersect(row_bbox_rect, col_bbox_rect)
        supercell["bbox"] = supercell_bbox

        # Only a true supercell if it joins across multiple rows or columns
        if (
            len(intersecting_rows) > 0
            and len(intersecting_cols) > 0
            and (len(intersecting_rows) > 1 or len(intersecting_cols) > 1)
        ):
            supercell["row_numbers"] = list(intersecting_rows)
            supercell["column_numbers"] = intersecting_cols
            aligned_supercells.append(supercell)

            # A span supercell in the header means there must be supercells above it in the header
            if "span" in supercell and supercell["header"] and len(supercell["column_numbers"]) > 1:
                for row_num in range(0, min(supercell["row_numbers"])):
                    new_supercell = {
                        "row_numbers": [row_num],
                        "column_numbers": supercell["column_numbers"],
                        "score": supercell["score"],
                        "propagated": True,
                    }
                    new_supercell_columns = [columns[idx] for idx in supercell["column_numbers"]]
                    new_supercell_rows = [rows[idx] for idx in supercell["row_numbers"]]
                    bbox = [
                        min([column["bbox"][0] for column in new_supercell_columns]),
                        min([row["bbox"][1] for row in new_supercell_rows]),
                        max([column["bbox"][2] for column in new_supercell_columns]),
                        max([row["bbox"][3] for row in new_supercell_rows]),
                    ]
                    new_supercell["bbox"] = bbox
                    aligned_supercells.append(new_supercell)

    return aligned_supercells


def nms_supercells(supercells: list[Object]) -> list[Object]:
    """Suppress overlapping supercells, shrinking them first to resolve the overlap.

    If two supercells overlap the same grid cell, the lower-confidence supercell is shrunk to
    resolve the overlap. If the shrunk supercell covers fewer than two grid cells, it is removed.
    """

    supercells = sort_objects_by_score(supercells)
    num_supercells = len(supercells)
    suppression = [False for supercell in supercells]

    for supercell2_num in range(1, num_supercells):
        supercell2 = supercells[supercell2_num]
        for supercell1_num in range(supercell2_num):
            supercell1 = supercells[supercell1_num]
            remove_supercell_overlap(supercell1, supercell2)
        if (
            (len(supercell2["row_numbers"]) < 2 and len(supercell2["column_numbers"]) < 2)
            or len(supercell2["row_numbers"]) == 0
            or len(supercell2["column_numbers"]) == 0
        ):
            suppression[supercell2_num] = True

    return [obj for idx, obj in enumerate(supercells) if not suppression[idx]]


def header_supercell_tree(supercells: list[Object]) -> None:
    """Remove, in place, header supercells that don't have exactly one parent in each row above.

    The cells in the header form a tree. A header supercell must be below exactly one header
    supercell covering its columns in every row above it; with more than one, some cell would have
    more than one parent. Any supercell that would violate this is eliminated.
    """
    header_supercells = [
        supercell for supercell in supercells if "header" in supercell and supercell["header"]
    ]
    header_supercells = sort_objects_by_score(header_supercells)

    for header_supercell in header_supercells[:]:
        ancestors_by_row: defaultdict[int, int] = defaultdict(int)
        min_row = min(header_supercell["row_numbers"])
        for header_supercell2 in header_supercells:
            max_row2 = max(header_supercell2["row_numbers"])
            if max_row2 < min_row:
                if set(header_supercell["column_numbers"]).issubset(
                    set(header_supercell2["column_numbers"])
                ):
                    for row2 in header_supercell2["row_numbers"]:
                        ancestors_by_row[row2] += 1
        for row in range(0, min_row):
            if not ancestors_by_row[row] == 1:
                supercells.remove(header_supercell)
                break


def table_structure_to_cells(
    table_structures: dict[str, list[Object]], table_spans: list[Object]
) -> tuple[list[Object], float]:
    """Convert refined, consistent table structures into table cells with their text.

    Each row and column intersection is a cell, unless more than half of it is inside a
    supercell; each supercell becomes one cell covering those grid cells. A cell is a header cell
    if it is in a header row (for a supercell, if all its grid cells are). The words in
    table_spans are slotted into the cells to give each its text, and the row, column and cell
    boxes are adjusted to fit the text. Returns (cells, confidence_score), where the score
    measures how well the words fit into the cells.
    """
    columns = table_structures["columns"]
    rows = table_structures["rows"]
    supercells = table_structures["supercells"]
    cells: list[Object] = []
    subcells: list[Object] = []

    # Identify complete cells and subcells
    for column_num, column in enumerate(columns):
        for row_num, row in enumerate(rows):
            cell_rect = _bbox_intersect(row["bbox"], column["bbox"])
            header = "header" in row and row["header"]
            cell: Object = {
                "bbox": list(cell_rect),
                "column_nums": [column_num],
                "row_nums": [row_num],
                "header": header,
            }

            cell["subcell"] = False
            cell_rect_area = _bbox_area(cell_rect)
            if cell_rect_area > 0:
                for supercell in supercells:
                    if (
                        _bbox_area(_bbox_intersect(supercell["bbox"], cell_rect)) / cell_rect_area
                    ) > 0.5:
                        cell["subcell"] = True
                        break

            if cell["subcell"]:
                subcells.append(cell)
            else:
                cell["subheader"] = False
                cells.append(cell)

    for supercell in supercells:
        cell_columns: set[int] = set()
        cell_rows: set[int] = set()
        supercell_rect: BBox | None = None
        header = True
        for subcell in subcells:
            subcell_rect_area = _bbox_area(subcell["bbox"])
            if (
                subcell_rect_area > 0
                and (
                    _bbox_area(_bbox_intersect(subcell["bbox"], supercell["bbox"]))
                    / subcell_rect_area
                )
                > 0.5
            ):
                supercell_rect = _bbox_include(supercell_rect, subcell["bbox"])
                cell_rows = cell_rows.union(set(subcell["row_nums"]))
                cell_columns = cell_columns.union(set(subcell["column_nums"]))
                # By convention here, all subcells must be classified
                # as header cells for a supercell to be classified as a header cell;
                # otherwise, this could lead to a non-rectangular header region
                header = header and "header" in subcell and subcell["header"]
        if supercell_rect is not None and len(cell_rows) > 0 and len(cell_columns) > 0:
            cell = {
                "bbox": list(supercell_rect),
                "column_nums": list(cell_columns),
                "row_nums": list(cell_rows),
                "header": header,
                "subheader": supercell["subheader"],
            }
            cells.append(cell)

    # Compute a confidence score based on how well the page tokens
    # slot into the cells reported by the model
    _, _, cell_match_scores = slot_into_containers(cells, table_spans)
    try:
        mean_match_score = sum(cell_match_scores) / len(cell_match_scores)
        min_match_score = min(cell_match_scores)
        confidence_score = (mean_match_score + min_match_score) / 2
    except ZeroDivisionError:
        confidence_score = 0

    # Dilate rows and columns before final extraction
    dilated_columns = columns
    dilated_rows = rows
    for cell in cells:
        column_rect: BBox | None = None
        for column_num in cell["column_nums"]:
            column_rect = _bbox_include(column_rect, dilated_columns[column_num]["bbox"])
        row_rect: BBox | None = None
        for row_num in cell["row_nums"]:
            row_rect = _bbox_include(row_rect, dilated_rows[row_num]["bbox"])
        # Every cell has at least one row and one column.
        assert column_rect is not None and row_rect is not None
        cell["bbox"] = _bbox_intersect(column_rect, row_rect)

    span_nums_by_cell, _, _ = slot_into_containers(
        cells, table_spans, overlap_threshold=0.001, unique_assignment=True, forced_assignment=False
    )

    for cell, cell_span_nums in zip(cells, span_nums_by_cell):
        cell_spans = [table_spans[num] for num in cell_span_nums]
        # TODO: Refine how text is extracted; should be character-based, not span-based;
        # but need to associate
        cell["cell_text"] = extract_text_from_spans(cell_spans)
        cell["spans"] = cell_spans

    # Adjust the row, column, and cell bounding boxes to reflect the extracted text
    num_rows = len(rows)
    rows = sort_objects_top_to_bottom(rows)
    num_columns = len(columns)
    columns = sort_objects_left_to_right(columns)
    min_y_values_by_row = defaultdict(list)
    max_y_values_by_row = defaultdict(list)
    min_x_values_by_column = defaultdict(list)
    max_x_values_by_column = defaultdict(list)
    for cell in cells:
        min_row = min(cell["row_nums"])
        max_row = max(cell["row_nums"])
        min_column = min(cell["column_nums"])
        max_column = max(cell["column_nums"])
        for span in cell["spans"]:
            min_x_values_by_column[min_column].append(span["bbox"][0])
            min_y_values_by_row[min_row].append(span["bbox"][1])
            max_x_values_by_column[max_column].append(span["bbox"][2])
            max_y_values_by_row[max_row].append(span["bbox"][3])
    for row_num, row in enumerate(rows):
        if len(min_x_values_by_column[0]) > 0:
            row["bbox"][0] = min(min_x_values_by_column[0])
        if len(min_y_values_by_row[row_num]) > 0:
            row["bbox"][1] = min(min_y_values_by_row[row_num])
        if len(max_x_values_by_column[num_columns - 1]) > 0:
            row["bbox"][2] = max(max_x_values_by_column[num_columns - 1])
        if len(max_y_values_by_row[row_num]) > 0:
            row["bbox"][3] = max(max_y_values_by_row[row_num])
    for column_num, column in enumerate(columns):
        if len(min_x_values_by_column[column_num]) > 0:
            column["bbox"][0] = min(min_x_values_by_column[column_num])
        if len(min_y_values_by_row[0]) > 0:
            column["bbox"][1] = min(min_y_values_by_row[0])
        if len(max_x_values_by_column[column_num]) > 0:
            column["bbox"][2] = max(max_x_values_by_column[column_num])
        if len(max_y_values_by_row[num_rows - 1]) > 0:
            column["bbox"][3] = max(max_y_values_by_row[num_rows - 1])
    for cell in cells:
        row_rect = None
        column_rect = None
        for row_num in cell["row_nums"]:
            row_rect = _bbox_include(row_rect, rows[row_num]["bbox"])
        for column_num in cell["column_nums"]:
            column_rect = _bbox_include(column_rect, columns[column_num]["bbox"])
        if row_rect and column_rect:
            cell_rect = _bbox_intersect(row_rect, column_rect)
            if _bbox_area(cell_rect) > 0:
                cell["bbox"] = cell_rect

    return cells, confidence_score


def remove_supercell_overlap(supercell1: Object, supercell2: Object) -> None:
    """Shrink supercell2, in place, until it no longer overlaps supercell1.

    Supercells must be disjoint, and supercell2 should be the less confident of the two. Each step
    removes one overlapping row or column at an edge of supercell2: a column if supercell2 has
    fewer rows than columns (a column then holds fewer grid cells than a row), and otherwise a
    row. If neither edge row (or column) overlaps, supercell2 loses all its rows (or columns).
    """
    common_rows = set(supercell1["row_numbers"]).intersection(set(supercell2["row_numbers"]))
    common_columns = set(supercell1["column_numbers"]).intersection(
        set(supercell2["column_numbers"])
    )

    # While the supercells have overlapping grid cells, continue shrinking the less-confident
    # supercell one row or one column at a time
    while len(common_rows) > 0 and len(common_columns) > 0:
        # Try to shrink the supercell as little as possible to remove the overlap;
        # if the supercell has fewer rows than columns, remove an overlapping column,
        # because this removes fewer grid cells from the supercell;
        # otherwise remove an overlapping row
        if len(supercell2["row_numbers"]) < len(supercell2["column_numbers"]):
            min_column = min(supercell2["column_numbers"])
            max_column = max(supercell2["column_numbers"])
            if max_column in common_columns:
                common_columns.remove(max_column)
                supercell2["column_numbers"].remove(max_column)
            elif min_column in common_columns:
                common_columns.remove(min_column)
                supercell2["column_numbers"].remove(min_column)
            else:
                supercell2["column_numbers"] = []
                common_columns = set()
        else:
            min_row = min(supercell2["row_numbers"])
            max_row = max(supercell2["row_numbers"])
            if max_row in common_rows:
                common_rows.remove(max_row)
                supercell2["row_numbers"].remove(max_row)
            elif min_row in common_rows:
                common_rows.remove(min_row)
                supercell2["row_numbers"].remove(min_row)
            else:
                supercell2["row_numbers"] = []
                common_rows = set()


def rotate_objects_90(img_size: Sequence[float], objs: list[Object]) -> None:
    """Rotate the objects' boxes 90 degrees clockwise with the page, in place.

    Also removes " rotated" from their labels. img_size is the page's (width, height).
    """
    for obj in objs:
        if "label" in obj:
            obj["label"] = obj["label"].replace(" rotated", "")
        obj["bbox"] = [
            img_size[1] - obj["bbox"][3],
            obj["bbox"][0],
            img_size[1] - obj["bbox"][1],
            obj["bbox"][2],
        ]


def rotate_objects_270(img_size: Sequence[float], objs: list[Object]) -> None:
    """Rotate the objects' boxes 90 degrees counterclockwise, in place, undoing rotate_objects_90.

    img_size is the (width, height) of the rotated page, which is the original page's
    (height, width). Labels are not changed.
    """
    for obj in objs:
        obj["bbox"] = [
            obj["bbox"][1],
            img_size[0] - obj["bbox"][2],
            obj["bbox"][3],
            img_size[0] - obj["bbox"][0],
        ]


def potatr_outputs_to_objects(
    pred_logits: torch.Tensor,
    pred_bboxes: torch.Tensor,
    img_size: Sequence[int],
    class_idx2name: list[str] = class_idx2name,
) -> list[Object]:
    """Convert one image's model outputs to a list of detected objects.

    The objects are the queries whose most likely class isn't "no object" and has probability
    above 0.5. Each has the query index ("idx"), "label", "score" and "bbox", in pixels
    (img_size is (width, height)).
    """
    m = pred_logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())
    pred_scores = list(m.values.detach().cpu().numpy())
    bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    idx = 0
    for label, score, bbox in zip(pred_labels, pred_scores, bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == "no object" and score > 0.5:
            objects.append(
                {
                    "idx": idx,
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )
        idx += 1

    return objects


def potatr_objects_to_table_structures(
    objects_in_table: list[Object],
    tokens_in_table: list[Object],
    class_thresholds: dict[str, float],
) -> dict[str, list[Object]]:
    """Turn the objects detected in a table into a consistent set of table structures.

    The structures are rows, columns, headers and supercells. This entails resolving
    conflicts/overlaps, and ensuring the boxes meet certain alignment conditions (for example:
    rows should all have the same width, etc.). Headers and supercells are refined only if the
    table has at least one row and two columns.
    """

    table_structures = {}

    columns = [obj for obj in objects_in_table if obj["label"] == "table column"]
    rows = [obj for obj in objects_in_table if obj["label"] == "table row"]
    headers = [obj for obj in objects_in_table if obj["label"] == "table column header"]
    supercells = [obj for obj in objects_in_table if obj["label"] == "table spanning cell"]
    for obj in supercells:
        obj["subheader"] = False
    subheaders = [obj for obj in objects_in_table if obj["label"] == "table projected row header"]
    for obj in subheaders:
        obj["subheader"] = True
    supercells += subheaders
    for obj in rows:
        obj["header"] = False
        for header_obj in headers:
            if iob(obj["bbox"], header_obj["bbox"]) >= 0.5:
                obj["header"] = True

    # Refine table structures
    rows = refine_rows(rows, tokens_in_table)
    columns = refine_columns(columns, tokens_in_table)

    # Shrink table bbox to just the total height of the rows
    # and the total width of the columns
    row_rect = None
    for obj in rows:
        row_rect = _bbox_include(row_rect, obj["bbox"])
    column_rect = None
    for obj in columns:
        column_rect = _bbox_include(column_rect, obj["bbox"])
    row_bbox = row_rect or [0, 0, 0, 0]
    col_bbox = column_rect or [0, 0, 0, 0]
    row_column_bbox = [col_bbox[0], row_bbox[1], col_bbox[2], row_bbox[3]]

    # Process the rows and columns into a complete segmented table
    columns = align_columns(columns, row_column_bbox)
    rows = align_rows(rows, row_column_bbox)

    table_structures["rows"] = rows
    table_structures["columns"] = columns
    table_structures["headers"] = headers
    table_structures["supercells"] = supercells

    if len(rows) > 0 and len(columns) > 1:
        table_structures = refine_table_structures(table_structures, class_thresholds)

    return table_structures


def potatr_objects_to_cells(
    objects_in_table: list[Object],
    tokens_in_table: list[Object],
    class_thresholds: dict[str, float],
) -> tuple[dict[str, list[Object]], list[Object], float]:
    """Convert the objects detected in a table, and the words inside it, into table cells.

    Returns (table_structures, cells, confidence_score). The score is based on how well the text
    was able to be uniquely slotted into the cells. A table with no rows or no columns gets no
    cells and a score of 0.
    """

    table_structures = potatr_objects_to_table_structures(
        objects_in_table, tokens_in_table, class_thresholds
    )

    # Check for a valid table
    if len(table_structures["columns"]) < 1 or len(table_structures["rows"]) < 1:
        cells: list[Object] = []
        confidence_score = 0.0
    else:
        cells, confidence_score = table_structure_to_cells(table_structures, tokens_in_table)

    return table_structures, cells, confidence_score


def potatr_outputs_to_tables_cells(
    pred_logits: torch.Tensor,
    pred_bboxes: torch.Tensor,
    tokens: list[Object],
    img_size: Sequence[int],
    class_idx2name: list[str] = class_idx2name,
    class_thresholds: dict[str, float] = class_thresholds,
) -> list[list[Object]]:
    """Convert one image's model outputs, and the page's words, into the cells of each table.

    tokens are the page's words, with "bbox" in pixels, and img_size is (width, height). Each
    detected table ("table" or "table rotated") gets the objects and words at least half inside
    it. A rotated table's objects and words are rotated 90 degrees clockwise first, and its cells
    rotated back to page coordinates; tokens itself is not changed. Returns a list with each
    table's cells.
    """
    # Objects can be rotated or not rotated
    objects = potatr_outputs_to_objects(pred_logits, pred_bboxes, img_size, class_idx2name)

    # Highest-level table objects
    table_class_objects = [
        obj for obj in objects if obj["label"] == "table" or obj["label"] == "table rotated"
    ]

    tables_cells = []
    for pred_table in table_class_objects:
        objects_in_table = [obj for obj in objects if iob(obj["bbox"], pred_table["bbox"]) >= 0.5]

        rotated = pred_table["label"] == "table rotated"

        table_bbox = pred_table["bbox"]
        table_tokens = tokens
        if rotated:
            # Rotate copies, so the page's objects and words keep their page coordinates, both
            # for the caller and for the page's other tables.
            rotated_table = [dict(pred_table)]
            objects_in_table = [dict(obj) for obj in objects_in_table]
            table_tokens = [dict(token) for token in tokens]
            for objs in (rotated_table, objects_in_table, table_tokens):
                rotate_objects_90(img_size, objs)
            table_bbox = rotated_table[0]["bbox"]

        tokens_in_table = [token for token in table_tokens if iob(token["bbox"], table_bbox) >= 0.5]

        # Determine the table cell structure from the objects
        table_structures, cells, confidence_score = potatr_objects_to_cells(
            objects_in_table, tokens_in_table, class_thresholds
        )

        if rotated:
            rotate_objects_270(img_size[::-1], cells)

        tables_cells.append(cells)

    return tables_cells
