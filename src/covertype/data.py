"""Loading Covertype, and the two things its column layout hides.

581,012 rows, 54 columns, 7 classes. The columns are 10 real-valued terrain
measurements plus 44 one-hot indicator columns (4 wilderness areas, 40 soil
types). No missing values.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

TARGET = "Cover_Type"

CLASS_NAMES: dict[int, str] = {
    1: "spruce/fir",
    2: "lodgepole pine",
    3: "ponderosa pine",
    4: "cottonwood/willow",
    5: "aspen",
    6: "douglas-fir",
    7: "krummholz",
}

CONTINUOUS_COLUMNS: tuple[str, ...] = (
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
)

ONE_HOT_PREFIXES: tuple[str, ...] = ("Wilderness_Area", "Soil_Type")


class DataError(ValueError):
    """The file is not the Covertype dataset this experiment expects."""


@dataclass(frozen=True, slots=True)
class Dataset:
    columns: tuple[str, ...]
    x: tuple[tuple[float, ...], ...]
    y: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return len(self.columns)

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = dict.fromkeys(CLASS_NAMES, 0)
        for label in self.y:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def imbalance_ratio(self) -> float:
        """Largest class over smallest present class.

        On the full dataset this is roughly 100:1, which is why a single
        accuracy figure cannot carry the result.
        """
        present = [n for n in self.class_counts().values() if n]
        return max(present) / min(present) if present else float("nan")

    def subset(self, indices: Sequence[int]) -> Dataset:
        return Dataset(
            self.columns,
            tuple(self.x[i] for i in indices),
            tuple(self.y[i] for i in indices),
        )

    def select(self, columns: Sequence[str]) -> Dataset:
        keep = [self.columns.index(c) for c in columns]
        return Dataset(
            tuple(columns),
            tuple(tuple(row[i] for i in keep) for row in self.x),
            self.y,
        )


def parse_csv(text: str, *, target: str = TARGET) -> Dataset:
    reader = csv.DictReader(text.splitlines())
    fieldnames = [f for f in (reader.fieldnames or []) if f]
    if target not in fieldnames:
        raise DataError(f"no {target!r} column; found {len(fieldnames)} columns")

    feature_columns = tuple(c for c in fieldnames if c != target)
    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    for record in reader:
        raw_label = (record.get(target) or "").strip()
        if not raw_label:
            continue
        try:
            labels.append(int(float(raw_label)))
            rows.append(tuple(float(record[c]) for c in feature_columns))
        except (TypeError, ValueError) as exc:
            raise DataError(f"unparseable row: {record}") from exc

    if not rows:
        raise DataError("no usable rows")
    return Dataset(feature_columns, tuple(rows), tuple(labels))


def load_csv(path: str | Path, **kwargs) -> Dataset:
    return parse_csv(Path(path).read_text(encoding="utf-8"), **kwargs)


def load_uci(dataset_id: int = 31) -> Dataset:
    """Fetch through `ucimlrepo`. Optional: it downloads on call and needs the network."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError(
            "ucimlrepo is not installed; run `pip install -r requirements.txt` "
            "or pass a CSV path. See the README."
        ) from exc

    fetched = fetch_ucirepo(id=dataset_id)
    frame = fetched.data.features.copy()
    frame[TARGET] = fetched.data.targets.iloc[:, 0].values
    return parse_csv(frame.to_csv(index=False))


def invert_one_hot(dataset: Dataset, prefix: str) -> list[int]:
    """Collapse a one-hot block back to a single integer code.

    Useful for a contingency table and **wrong as a model feature**. Soil type
    17 is not "between" 16 and 18, and a model given the collapsed column will
    split on an ordering that does not exist. It is also why ranking features by
    Pearson correlation against `Cover_Type` — a nominal 7-class label — produces
    numbers with no interpretation: the correlation depends entirely on the
    arbitrary integer codes assigned to the categories.
    """
    indices = [i for i, c in enumerate(dataset.columns) if c.startswith(prefix)]
    if not indices:
        raise DataError(f"no columns starting with {prefix!r}")

    codes = []
    for row in dataset.x:
        active = [i for i in indices if row[i]]
        if len(active) != 1:
            # A row with no active indicator, or two, means the block is not a
            # one-hot encoding and collapsing it would invent a category.
            raise DataError(
                f"{prefix}: expected exactly one active indicator per row, "
                f"found {len(active)}"
            )
        codes.append(indices.index(active[0]) + 1)
    return codes


def continuous_only(dataset: Dataset) -> Dataset:
    present = [c for c in CONTINUOUS_COLUMNS if c in dataset.columns]
    if not present:
        raise DataError("no continuous columns found")
    return dataset.select(present)


def iter_rows(dataset: Dataset) -> Iterator[tuple[tuple[float, ...], int]]:
    yield from zip(dataset.x, dataset.y, strict=True)
