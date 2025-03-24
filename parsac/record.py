import sqlite3
import os
from typing import Any, Mapping, Union, Optional, Iterable, Sequence
from pathlib import Path
import json

import numpy as np


def _get_sqlite_type(value: Any) -> str:
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    elif isinstance(value, str):
        return "TEXT"
    else:
        return "BLOB"


class Recorder:
    def __init__(self, file: Union[os.PathLike[Any], str]):
        self.file = Path(file)
        self.conn = sqlite3.connect(self.file, detect_types=sqlite3.PARSE_DECLTYPES)
        self.run_id: Optional[int] = None

    def start(
        self, parameters: Mapping[str, Any], output: Mapping[str, Any], info: Any
    ) -> None:
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS "runs" (
            id INTEGER PRIMARY KEY,
            time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            info TEXT
            )
        """
        )

        cursor = self.conn.execute(
            "INSERT INTO runs (info) VALUES (?)", (json.dumps(info),)
        )
        self.run_id = cursor.lastrowid
        assert self.run_id is not None

        pcolumns = [
            f'"{k}" {_get_sqlite_type(v)} NOT NULL' for k, v in parameters.items()
        ]
        ocolumns = [f'"{k}" {_get_sqlite_type(v)}' for k, v in output.items()]
        combined_columns = ", ".join(pcolumns + ocolumns)
        self.conn.execute(
            f"""CREATE TABLE IF NOT EXISTS "results" (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            {combined_columns}
            )
        """
        )

        self.conn.commit()

    def record(self, parameters: Mapping[str, Any], output: Mapping[str, Any]) -> None:
        if self.run_id is None:
            return
        items = {"run_id": self.run_id, **parameters, **output}
        self.conn.execute(
            f"""
            INSERT INTO results
            ({", ".join(f'"{k}"' for k in items)})
            VALUES ({", ".join('?' for _ in items)})
        """,
            list(items.values()),
        )
        self.conn.commit()

    @property
    def headers(self) -> Sequence[str]:
        """Column headers for the results table."""
        cursor = self.conn.execute("PRAGMA table_info(results)")
        return [info[1] for info in cursor]

    @property
    def rows(self) -> Iterable[Iterable[Any]]:
        """Iterate over the rows of the results table."""
        cursor = self.conn.execute("SELECT * FROM results")
        for row in cursor:
            yield row

    @property
    def run_info(self) -> Mapping[int, Any]:
        cursor = self.conn.execute("SELECT * FROM runs")
        result = {}
        for row in cursor:
            result[row[0]] = dict(info=json.loads(row[2]))
        return result

    def to_ndarray(self) -> np.ndarray:
        """Convert the results table to a numpy array."""
        return np.array(list(self.rows), dtype=float)

    def to_text(self, file: Union[os.PathLike[Any], str], sep: str = "\t") -> None:
        """Write the results table to a text file.

        Args:
            file: The file to write to.
            sep: The separator between columns.
        """
        with open(file, "w") as f:
            f.write(sep.join(self.headers) + "\n")
            for row in self.rows:
                f.write(sep.join(map(str, row)) + "\n")

    def close(self) -> None:
        self.conn.close()
