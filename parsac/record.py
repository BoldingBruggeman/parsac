import sqlite3
import os
from typing import Any, Mapping, Union, Optional, Iterable, Sequence
from pathlib import Path
import pickle

import numpy as np


def _get_sqlite_type(value: Any) -> str:
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    elif isinstance(value, str):
        return "TEXT"
    else:
        raise NotImplementedError(f"Unsupported type {type(value)}")


class Recorder:
    def __init__(self, file: Union[os.PathLike[Any], str]) -> None:
        self.file = Path(file)
        self.create = not self.file.is_file()
        self.conn = sqlite3.connect(self.file, detect_types=sqlite3.PARSE_DECLTYPES)
        self.run_id: Optional[int] = None

    def start(
        self,
        config: Mapping[str, Any],
        required: Mapping[str, Any],
        optional: Mapping[str, Any],
    ) -> None:
        if self.create:
            columns = [f'"{k}" BLOB' for k in config]
            self.conn.execute(f'CREATE TABLE "config" ({", ".join(columns)})')
            keys = ", ".join(f'"{k}"' for k in config)
            values = [pickle.dumps(v) for v in config.values()]
            qs = ", ".join("?" for _ in values)
            self.conn.execute(f"INSERT INTO config ({keys}) VALUES ({qs})", values)

            self.conn.execute(
                """CREATE TABLE "runs" (
                id INTEGER PRIMARY KEY,
                time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            req_columns = [
                f'"{k}" {_get_sqlite_type(v)} NOT NULL' for k, v in required.items()
            ]
            opt_columns = [f'"{k}" {_get_sqlite_type(v)}' for k, v in optional.items()]
            combined_columns = ", ".join(req_columns + opt_columns)
            self.conn.execute(
                f"""CREATE TABLE "results" (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                exception TEXT,
                {combined_columns}
                )
            """
            )

        cursor = self.conn.execute("INSERT INTO runs DEFAULT VALUES")
        self.run_id = cursor.lastrowid
        assert self.run_id is not None

        self.conn.commit()

    def record(self, exception: Optional[Exception], **fields: Any) -> None:
        if self.run_id is None:
            return
        if exception is not None:
            fields["exception"] = repr(exception)
        self.conn.execute(
            f"""
            INSERT INTO results
            ("run_id"{''.join(f', "{k}"' for k in fields)})
            VALUES (?{', ?' * len(fields)})
        """,
            [self.run_id] + list(fields.values()),
        )
        self.conn.commit()

    @property
    def headers(self) -> Sequence[str]:
        """Column headers for the results table."""
        cursor = self.conn.execute("PRAGMA table_info(results)")
        return [info[1] for info in cursor]

    def rows(self, where: str = "") -> Iterable[Iterable[Any]]:
        """Iterate over the rows of the results table."""
        cursor = self.conn.execute(f"SELECT * FROM results {where}")
        for row in cursor:
            yield row

    @property
    def config(self) -> Mapping[str, Any]:
        """Global experiment configuration."""
        cursor = self.conn.execute("SELECT * FROM config")
        row = cursor.fetchone()
        return {d[0]: pickle.loads(v) for d, v in zip(cursor.description, row)}

    def to_ndarray(self, where: str = "") -> np.ndarray:
        """Convert the results table to a numpy array."""
        return np.array(list(self.rows(where)), dtype=float)

    def to_text(self, file: Union[os.PathLike[Any], str], sep: str = "\t") -> None:
        """Write the results table to a text file.

        Args:
            file: The file to write to.
            sep: The separator between columns.
        """
        with open(file, "w") as f:
            f.write(sep.join(self.headers) + "\n")
            for row in self.rows():
                values = ["" if v is None else str(v) for v in row]
                f.write(sep.join(values) + "\n")

    def close(self) -> None:
        self.conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump")
    parser.add_argument(
        "db_file", help="SQLite database file with optimization results"
    )
    args = parser.parse_args()
    rec = Recorder(args.db_file)
    print(f"{rec.file} contains {len(list(rec.rows()))} rows.")
    if args.dump:
        rec.to_text(args.dump, sep="\t")
        print(f"Dumped results table to {args.dump}.")
