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


class Record:
    def __init__(
        self, conn: Optional[sqlite3.Connection], run_id: Optional[int], **fields: Any
    ):
        self.conn = conn
        self.run_id = run_id
        self.fields = fields

    def __enter__(self) -> "Record":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.run_id is None:
            return
        assert self.conn is not None
        if exc_value is not None:
            self.fields["exception"] = repr(exc_value)
        self.conn.execute(
            f"""
            INSERT INTO results
            ("run_id"{''.join(f', "{k}"' for k in self.fields)})
            VALUES (?{', ?' * len(self.fields)})
        """,
            [self.run_id] + list(self.fields.values()),
        )
        self.conn.commit()

    def update(self, **fields: Any):
        self.fields.update(fields)


class Recorder:
    def __init__(
        self, file: Union[os.PathLike[Any], str], read_only: bool = False
    ) -> None:
        self.file = Path(file)
        self.conn: Optional[sqlite3.Connection] = None
        self.mode = "ro" if read_only else "rwc"
        if read_only:
            self.connect()
        self.run_id: Optional[int] = None

    def connect(self) -> sqlite3.Connection:
        uri = f"file:{self.file}?mode={self.mode}"
        self.conn = sqlite3.connect(uri, detect_types=sqlite3.PARSE_DECLTYPES, uri=True)
        return self.conn

    def start(
        self,
        config: Mapping[str, Any],
        required: Mapping[str, Any],
        optional: Mapping[str, Any],
    ) -> None:
        create = not self.file.exists()
        conn = self.connect()
        if create:
            columns = [f'"{k}" BLOB' for k in config]
            conn.execute(f'CREATE TABLE "config" ({", ".join(columns)})')
            keys = ", ".join(f'"{k}"' for k in config)
            values = [pickle.dumps(v) for v in config.values()]
            qs = ", ".join("?" for _ in values)
            conn.execute(f"INSERT INTO config ({keys}) VALUES ({qs})", values)

            conn.execute(
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
            conn.execute(
                f"""CREATE TABLE "results" (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                exception TEXT,
                {combined_columns}
                )
            """
            )
        else:
            found_columns = frozenset(self.headers)
            expected_columns = set(required)
            expected_columns |= frozenset(optional)
            expected_columns |= frozenset({"exception", "id", "run_id"})
            if found_columns != expected_columns:
                missing = expected_columns - found_columns
                unexpected = found_columns - expected_columns
                error = "Result columns have changed."
                if missing:
                    error += f" Missing columns: {', '.join(missing)}."
                if unexpected:
                    error += f" Unexpected columns: {', '.join(unexpected)}."
                raise Exception(
                    f"{error} To continue anyway, delete database {self.file} first."
                )
        cursor = conn.execute("INSERT INTO runs DEFAULT VALUES")
        self.run_id = cursor.lastrowid
        assert self.run_id is not None

        conn.commit()

    def record(self, **fields: Any) -> Record:
        return Record(self.conn, self.run_id, **fields)

    @property
    def headers(self) -> Sequence[str]:
        """Column headers for the results table."""
        assert self.conn is not None
        cursor = self.conn.execute("PRAGMA table_info(results)")
        return [info[1] for info in cursor]

    def rows(self, where: str = "") -> Iterable[Sequence[Any]]:
        assert self.conn is not None
        """Iterate over the rows of the results table."""
        cursor = self.conn.execute(f"SELECT * FROM results {where}")
        for row in cursor:
            yield row

    def get_config(self, key: str, default: Any = None) -> Any:
        """Global experiment configuration."""
        assert self.conn is not None
        cursor = self.conn.execute(f"SELECT {key} FROM config")
        row = cursor.fetchone()
        return pickle.loads(row[0])

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
        assert self.conn is not None
        self.conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump")
    parser.add_argument(
        "db_file", help="SQLite database file with optimization results"
    )
    args = parser.parse_args()
    rec = Recorder(args.db_file, read_only=True)
    print(f"{rec.file} contains {len(list(rec.rows()))} rows.")
    if args.dump:
        rec.to_text(args.dump, sep="\t")
        print(f"Dumped results table to {args.dump}.")
