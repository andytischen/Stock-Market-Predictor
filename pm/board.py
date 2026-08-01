"""Task storage: a JSON file holding the project board."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

DEFAULT_BOARD = Path("project.json")

STATUSES = ("todo", "doing", "blocked", "done")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Task:
    id: int
    title: str
    status: str = "todo"
    owner: str = ""
    due: str = ""
    resources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @property
    def is_overdue(self) -> bool:
        if not self.due or self.status == "done":
            return False
        return date.fromisoformat(self.due) < datetime.now(timezone.utc).date()

    def touch(self) -> None:
        self.updated_at = _now()


class Board:
    """A collection of tasks persisted as JSON."""

    def __init__(self, name: str = "project", tasks: list[Task] | None = None) -> None:
        self.name = name
        self.tasks = tasks or []

    @classmethod
    def load(cls, path: Path) -> Board:
        if not path.exists():
            return cls(name=path.stem)
        payload = json.loads(path.read_text())
        return cls(
            name=payload.get("name", path.stem),
            tasks=[Task(**t) for t in payload.get("tasks", [])],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": self.name, "tasks": [asdict(t) for t in self.tasks]}
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def add(
        self,
        title: str,
        owner: str = "",
        due: str = "",
        resources: list[str] | None = None,
    ) -> Task:
        if not title.strip():
            raise ValueError("a task needs a title")
        if due:
            date.fromisoformat(due)  # validate, raises ValueError
        task = Task(
            id=max((t.id for t in self.tasks), default=0) + 1,
            title=title.strip(),
            owner=owner,
            due=due,
            resources=list(resources or []),
        )
        self.tasks.append(task)
        return task

    def get(self, task_id: int) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"no task with id {task_id}")

    def set_status(self, task_id: int, status: str) -> Task:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; choose from {', '.join(STATUSES)}")
        task = self.get(task_id)
        task.status = status
        task.touch()
        return task

    def add_note(self, task_id: int, note: str) -> Task:
        task = self.get(task_id)
        task.notes.append(f"{datetime.now(timezone.utc).date().isoformat()}: {note}")
        task.touch()
        return task

    def add_resource(self, task_id: int, resource: str) -> Task:
        task = self.get(task_id)
        if resource not in task.resources:
            task.resources.append(resource)
        task.touch()
        return task

    def by_status(self, status: str) -> list[Task]:
        return [t for t in self.tasks if t.status == status]

    @property
    def progress(self) -> float:
        if not self.tasks:
            return 0.0
        return len(self.by_status("done")) / len(self.tasks)

    def resources_needed(self) -> dict[str, list[Task]]:
        """Requested resources mapped to the still-open tasks that need them."""
        wanted: dict[str, list[Task]] = {}
        for task in self.tasks:
            if task.status == "done":
                continue
            for resource in task.resources:
                wanted.setdefault(resource, []).append(task)
        return wanted
