import hashlib
import json
import threading
from typing import Dict, Any, Optional, Tuple

class TaskStore:
    def __init__(self):
        self._lock = threading.Lock()
        # principal -> { task_id -> task_dict }
        self.tasks_by_user: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # (principal, idempotency_hash) -> task_id
        self.idempotency_map: Dict[Tuple[str, str], str] = {}
        # task_id -> principal
        self.task_owner: Dict[str, str] = {}

    @staticmethod
    def compute_message_hash(message: dict) -> str:
        # Ignore configuration, normalize and hash message only recursively key-sorted
        clean_msg = {k: v for k, v in message.items() if k != "configuration"}
        canonical_str = json.dumps(clean_msg, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

    def get_task_by_id(self, principal: str, task_id: str) -> Optional[dict]:
        with self._lock:
            owner = self.task_owner.get(task_id)
            if not owner or owner != principal:
                return None
            return self.tasks_by_user.get(principal, {}).get(task_id)

    def list_tasks(self, principal: str) -> list:
        with self._lock:
            user_tasks = self.tasks_by_user.get(principal, {})
            return list(user_tasks.values())

    def save_task_idempotent(self, principal: str, message: dict, create_task_fn) -> Tuple[dict, bool]:
        msg_hash = self.compute_message_hash(message)
        with self._lock:
            if principal not in self.tasks_by_user:
                self.tasks_by_user[principal] = {}

            key = (principal, msg_hash)
            if key in self.idempotency_map:
                existing_task_id = self.idempotency_map[key]
                return self.tasks_by_user[principal][existing_task_id], True

            # Create new task via callback
            new_task = create_task_fn()
            task_id = new_task["id"]
            
            self.tasks_by_user[principal][task_id] = new_task
            self.idempotency_map[key] = task_id
            self.task_owner[task_id] = principal
            return new_task, False

    def update_task(self, principal: str, task_id: str, updater_fn) -> Optional[dict]:
        with self._lock:
            owner = self.task_owner.get(task_id)
            if not owner or owner != principal:
                return None
            task = self.tasks_by_user[principal].get(task_id)
            if not task:
                return None
            updated_task = updater_fn(task)
            self.tasks_by_user[principal][task_id] = updated_task
            return updated_task
