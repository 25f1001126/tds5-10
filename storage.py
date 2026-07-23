import hashlib
import json
import threading
from typing import Dict, Any, Optional, Tuple

class TaskStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.tasks_by_user: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.idempotency_map: Dict[Tuple[str, str], str] = {}
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

    def save_task_idempotent(self, principal: str, message: dict, create_task_fn) -> Tuple[dict, bool, bool]:
        msg_hash = self.compute_message_hash(message)
        message_id = message.get("messageId")
        
        with self._lock:
            if principal not in self.tasks_by_user:
                self.tasks_by_user[principal] = {}

            key = (principal, msg_hash)
            if key in self.idempotency_map:
                existing_task_id = self.idempotency_map[key]
                return self.tasks_by_user[principal][existing_task_id], True, False

            # Check semantic content conflict for reused messageIds with different payload
            for t_id, t_data in self.tasks_by_user[principal].items():
                for hist_msg in t_data.get("history", []):
                    if hist_msg.get("messageId") == message_id:
                        existing_hash = self.compute_message_hash(hist_msg)
                        if existing_hash != msg_hash:
                            return None, False, True # Conflict (409 IDEMPOTENCY_CONFLICT)

            new_task = create_task_fn()
            task_id = new_task["id"]
            
            self.tasks_by_user[principal][task_id] = new_task
            self.idempotency_map[key] = task_id
            self.task_owner[task_id] = principal
            return new_task, False, False

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
