import uuid
from datetime import datetime
from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from storage import TaskStore
from agent import InvoiceAgentCore

app = FastAPI(title="A2A Invoice Agent", version="1.0.0")
store = TaskStore()
agent_core = InvoiceAgentCore()

def a2a_json_response(content: dict, status_code: int = 200) -> Response:
    return JSONResponse(
        status_code=status_code,
        content=content,
        media_type="application/a2a+json"
    )

@app.middleware("http")
async def protocol_and_auth_middleware(request: Request, call_next):
    path = request.url.path

    # Allow public Agent Card discovery
    if path == "/.well-known/agent-card.json" and request.method == "GET":
        response = await call_next(request)
        response.headers["Content-Type"] = "application/json"
        return response

    # Validate A2A-Version header
    a2a_version = request.headers.get("A2A-Version")
    if a2a_version and a2a_version != "1.0":
        return a2a_json_response({"error": "Invalid or unsupported A2A-Version header. Expected '1.0'."}, status_code=400)

    # Validate Bearer Token Authentication
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return a2a_json_response({"error": "Missing or invalid Bearer authentication token."}, status_code=401)
    
    token = auth_header.split(" ")[1].strip()
    if not token:
        return a2a_json_response({"error": "Forbidden: Empty bearer token."}, status_code=403)
    
    request.state.principal = token
    response = await call_next(request)
    
    # Force application/a2a+json for all A2A responses
    response.headers["Content-Type"] = "application/a2a+json"
    return response


@app.get("/.well-known/agent-card.json")
async def get_agent_card(request: Request):
    base_url = str(request.base_url).rstrip("/") + "/a2a"
    return {
        "name": "Enterprise Invoice Action Agent",
        "description": "Autonomous A2A 1.0 agent analyzing invoice packages and orchestrating financial actions.",
        "version": "1.0.0",
        "capabilities": {
            "batchProcessing": True,
            "idempotencySupported": True
        },
        "supportedInterfaces": [
            {
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
                "url": base_url
            }
        ],
        "defaultInputModes": [
            "application/vnd.ga5.invoice-claim-batch+json",
            "application/vnd.ga5.invoice-action-results+json"
        ],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ],
        "skills": [
            {
                "name": "invoice_action_agent",
                "description": "Evaluates invoice claim packages to settle, approve, hold, reject, or open exceptions.",
                "tags": ["finance", "invoices", "audit", "a2a"]
            }
        ]
    }


@app.post("/a2a/message:send")
async def send_message(request: Request):
    principal = request.state.principal
    body = await request.json()
    message = body.get("message", {})
    
    message_id = message.get("messageId")
    if not message_id:
        return a2a_json_response({"error": "Missing messageId"}, status_code=400)

    parts = message.get("parts", [])
    
    # Check if this is a result continuation
    is_result_continuation = False
    result_data = None
    for part in parts:
        if part.get("mediaType") == "application/vnd.ga5.invoice-action-results+json":
            is_result_continuation = True
            result_data = part.get("data", {})

    if is_result_continuation:
        task_id = message.get("taskId")
        task = store.get_task_by_id(principal, task_id)
        if not task:
            return a2a_json_response({"error": "Task not found or unauthorized"}, status_code=404)
        
        if task["status"]["state"] == "TASK_STATE_COMPLETED":
            return a2a_json_response({"task": task})

        proposals_artifact = next((p for p in task["artifacts"] if p["mediaType"] == "application/vnd.ga5.invoice-action-proposals+json"), None)
        if not proposals_artifact:
            return a2a_json_response({"error": "No active proposal found for continuation"}, status_code=400)
        
        stored_proposals = proposals_artifact["data"].get("proposals", [])
        incoming_results = result_data.get("results", [])

        # Resilient matching supporting optional/missing actionId in results
        validated_executions = []
        for res in incoming_results:
            pkg_id = res.get("packageId")
            act_id = res.get("actionId")
            action_type = res.get("action")

            # Try exact match first, then fallback to packageId + action type matching
            match = next((
                p for p in stored_proposals 
                if p["packageId"] == pkg_id and 
                (not act_id or p.get("actionId") == act_id) and 
                (not action_type or p.get("action") == action_type)
            ), None)

            if not match:
                return a2a_json_response({"error": "Continuation mismatch with stored proposal"}, status_code=400)

            if res.get("outcome") == "ACCEPTED":
                validated_executions.append({
                    "packageId": pkg_id,
                    "actionId": match.get("actionId"),
                    "action": match.get("action"),
                    "receiptNonce": res.get("receiptNonce", "nonce-missing"),
                    "facts": match["facts"],
                    "evidenceRefs": match["evidenceRefs"]
                })

        def update_to_completed(t):
            t["history"].append(message)
            t["status"] = {"state": "TASK_STATE_COMPLETED"}
            t["artifacts"].append({
                "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
                "data": {
                    "batchId": result_data.get("batchId", "batch-default"),
                    "executions": validated_executions
                }
            })
            return t

        updated_task = store.update_task(principal, task_id, update_to_completed)
        return a2a_json_response({"task": updated_task})

    def create_new_task():
        task_id = f"task-{uuid.uuid4()}"
        context_id = f"ctx-{uuid.uuid4()}"
        batch_id = "unknown-batch"
        packages = []
        
        for part in parts:
            if part.get("mediaType") == "application/vnd.ga5.invoice-claim-batch+json":
                batch_data = part.get("data", {})
                batch_id = batch_data.get("batchId", "batch-default")
                packages = batch_data.get("packages", [])

        proposals = agent_core.process_batch(batch_id, packages)

        return {
            "id": task_id,
            "contextId": context_id,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED"},
            "history": [message],
            "artifacts": [
                {
                    "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                    "data": {
                        "batchId": batch_id,
                        "proposals": proposals
                    }
                }
            ]
        }

    try:
        task, reused = store.save_task_idempotent(principal, message, create_new_task)
        return a2a_json_response({"task": task})
    except Exception as e:
        return a2a_json_response({"error": "IDEMPOTENCY_CONFLICT", "details": str(e)}, status_code=409)

@app.get("/a2a/tasks/{id}")
async def get_task(id: str, request: Request):
    principal = request.state.principal
    task = store.get_task_by_id(principal, id)
    if not task:
        return a2a_json_response({"error": "Task not found"}, status_code=404)
    return a2a_json_response(task)


@app.get("/a2a/tasks")
async def list_tasks(request: Request):
    principal = request.state.principal
    tasks = store.list_tasks(principal)
    return a2a_json_response({"tasks": tasks})


@app.post("/a2a/tasks/{id}:cancel")
async def cancel_task(id: str, request: Request):
    principal = request.state.principal
    
    def apply_cancel(t):
        if t["status"]["state"] == "TASK_STATE_COMPLETED":
            raise ValueError("Cannot cancel completed task")
        if t["status"]["state"] == "TASK_STATE_CANCELED":
            return t
        t["status"] = {"state": "TASK_STATE_CANCELED"}
        return t

    try:
        updated = store.update_task(principal, id, apply_cancel)
        if not updated:
            return a2a_json_response({"error": "Task not found or unauthorized"}, status_code=404)
        return a2a_json_response(updated)
    except ValueError as ve:
        return a2a_json_response({"error": "CONFLICT", "message": str(ve)}, status_code=409)
