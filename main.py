import uuid
from datetime import datetime
from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from storage import TaskStore
from agent import InvoiceAgentCore

app = FastAPI(title="A2A Invoice Agent", version="1.0.0")
store = TaskStore()
agent_core = InvoiceAgentCore()

@app.middleware("http")
async def protocol_and_auth_middleware(request: Request, call_next):
    path = request.url.path

    # Allow public Agent Card discovery
    if path == "/.well-known/agent-card.json" and request.method == "GET":
        return await call_next(request)

    # Validate A2A-Version header
    a2a_version = request.headers.get("A2A-Version")
    if a2a_version and a2a_version != "1.0":
        return JSONResponse(status_code=400, content={"error": "Invalid or unsupported A2A-Version header. Expected '1.0'."})

    # Validate Media Types for state-changing routes
    content_type = request.headers.get("Content-Type", "")
    if request.method in ["POST", "PUT"] and path != "/.well-known/agent-card.json":
        if "application/a2a+json" not in content_type and "application/json" not in content_type:
            return JSONResponse(status_code=400, content={"error": "Missing or invalid media type. Expected application/a2a+json."})

    # Validate Bearer Token Authentication
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Missing or invalid Bearer authentication token."})
    
    token = auth_header.split(" ")[1].strip()
    if not token:
        return JSONResponse(status_code=403, content={"error": "Forbidden: Empty bearer token."})
    
    request.state.principal = token
    return await call_next(request)


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
    config = body.get("configuration", {})
    
    message_id = message.get("messageId")
    if not message_id:
        raise HTTPException(status_code=400, content="Missing messageId")

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
            raise HTTPException(status_code=404, content="Task not found or unauthorized")
        
        if task["status"]["state"] == "TASK_STATE_COMPLETED":
            return {"task": task}

        # Validate continuation match against stored proposal
        proposals_artifact = next((p for p in task["artifacts"] if p["mediaType"] == "application/vnd.ga5.invoice-action-proposals+json"), None)
        if not proposals_artifact:
            raise HTTPException(status_code=400, content="No active proposal found for continuation")
        
        stored_proposals = proposals_artifact["data"].get("proposals", [])
        incoming_results = result_data.get("results", [])

        # Validate match
        for res in incoming_results:
            match = next((p for p in stored_proposals if p["packageId"] == res["packageId"] and p["actionId"] == res["actionId"] and p["action"] == res["action"]), None)
            if not match:
                raise HTTPException(status_code=400, content="Continuation mismatch with stored proposal")

        # Build executions array for accepted outcomes only
        executions = []
        for res in incoming_results:
            if res.get("outcome") == "ACCEPTED":
                match = next((p for p in stored_proposals if p["packageId"] == res["packageId"] and p["actionId"] == res["actionId"]), None)
                if match:
                    executions.append({
                        "packageId": res["packageId"],
                        "actionId": res["actionId"],
                        "action": res["action"],
                        "receiptNonce": res.get("receiptNonce"),
                        "facts": match["facts"],
                        "evidenceRefs": match["evidenceRefs"]
                    })

        def update_to_completed(t):
            t["history"].append(message)
            t["status"] = {"state": "TASK_STATE_COMPLETED"}
            t["artifacts"].append({
                "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
                "data": {
                    "batchId": result_data.get("batchId"),
                    "executions": executions
                }
            })
            return t

        updated_task = store.update_task(principal, task_id, update_to_completed)
        return {"task": updated_task}

    # Initial batch message handling with idempotency
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

        # Call AI reasoning layer
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
        return {"task": task}
    except Exception as e:
        return JSONResponse(status_code=409, content={"error": "IDEMPOTENCY_CONFLICT", "details": str(e)})


@app.get("/a2a/tasks/{id}")
async def get_task(id: str, request: Request):
    principal = request.state.principal
    task = store.get_task_by_id(principal, id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/a2a/tasks")
async def list_tasks(request: Request):
    principal = request.state.principal
    tasks = store.list_tasks(principal)
    return {"tasks": tasks}


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
            raise HTTPException(status_code=404, detail="Task not found or unauthorized")
        return updated
    except ValueError as ve:
        # Handle cancel vs result race condition rules
        return JSONResponse(status_code=409, content={"error": "CONFLICT", "message": str(ve)})
