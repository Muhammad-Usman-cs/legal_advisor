// ---------------------------------------------------------------------------
// API CLIENT
// All calls to the FastAPI backend live here. Components never call fetch()
// directly — they import functions from this file instead.
//
// Because vite.config.js proxies /auth and /chat to localhost:8000, we use
// relative paths (no http://localhost:8000 prefix needed).
// ---------------------------------------------------------------------------

async function handleResponse(resPromise) {
  const res = await resPromise
  if (res.status === 204) return null          // DELETE returns no body
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Something went wrong')
  return data
}

function authHeaders(token) {
  return {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  }
}

// --- Auth ---

export function register(username, email, password) {
  return handleResponse(fetch('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password }),
  }))
}

export function login(username, password) {
  // FastAPI's OAuth2PasswordRequestForm expects form data, NOT JSON
  const body = new URLSearchParams({ username, password })
  return handleResponse(fetch('/auth/login', {
    method: 'POST',
    body,
  }))
}

export function getMe(token) {
  return handleResponse(fetch('/auth/me', {
    headers: authHeaders(token),
  }))
}

// --- Chat ---

export function sendMessage(token, message, conversationId = null) {
  return handleResponse(fetch('/chat/message', {
    method: 'POST',
    headers: authHeaders(token),
    body: JSON.stringify({ message, conversation_id: conversationId }),
  }))
}

export function getConversations(token) {
  return handleResponse(fetch('/chat/conversations', {
    headers: authHeaders(token),
  }))
}

export function getConversation(token, id) {
  return handleResponse(fetch(`/chat/conversations/${id}`, {
    headers: authHeaders(token),
  }))
}

export function deleteConversation(token, id) {
  return handleResponse(fetch(`/chat/conversations/${id}`, {
    method: 'DELETE',
    headers: authHeaders(token),
  }))
}
