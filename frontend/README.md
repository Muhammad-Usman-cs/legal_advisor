# Legal Advisor — Frontend

The React + Vite chat UI for the Pakistani Law chatbot. This document explains every file, every React concept used, and the decisions behind each choice.

---

## Table of Contents

1. [Tech Stack & Why](#tech-stack--why)
2. [File Structure](#file-structure)
3. [How Data Flows Through the App](#how-data-flows-through-the-app)
4. [File Guide](#file-guide)
   - [vite.config.js](#viteconfigjs--dev-server--proxy)
   - [src/main.jsx](#srcmainjsx--entry-point)
   - [src/App.jsx](#srcappjsx--routing)
   - [src/api/client.js](#srcapiclientjs--backend-communication)
   - [src/context/AuthContext.jsx](#srccontextauthcontextjsx--global-auth-state)
   - [src/pages/LoginPage.jsx](#srcpagesloginpagejsx)
   - [src/pages/RegisterPage.jsx](#srcpagesregisterpagejsx)
   - [src/pages/ChatPage.jsx](#srcpageschatpagejsx--main-orchestrator)
   - [src/components/Sidebar.jsx](#srccomponentssidebarjsx)
   - [src/components/ChatWindow.jsx](#srccomponentschatwindowjsx)
   - [src/components/MessageBubble.jsx](#srccomponentsmessagebubblejsx)
   - [src/components/ChatInput.jsx](#srccomponentschatinputjsx)
   - [src/index.css](#srcindexcss--all-styles)
5. [Key React Concepts Used](#key-react-concepts-used)
6. [Auth Flow Step by Step](#auth-flow-step-by-step)
7. [Chat Flow Step by Step](#chat-flow-step-by-step)
8. [Running the Frontend](#running-the-frontend)

---

## Tech Stack & Why

| Technology | Version | Why |
|---|---|---|
| **React** | 19 | Component-based UI — each piece of the chat UI is an isolated, reusable component |
| **Vite** | 8 | Extremely fast dev server with Hot Module Replacement (HMR) — saves are reflected instantly without a full reload |
| **react-router-dom** | 7 | Navigates between `/login`, `/register`, and `/` (chat) without reloading the page |
| **Native `fetch`** | built-in | No extra library needed — `fetch` handles all HTTP calls cleanly |
| **Plain CSS** | — | No UI library — keeps the bundle small and makes the styles easy to read and change |

---

## File Structure

```
frontend/
├── index.html                  # HTML shell — React mounts here
├── vite.config.js              # Dev server config + API proxy
├── package.json                # Dependencies and npm scripts
│
└── src/
    ├── main.jsx                # Entry point — renders <App /> into index.html
    ├── App.jsx                 # Router — maps URLs to pages
    ├── App.css                 # Empty — all styles live in index.css
    ├── index.css               # All CSS for the entire app
    │
    ├── api/
    │   └── client.js           # Every fetch() call to FastAPI lives here
    │
    ├── context/
    │   └── AuthContext.jsx     # Global auth state (token, user, login, logout)
    │
    ├── pages/
    │   ├── LoginPage.jsx       # /login route
    │   ├── RegisterPage.jsx    # /register route
    │   └── ChatPage.jsx        # / route — owns all chat state
    │
    └── components/
        ├── Sidebar.jsx         # Left panel: conversation list + new chat + logout
        ├── ChatWindow.jsx      # Scrolling message list
        ├── MessageBubble.jsx   # One message (user or AI)
        └── ChatInput.jsx       # Textarea + Send button
```

---

## How Data Flows Through the App

Understanding how data moves is more important than memorising the code.

```
localStorage
    │
    │  token is saved here on login, read back on refresh
    ▼
AuthContext
    │  provides { token, user, login(), logout() }
    │  to every component without prop drilling
    ▼
App.jsx (Router)
    ├── /login      → LoginPage
    ├── /register   → RegisterPage
    └── /           → ProtectedRoute → ChatPage
                              │
                    owns all chat state:
                    conversations, activeConvId,
                    messages, isLoading, error
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           Sidebar       ChatWindow       ChatInput
        (display only)  (display only)  (calls onSend)
                                              │
                                    ChatPage.sendMessage()
                                              │
                                       api.sendMessage()
                                              │
                                       FastAPI backend
                                              │
                                       reply received
                                              │
                                    messages state updated
                                              │
                                    ChatWindow re-renders
```

---

## File Guide

---

### `vite.config.js` — Dev Server & Proxy

```js
server: {
  proxy: {
    '/auth': 'http://localhost:8000',
    '/chat': 'http://localhost:8000',
  }
}
```

**What the proxy does:** React runs on port 5173. FastAPI runs on port 8000. Without the proxy, `fetch('/auth/login')` would look for that path on port 5173 and find nothing.

The proxy intercepts any request starting with `/auth` or `/chat` and forwards it to FastAPI. This means every call in `client.js` uses a short relative path like `/auth/login` instead of `http://localhost:8000/auth/login`.

**Why this matters:** When you deploy to production, you only change the proxy target in one place, not every fetch call.

---

### `src/main.jsx` — Entry Point

```jsx
createRoot(document.getElementById('root')).render(
  <StrictMode><App /></StrictMode>
)
```

The single starting point of the whole app. `createRoot` mounts React onto the `<div id="root">` in `index.html`. `StrictMode` runs components twice in development to catch bugs early — it has no effect in the production build.

---

### `src/App.jsx` — Routing

```jsx
<AuthProvider>
  <BrowserRouter>
    <Routes>
      <Route path="/login"    element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/" element={
        <ProtectedRoute><ChatPage /></ProtectedRoute>
      } />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </BrowserRouter>
</AuthProvider>
```

**`AuthProvider` wraps everything** — that is what makes the token available to any page or component without passing it as a prop.

**`BrowserRouter`** enables client-side navigation. Clicking a link or calling `navigate('/')` changes the URL without a full page reload.

**`Routes` + `Route`** — React Router reads the current URL and renders only the matching component.

**`ProtectedRoute`:**
```jsx
function ProtectedRoute({ children }) {
  const { token } = useAuth()
  return token ? children : <Navigate to="/login" replace />
}
```
If there is no token, the user is sent to `/login` before the chat page ever renders. This is the frontend equivalent of the `get_current_user` dependency on the backend.

**`path="*"`** is a catch-all — any unrecognised URL (e.g. `/dashboard`) redirects to `/`.

---

### `src/api/client.js` — Backend Communication

All `fetch()` calls are centralised here. No component ever calls `fetch()` directly. This gives you one file to update if the API changes.

```js
async function handleResponse(res) {
  if (res.status === 204) return null          // DELETE returns no body
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Something went wrong')
  return data
}
```

Every exported function passes its response through `handleResponse`. This means:
- HTTP errors (400, 401, 422, 500) automatically throw a JavaScript `Error` with the backend's `detail` message
- Components only need a `try/catch` — they never parse responses themselves
- `204 No Content` (what DELETE returns) is handled cleanly

**Login sends form data, not JSON:**
```js
export function login(username, password) {
  const body = new URLSearchParams({ username, password })
  return handleResponse(fetch('/auth/login', { method: 'POST', body }))
}
```
FastAPI's `OAuth2PasswordRequestForm` expects `application/x-www-form-urlencoded`, not JSON. `URLSearchParams` creates the correct encoding. All other endpoints use `JSON.stringify`.

**The auth header:**
```js
function authHeaders(token) {
  return { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' }
}
```
`Authorization: Bearer <token>` is how FastAPI's `OAuth2PasswordBearer` identifies who is making the request. Every protected endpoint requires this header.

---

### `src/context/AuthContext.jsx` — Global Auth State

**The problem it solves:** The `token` is needed by `ChatPage` and every API call. Without Context, you would pass it as a prop from `App` → `ChatPage` → `Sidebar` → every component. Context lets any component read it directly.

```jsx
const [token, setToken] = useState(() => localStorage.getItem('token'))
```

The function `() => localStorage.getItem('token')` is a **lazy initializer** — it runs only once on mount, reading a previously saved token so the user stays logged in after a page refresh.

```jsx
useEffect(() => {
  if (token && !user) {
    api.getMe(token)
      .then(setUser)
      .catch(() => logout())
  }
}, [token])
```

On first load, if a token is found in localStorage, `GET /auth/me` is called to verify it is still valid and load the user's profile. If the call fails (token expired), `logout()` clears everything automatically.

**`login(newToken)`** — saves to localStorage (survives page refresh) and React state (triggers re-render immediately).

**`logout()`** — clears both. Any component that reads `token` re-renders, and `ProtectedRoute` redirects to `/login`.

---

### `src/pages/LoginPage.jsx`

A controlled form — React state tracks every character the user types.

```jsx
const [username, setUsername] = useState('')
<input value={username} onChange={e => setUsername(e.target.value)} />
```

`value={username}` displays the state. `onChange` updates the state on every keystroke. At submit time, `username` is immediately available — no need to read from the DOM.

**Submit flow:**
```
handleSubmit()
  → clear error
  → setLoading(true) — disables button
  → api.login(username, password) — POST /auth/login
      success → login(token) → navigate('/')
      error   → setError(err.message) — shows red box
  → setLoading(false) — re-enables button
```

---

### `src/pages/RegisterPage.jsx`

Uses a single state object instead of three `useState` calls:

```jsx
const [form, setForm] = useState({ username: '', email: '', password: '' })

function handleChange(e) {
  setForm(prev => ({ ...prev, [e.target.name]: e.target.value }))
}
```

`[e.target.name]` is a **computed property key** — the `name` attribute on each `<input>` determines which key in `form` gets updated. `<input name="email">` updates `form.email`. One handler for all three fields.

After a successful registration, it navigates to `/login` rather than auto-logging in. This is intentional — the user sees the login step, which is simpler to implement and teaches the flow.

---

### `src/pages/ChatPage.jsx` — Main Orchestrator

The most important file in the frontend. It owns all the chat state and passes pieces down to child components.

**Why own state here?** The sidebar needs to know which conversation is active. The chat window needs the messages. The input needs to know if the AI is still replying. All of these are connected. Putting state in the common parent and passing it down keeps everything in sync — this pattern is called **lifting state up**.

**Optimistic update in `sendMessage()`:**
```jsx
// 1. Show message instantly — do not wait for the server
const tempId = Date.now()
setMessages(prev => [...prev, { id: tempId, role: 'user', content: text }])
setIsLoading(true)

try {
  const data = await api.sendMessage(token, text, activeConvId)
  // 2. Append the real AI reply
  setMessages(prev => [...prev, { id: tempId + 1, role: 'assistant', content: data.reply }])
} catch (err) {
  // 3. Call failed — remove the message so the UI doesn't show a dead entry
  setMessages(prev => prev.filter(m => m.id !== tempId))
  setError(err.message)
}
```

**Why optimistic updates?** The Groq API call takes 2–5 seconds. Showing the user's message instantly makes the app feel responsive. If the call fails, the message is removed and an error is shown.

**Auto-creating conversations:** When `conversation_id` is `null`, the backend creates a new conversation and returns its ID. `ChatPage` saves this ID so all follow-up messages go to the same conversation.

---

### `src/components/Sidebar.jsx`

A **stateless (presentational) component** — it has no `useState`. It receives all its data and callbacks as props and calls them when the user acts. All logic lives in `ChatPage`.

```jsx
function handleDelete(e, id) {
  e.stopPropagation()    // stop the click bubbling up to the conversation's onClick
  if (confirm('Delete this conversation?')) onDelete(id)
}
```

`e.stopPropagation()` is important here. Without it, clicking the Delete button would also trigger the parent `div`'s `onClick` (which loads the conversation). Stopping propagation keeps the two actions separate.

---

### `src/components/ChatWindow.jsx`

```jsx
const bottomRef = useRef(null)

useEffect(() => {
  bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
}, [messages, isLoading])
```

**`useRef`** holds a reference to the invisible `<div>` at the very bottom of the message list. **`useEffect`** fires after every render where `messages` or `isLoading` changed. Together they scroll the chat down whenever a new message arrives.

`?.` (optional chaining) safely does nothing if `bottomRef.current` is null on the very first render.

When there are no messages and `isLoading` is false, the component shows a **welcome screen** instead of an empty box, giving the user instructions.

---

### `src/components/MessageBubble.jsx`

```jsx
<div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
```

One component, two visual appearances based on the `role` prop. The CSS class changes, and the stylesheet handles the rest — purple bubble on the right for the user, white bubble on the left for the AI.

`white-space: pre-wrap` in CSS is the key style for AI messages. Groq structures legal answers with newlines and dashes. `pre-wrap` preserves those newlines so the response reads as formatted text rather than one long line.

---

### `src/components/ChatInput.jsx`

```jsx
function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) handleSubmit(e)
}
```

**Enter sends, Shift+Enter inserts a newline.** This is the standard chat-app behaviour. A `<textarea>` is used instead of `<input type="text">` to support multi-line questions naturally.

```jsx
<button disabled={disabled || !text.trim()}>
```

The Send button is disabled when the AI is still replying (`disabled`) or the textarea is empty (`!text.trim()`). This prevents duplicate submissions and empty messages with no extra validation code.

---

### `src/index.css` — All Styles

All CSS lives in one file. No CSS Modules, no Tailwind — plain CSS makes it easy to see exactly what each class does.

**Core layout — flexbox:**
```css
/* Sidebar and main area sit side by side */
.chat-layout {
  display: flex;
  height: 100vh;
}

/* Main area fills all remaining width after the sidebar */
.main-area {
  flex: 1;              /* grow to fill available space */
  display: flex;
  flex-direction: column;
}

/* Messages fill all vertical space; input stays pinned at the bottom */
.chat-window {
  flex: 1;
  overflow-y: auto;     /* scroll when messages overflow */
}
```

**Message bubble specificity:**
```css
.message-bubble { ... }                     /* shared: padding, border-radius */
.message-user .message-bubble { ... }      /* purple override */
.message-assistant .message-bubble { ... } /* white override */
```

The two-class selector is more specific than the one-class selector, so it wins — only the needed properties are overridden.

**Animated loading dots:**
```css
.dot { animation: bounce 1.2s infinite ease-in-out; }
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; }
  40%           { transform: scale(1);   opacity: 1;   }
}
```

Three identical dots with staggered `animation-delay` create the wave effect. Each dot runs the same animation but starts slightly later.

---

## Key React Concepts Used

### `useState` — local component state
```jsx
const [messages, setMessages] = useState([])
// messages → current value
// setMessages → call to update it and trigger a re-render
```

### `useEffect` — side effects after render
```jsx
useEffect(() => { loadConversations() }, [])        // once on mount
useEffect(() => { scrollToBottom() }, [messages])   // every time messages changes
```

### `useRef` — reference a DOM element
```jsx
const bottomRef = useRef(null)
<div ref={bottomRef} />
bottomRef.current.scrollIntoView()  // directly access the DOM node
```

### `useContext` — read from a Context
```jsx
const { token, logout } = useAuth()  // works inside any component under <AuthProvider>
```

### Props — data parent → child
```jsx
// Parent:
<Sidebar conversations={conversations} onNew={newChat} />

// Child:
export default function Sidebar({ conversations, onNew }) { ... }
```

### Lifting state up
State that multiple components need lives in their closest common parent (`ChatPage`), and is passed down as props. Components never share state directly with siblings.

---

## Auth Flow Step by Step

**First visit (no stored token):**
```
User visits /
→ ProtectedRoute reads token from AuthContext → null
→ Redirected to /login
→ User fills form → Submit
→ api.login() → POST /auth/login → JWT received
→ login(token) → saved to localStorage + state
→ navigate('/') → ProtectedRoute finds token → ChatPage renders
```

**Returning visit (token in localStorage):**
```
User opens app
→ AuthContext useState reads token from localStorage → found
→ ChatPage renders immediately (no login screen)
→ AuthContext useEffect fires → api.getMe(token) → GET /auth/me
    valid   → user profile loaded into state
    expired → logout() → localStorage cleared → redirect to /login
```

**Logout:**
```
User clicks Logout → logout() called
→ localStorage cleared, token state = null
→ ProtectedRoute re-checks → no token → redirect to /login
```

---

## Chat Flow Step by Step

**New conversation (first message):**
```
User types + presses Enter
→ ChatInput calls onSend(text)
→ ChatPage.sendMessage(text):
    1. User message added to state instantly (optimistic update)
    2. isLoading = true → loading dots appear
    3. api.sendMessage(token, text, null)  ← conversation_id is null
    4. FastAPI creates Conversation, calls Groq, saves messages
    5. Response: { reply: "...", conversation_id: 3 }
    6. AI reply added to state
    7. activeConvId set to 3
    8. Sidebar refreshes → new conversation appears with auto-title
    9. isLoading = false
```

**Follow-up message (same conversation):**
```
Same as above, but conversation_id = 3 is passed to the API
FastAPI fetches last 20 messages for memory, appends new ones
```

**Loading an old conversation:**
```
User clicks conversation in sidebar
→ selectConversation(id) called
→ api.getConversation(token, id) → GET /chat/conversations/3
→ Response contains full message history
→ messages state updated → ChatWindow re-renders with history
```

---

## Running the Frontend

```bash
cd frontend

# Install dependencies (only needed once)
npm install

# Start the dev server
npm run dev
```

The app is at `http://localhost:5173`. The backend must also be running:

```bash
# In a separate terminal
cd backend
venv\Scripts\activate
uvicorn main:app --reload
```

| Command | What it does |
|---|---|
| `npm run dev` | Start dev server with live reload |
| `npm run build` | Build optimised production bundle into `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | Check code with ESLint |
