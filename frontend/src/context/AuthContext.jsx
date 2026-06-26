import { createContext, useContext, useState, useEffect } from 'react'
import * as api from '../api/client'

// ---------------------------------------------------------------------------
// AuthContext makes the logged-in user and token available to any component
// in the tree without passing props through every level.
//
// Usage in any component:
//   const { token, user, login, logout } = useAuth()
// ---------------------------------------------------------------------------

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  // Read token from localStorage on first load so users stay logged in
  // after a page refresh.
  const [token, setToken] = useState(() => localStorage.getItem('token'))
  const [user, setUser] = useState(null)

  // When the app loads with a stored token, verify it is still valid
  // by calling GET /auth/me. If the token expired, log out silently.
  useEffect(() => {
    if (token && !user) {
      api.getMe(token)
        .then(setUser)
        .catch(() => logout())
    }
  }, [token])

  function login(newToken) {
    localStorage.setItem('token', newToken)
    setToken(newToken)
  }

  function logout() {
    localStorage.removeItem('token')
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ token, user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
