import { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import { useAuth } from '../context/AuthContext'
import Sidebar from '../components/Sidebar'
import ChatWindow from '../components/ChatWindow'
import ChatInput from '../components/ChatInput'
import * as api from '../api/client'

export default function ChatPage() {
  const { token, user, logout } = useAuth()

  const [conversations, setConversations] = useState([])
  const [activeConvId, setActiveConvId] = useState(null)
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  // Load the conversation list when the page first mounts
  useEffect(() => {
    loadConversations()
  }, [])

  async function loadConversations() {
    try {
      const data = await api.getConversations(token)
      setConversations(data)
    } catch (err) {
      toast.error(err.message)
    }
  }

  async function selectConversation(id) {
    setActiveConvId(id)
    try {
      const data = await api.getConversation(token, id)
      setMessages(data.messages)
    } catch (err) {
      toast.error(err.message)
    }
  }

  function newChat() {
    setActiveConvId(null)
    setMessages([])
  }

  async function sendMessage(text) {
    const tempId = Date.now()
    const userMsg = { id: tempId, role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    try {
      const data = await api.sendMessage(token, text, activeConvId)
      const aiMsg = { id: tempId + 1, role: 'assistant', content: data.reply }
      setMessages(prev => [...prev, aiMsg])
      if (!activeConvId) {
        setActiveConvId(data.conversation_id)
      }
      loadConversations()
    } catch (err) {
      setMessages(prev => prev.filter(m => m.id !== tempId))
      toast.error(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  async function deleteConversation(id) {
    try {
      await api.deleteConversation(token, id)
      setConversations(prev => prev.filter(c => c.id !== id))
      if (activeConvId === id) newChat()
      toast.success('Conversation deleted.')
    } catch (err) {
      toast.error(err.message)
    }
  }

  return (
    <div className="chat-layout">
      <Sidebar
        conversations={conversations}
        activeConvId={activeConvId}
        onSelect={selectConversation}
        onNew={newChat}
        onDelete={deleteConversation}
        user={user}
        onLogout={logout}
      />

      <div className="main-area">
        <ChatWindow messages={messages} isLoading={isLoading} />
        <ChatInput onSend={sendMessage} disabled={isLoading} />
      </div>
    </div>
  )
}
